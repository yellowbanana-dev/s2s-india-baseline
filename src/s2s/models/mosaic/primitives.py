# Vendored from: Zhdanov, Lucic, Welling, van de Meent — Mosaic (ICML 2026)
# Original: https://github.com/maxxxzdn/mosaic  License: CC-BY-NC-4.0
# LOCAL MODIFICATIONS (see ADR-0002):
#   1. flash_attn import made optional; SDPA fallback added to block_attention().
#   2. `from ops import mosaic_sparse_attn` removed; sparse path raises ImportError
#      (never reached when sparse_every <= 0, which is our permanent config).
#   3. Import paths updated: `from utils` → `from s2s.models.mosaic.utils`.
#   4. MosaicBlock: added residual dropout (drop_rate from config, default 0.0).
"""
Primitive building blocks for the Mosaic transformer.

Components:
- Block-sparse attention with learned strategy weighting (local block, compressed,
  and top-k selection branches combined with a learned gate)
- Rotary positional embeddings (RoPE) for 2D lon/lat
- Cross-attention interpolation between grids
- HEALPix spatial up/downsampling
- Conditional SwiGLU FFN with noise injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.nn import RMSNorm
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

# LOCAL MODIFICATION: flash_attn is optional; fall back to torch SDPA.
try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    try:
        import flash_attn_interface as _fa
        _flash_attn_func = _fa.flash_attn_func
        _FLASH_ATTN_AVAILABLE = True
    except ImportError:
        _FLASH_ATTN_AVAILABLE = False

from s2s.models.mosaic.utils import get_healpix_grid, get_neighbors, rad_to_xyz


def block_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int):
    # LOCAL MODIFICATION: SDPA fallback when flash_attn is not installed.
    # q, k, v: (batch, seq, heads, head_dim) — flash_attn layout.
    if q.shape[2] != k.shape[2]:
        # MIN-3 (review 2026-07-14): the SDPA fallback has no GQA broadcast, so unequal head
        # counts would raise an opaque shape error (or diverge from flash_attn). Names gqa_ratio.
        raise ValueError(
            f"block_attention requires equal q/kv head counts (gqa_ratio=1); got q heads "
            f"{q.shape[2]} vs k heads {k.shape[2]}."
        )
    batch_size = q.shape[0]
    q, k, v = map(lambda x: rearrange(x, 'b (nb bs) h d -> (b nb) bs h d', bs=block_size), (q, k, v))
    if _FLASH_ATTN_AVAILABLE:
        o_ba = _flash_attn_func(q, k, v)
    else:
        # SDPA expects (batch, heads, seq, dim).
        q_, k_, v_ = [rearrange(t, 'b s h d -> b h s d') for t in (q, k, v)]
        o_ba = F.scaled_dot_product_attention(q_, k_, v_)
        o_ba = rearrange(o_ba, 'b h s d -> b s h d')
    return rearrange(o_ba, '(b nb) bs h d -> b (nb bs) h d', b=batch_size)


@torch.no_grad()
def attn_topk(q: torch.Tensor, k: torch.Tensor, block_count: int):
    Hq, Hk = q.shape[2], k.shape[2]
    G = Hq // Hk
    k = k.repeat_interleave(G, dim=2)

    scores = torch.matmul(
        rearrange(q, 'b t h d -> b h t d'),
        rearrange(k, 'b t h d -> b h d t')
    )

    if Hq != Hk:
        scores = reduce(scores, 'b (g h) t k -> b h t k', 'mean', g=G)

    scores = rearrange(scores, 'b h t k -> b t h k')
    top_indices = scores.topk(k=block_count, dim=-1, largest=True)[1]
    return top_indices


def _sdpa_cross(q, k, v):
    """Scaled dot-product attention, q attending to a (smaller) k/v set.

    Flash layout in/out: (b, s, h, d). Assumes q and k/v share head count
    (gqa_ratio=1); memory-efficient (SDPA / flash never materialises the score matrix).
    """
    if q.shape[2] != k.shape[2]:
        # MIN-3 (review 2026-07-14): SDPA would otherwise fail with an opaque shape error, or
        # silently differ from the flash_attn path. Names gqa_ratio so the cause is obvious.
        raise ValueError(
            f"_sdpa_cross requires equal q/kv head counts (gqa_ratio=1); got q heads "
            f"{q.shape[2]} vs k heads {k.shape[2]}."
        )
    q_, k_, v_ = [rearrange(t, 'b s h d -> b h s d') for t in (q, k, v)]
    o = F.scaled_dot_product_attention(q_, k_, v_)
    return rearrange(o, 'b h s d -> b s h d')


def selection_attention(q, k, v, query_block_size: int, sparse_block_size: int,
                        sparse_block_count: int):
    """Fine top-k SELECTION branch, per QUERY-BLOCK (ADR-0010).

    All queries in a block of `query_block_size` share ONE selected key-block set, chosen by
    scoring the block-mean query against the block-mean (compressed) keys. Each selected key
    block contributes its FINE (uncompressed) tokens. This is the NSA formulation, and it is
    what makes the branch tractable: per-TOKEN selection needs a
    (b, seq, h, k*bs, d) gather (~1e11 elements at nside=64) whereas per-BLOCK selection
    processes one query block at a time, bounded by (b, h, query_block_size, k*bs).

    Cost at nside=64 (seq=49152, qb=512, k=16, bs=64): 5.0e7 score-pairs vs 2.4e9 for dense
    global attention (48x cheaper) and 3.8e7 for the compressed branch -- i.e. affordable.

    q, k, v: (b, seq, h, d). Returns (b, seq, h, d). Assumes gqa_ratio=1 (guarded by caller).
    """
    b, seq, h, d = q.shape
    n_kv_blocks = seq // sparse_block_size
    n_sel = min(int(sparse_block_count), n_kv_blocks)
    if n_sel <= 0:
        raise ValueError(f"sparse_block_count must be >= 1 to use the selection branch")

    # Block-mean keys drive BOTH the selection scores and (in the caller) the compressed branch.
    kc = reduce(k, 'b (nb bs) h d -> b nb h d', 'mean', bs=sparse_block_size)
    # Block-mean queries: one representative per query block (the NSA sharing trick).
    qb = reduce(q, 'b (nq qs) h d -> b nq h d', 'mean', qs=query_block_size)

    # Selection scores: (b, n_qblocks, h, n_kv_blocks) -> top-n_sel key blocks per query block.
    scores = torch.einsum('bqhd,bkhd->bqhk', qb, kc) * (d ** -0.5)
    with torch.no_grad():
        top = scores.topk(k=n_sel, dim=-1, largest=True).indices  # (b, nq, h, n_sel)

    kb = rearrange(k, 'b (nb bs) h d -> b nb bs h d', bs=sparse_block_size)
    vb = rearrange(v, 'b (nb bs) h d -> b nb bs h d', bs=sparse_block_size)
    n_q_blocks = seq // query_block_size

    outs = []
    for i in range(n_q_blocks):
        q_i = q[:, i * query_block_size:(i + 1) * query_block_size]        # (b, qs, h, d)
        idx = top[:, i]                                                    # (b, h, n_sel)
        # Gather the selected FINE key/value blocks for this query block only.
        gi = idx.permute(0, 2, 1)[..., None, None]                         # (b, n_sel, h, 1, 1)
        gi_k = gi.expand(-1, -1, -1, sparse_block_size, d)                 # (b, n_sel, h, bs, d)
        kb_t = kb.permute(0, 1, 3, 2, 4)                                   # (b, nb, h, bs, d)
        vb_t = vb.permute(0, 1, 3, 2, 4)
        k_sel = torch.gather(kb_t, 1, gi_k)                                # (b, n_sel, h, bs, d)
        v_sel = torch.gather(vb_t, 1, gi_k)
        k_sel = rearrange(k_sel, 'b n h bs d -> b (n bs) h d')             # (b, n_sel*bs, h, d)
        v_sel = rearrange(v_sel, 'b n h bs d -> b (n bs) h d')
        outs.append(_sdpa_cross(q_i, k_sel, v_sel))                        # (b, qs, h, d)
    return torch.cat(outs, dim=1)


def mosaic_attn_func(
    q, k, v,
    weight_ba_cmp_slc,
    block_attn_size, sparse_block_size, sparse_block_count,
    block_attn_only, no_compression=False, selection=False,
):
    if q.shape[1] % block_attn_size != 0:
        raise ValueError(
            f"seq_len {q.shape[1]} not divisible by block_attn_size {block_attn_size}"
        )
    o_ba = block_attention(q, k, v, block_attn_size)

    if block_attn_only:
        return o_ba

    # LOCAL MODIFICATION (lever f / ADR-0007): PyTorch reference for the block-sparse
    # path, replacing the upstream Triton ops.py (never vendored). At high resolution
    # (nside=64 -> ~49k tokens) dense global attention is infeasible, so each query gets
    # cheap GLOBAL context from a COMPRESSED key/value set (block-mean-pool over
    # sparse_block_size), combined with the LOCAL block branch by the learned 3-way
    # strategy gate (weight_ba_cmp_slc).
    #
    # SELECTION BRANCH (corrected 2026-07-21, ADR-0010). The earlier claim that the fine
    # top-k branch is "memory-infeasible in pure PyTorch" was WRONG: it holds only for
    # PER-TOKEN selection (which is what the vendored attn_topk computes, ~1e11 elements at
    # nside=64). NSA selects per QUERY-BLOCK -- queries in a block share one key-block set --
    # which is bounded and ~48x cheaper than dense global attention. See selection_attention().
    # Enabled by selection=True (default False keeps the legacy placeholder, byte-identical).
    if no_compression:
        # MIN-2 (review 2026-07-14): the flag is accepted but the reference implementation has
        # no uncompressed global path, so honouring it silently would still compress. Fail
        # loudly instead. Config sets it false, so this is inert today.
        raise NotImplementedError(
            "no_compression=True is not supported by the lever-f PyTorch reference: the "
            "global branch is always the compressed (block-mean-pooled) one. Set "
            "no_compression=false, or use block_attn_only for a purely local model."
        )

    if q.shape[1] % sparse_block_size != 0:
        raise ValueError(
            f"seq_len {q.shape[1]} not divisible by sparse_block_size {sparse_block_size}"
        )
    kc = reduce(k, 'b (nb bs) h d -> b nb h d', 'mean', bs=sparse_block_size)
    vc = reduce(v, 'b (nb bs) h d -> b nb h d', 'mean', bs=sparse_block_size)
    o_cmp = _sdpa_cross(q, kc, vc)
    w = weight_ba_cmp_slc  # (n_slots, b, s, h, 1)
    if w.shape[0] == 2:
        # gate_slots=2 (ADR-0009): the duplicate selection slot is gone, so the gate is a
        # genuine local-vs-compressed choice and starts unbiased at 1/2 - 1/2.
        return w[0] * o_ba + w[1] * o_cmp
    if selection:
        # ADR-0010: the third slot is a REAL fine top-k branch, so the 3-way gate is genuine
        # (three distinct tensors) and the MAJ-1 2:1 init bias is gone -- fixed by making the
        # slot real rather than by deleting it (gate_slots=2).
        o_slc = selection_attention(
            q, k, v, block_attn_size, sparse_block_size, sparse_block_count
        )
    else:
        # Legacy 3-slot placeholder: selection duplicates compressed (MAJ-1), so the effective
        # compressed weight is w1+w2 and the model starts ~2:1 biased toward compressed.
        o_slc = o_cmp
    return w[0] * o_ba + w[1] * o_cmp + w[2] * o_slc


class cSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, noise_dim: int):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.act_fn = nn.SiLU()

        if noise_dim > 0:
            self.noise_bias = nn.Linear(noise_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, z: torch.Tensor = None):
        noise = self.noise_bias(z).unsqueeze(0) if z is not None else 0
        x1, x3 = self.w13(x).chunk(2, dim=-1)
        return self.w2(self.act_fn(x1 + noise) * x3)


class RoPE(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.theta = theta

    def initialize_rope(self, positions):
        base_freqs = 1. / (self.theta ** (torch.arange(0, self.dim // 2, 2).float() / (self.dim // 2)))
        lon_pos = torch.deg2rad(positions[:, 0:1])
        lat_pos = torch.deg2rad(positions[:, 1:2])
        lon_freqs = torch.matmul(lon_pos, base_freqs.unsqueeze(0))
        lat_freqs = torch.matmul(lat_pos, base_freqs.unsqueeze(0))
        freqs = torch.cat([lon_freqs, lat_freqs], dim=-1)
        self.register_buffer('cos_freqs', freqs.cos().contiguous(), persistent=True)
        self.register_buffer('sin_freqs', freqs.sin().contiguous(), persistent=True)

    @staticmethod
    def rotate_half(x):
        x = rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d r -> ... (d r)')

    def forward(self, x):
        cos = self.cos_freqs.unsqueeze(0).unsqueeze(2).repeat_interleave(2, dim=-1)
        sin = self.sin_freqs.unsqueeze(0).unsqueeze(2).repeat_interleave(2, dim=-1)
        return (x.float() * cos + self.rotate_half(x.float()) * sin).to(x.dtype)


class MosaicAttention(nn.Module):
    def __init__(self, config, block_attn_only: bool, no_compression: bool = False):
        super().__init__()
        self.block_attn_only = block_attn_only
        self.no_compression = no_compression
        self.block_attn_size = config.block_attn_size
        self.sparse_block_size = config.sparse_block_size
        self.sparse_block_count = config.sparse_block_count

        q_heads = config.num_heads
        gqa_ratio = config.gqa_ratio
        # MAJ-1 (review 2026-07-14): the lever-f PyTorch reference supports gqa_ratio=1 only.
        # Both the local block path (block_attention) and the compressed-global path
        # (_sdpa_cross) assume q and k/v share head count; GQA (kv_heads < q_heads) would hit
        # an opaque SDPA shape error under the (flash_attn-absent) fallback. Grouped-query
        # attention belongs to the deferred Triton selection kernel, not this reference.
        if gqa_ratio != 1:
            raise ValueError(
                f"MosaicAttention (ADR-0007 PyTorch reference) supports gqa_ratio=1 only; "
                f"got gqa_ratio={gqa_ratio}."
            )
        dim = config.dim
        qkv_compress_ratio = config.qkv_compress_ratio
        rope = config.rope
        rope_theta = config.rope_theta

        kv_heads = q_heads // gqa_ratio
        head_dim = int(dim // q_heads // qkv_compress_ratio)

        self.q_heads = q_heads
        self.kv_heads = kv_heads

        self.to_q = nn.Linear(dim, q_heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.to_o = nn.Linear(q_heads * head_dim, dim, bias=False)

        self.q_rope = RoPE(head_dim, rope_theta) if rope else None
        self.k_rope = RoPE(head_dim, rope_theta) if rope else None

        # Strategy-gate slots. 3 = legacy MAJ-1 placeholder layout (local, compressed,
        # selection) where the selection slot DUPLICATES compressed, so a uniform softmax
        # starts the model at 1/3 local + 2/3 compressed -- a structural bias toward the
        # deliberately low-variance mean-pooled branch (ADR-0007). 2 = duplicate slot removed
        # (local, compressed), restoring a 1/2-1/2 start. Default 3 keeps every existing
        # checkpoint loadable; 2 changes the gate layer shape and is a NEW experiment (ADR-0009).
        # ADR-0010: real fine top-k selection branch (default OFF -> byte-identical legacy
        # placeholder, and every existing checkpoint keeps loading).
        self.selection = bool(getattr(config, "selection", False))
        self.gate_slots = int(getattr(config, "gate_slots", 3))
        if self.selection and self.gate_slots != 3:
            raise ValueError(
                "selection=True requires gate_slots=3 (local, compressed, selection); "
                f"got gate_slots={self.gate_slots}. gate_slots=2 DELETES the selection slot, "
                "which is the alternative fix to the same MAJ-1 defect -- pick one."
            )
        if self.gate_slots not in (2, 3):
            raise ValueError(
                f"gate_slots must be 2 (selection slot removed) or 3 (legacy placeholder), "
                f"got {self.gate_slots}"
            )
        if block_attn_only:
            self.to_strategy_combine_mlp = None
        else:
            self.to_strategy_combine_mlp = nn.Linear(dim, self.gate_slots * q_heads, bias=False)

    def generate_strategy_weights(self, x):
        if self.block_attn_only:
            return [None, None, None]
        strategy_logits = self.to_strategy_combine_mlp(x)
        strategy_logits = rearrange(strategy_logits, 't b (h s) -> s b t h', h=self.q_heads)
        strategy_weights = torch.softmax(strategy_logits.float(), dim=0).type_as(x)
        strategy_weights = strategy_weights.unsqueeze(-1)
        return strategy_weights

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        strategy_weights = self.generate_strategy_weights(x)

        q = rearrange(q, 's b (h d) -> b s h d', h=self.q_heads)
        k = rearrange(k, 's b (h d) -> b s h d', h=self.kv_heads)
        v = rearrange(v, 's b (h d) -> b s h d', h=self.kv_heads)

        if self.q_rope is not None:
            q = self.q_rope(q)
            k = self.k_rope(k)

        output = mosaic_attn_func(
            q=q, k=k, v=v,
            weight_ba_cmp_slc=strategy_weights,
            block_attn_size=self.block_attn_size,
            sparse_block_size=self.sparse_block_size,
            sparse_block_count=self.sparse_block_count,
            block_attn_only=self.block_attn_only,
            no_compression=self.no_compression,
            selection=self.selection,
        )

        output = rearrange(output, 'b s h d -> s b (h d)')
        output = self.to_o(output)
        return output


class MosaicBlock(nn.Module):
    def __init__(self, config, block_attn_only: bool, no_compression: bool = False):
        super().__init__()
        dim = config.dim
        noise_dim = config.noise_dim
        mlp_ratio = config.mlp_ratio
        drop_rate = getattr(config, 'drop_rate', 0.0)

        self.attention = MosaicAttention(config, block_attn_only, no_compression)
        self.norm1 = RMSNorm(dim, elementwise_affine=config.rmsnorm_elementwise_affine)
        self.norm2 = RMSNorm(dim, elementwise_affine=config.rmsnorm_elementwise_affine)
        self.ffn = cSwiGLU(dim, int(dim * mlp_ratio), noise_dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, z: torch.Tensor = None):
        x = x + self.drop(self.attention(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x), z))
        return x


class CrossAttentionInterpolate(nn.Module):
    space_dim = 3

    def __init__(self, config):
        super().__init__()
        self.k_neighbors = config.k_neighbors

        dim = config.dim
        num_heads = config.num_heads

        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        # Target-dim chunk size for forward(), in ELEMENTS -- not bytes, so the actual
        # transient scales with dtype (x2 bf16, x4 fp32) (MIN-9). No-op at 5.625 deg.
        #
        # MAJ-4 (review 2026-07-14) -- WHAT CHUNKING DOES AND DOES NOT BOUND. Chunking bounds
        # the TRANSIENT working set only; it does NOT reduce TRAINING memory. Under autograd
        # every chunk's gathered k_c/v_c is saved for backward, so the total saved-activation
        # footprint is IDENTICAL chunked or not (~6.4 GB to-HEALPix + ~3.8 GB to-lonlat, bf16,
        # at B=4, M=8, k=8, h=8, d=16). Chunking caps only the ADDITIONAL transient
        # (~1 GB/chunk at 256M elements). At eval (M=1, b=4) the chunk size exceeds both target
        # sizes, so chunking is inactive there entirely. The earlier wording ("bounds the gather
        # so interpolation scales to high resolution") was true of inference, not training.
        # For the NEXT scale-up (more members, bigger batch, nside=128) the correct lever is
        # gradient checkpointing -- see interp_grad_checkpoint below.
        self.interp_chunk_budget_elems = 256_000_000

        # Opt-in gradient checkpointing around the per-chunk attend (MAJ-4). Default False =>
        # bit-identical behaviour and identical memory to before. When True, k_c/v_c are
        # RECOMPUTED in backward instead of saved, which is what actually bounds training
        # memory, at the cost of one extra forward per chunk.
        self.interp_grad_checkpoint = False

        self.kv_norm = RMSNorm(dim, elementwise_affine=config.rmsnorm_elementwise_affine)
        self.to_q = nn.Linear(self.space_dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.to_o = nn.Linear(dim, dim, bias=False)

        # Buffers populated by initialize_interpolation_scheme().
        self.register_buffer('neighbors', None)
        self.register_buffer('rel_pos', None)

    @torch.no_grad()
    def initialize_interpolation_scheme(self, pos_from_rad, pos_to_rad):
        neighbors_np = get_neighbors(pos_from_rad.cpu().numpy(), pos_to_rad.cpu().numpy(), k=self.k_neighbors)
        neighbors = torch.from_numpy(neighbors_np).long().to(pos_from_rad.device).contiguous()

        pos_to_xyz = rad_to_xyz(pos_to_rad)
        pos_from_xyz = rad_to_xyz(pos_from_rad)

        rel_pos_xyz = (pos_to_xyz.unsqueeze(1) - pos_from_xyz[neighbors]).contiguous()
        norm_rel_pos_xyz = torch.nn.functional.normalize(rel_pos_xyz, dim=-1).contiguous()

        self.register_buffer('neighbors', neighbors, persistent=True)
        self.register_buffer('rel_pos', norm_rel_pos_xyz, persistent=True)

    def forward(self, x_from: torch.Tensor):
        if self.neighbors is None or self.rel_pos is None:
            raise ValueError("Interpolation scheme not initialized.")

        q = self.to_q(self.rel_pos)
        q = rearrange(q, 's k (h d) -> s k 1 h d', h=self.num_heads)

        kv = self.to_kv(self.kv_norm(x_from))
        kv = rearrange(kv, 's b (n h d) -> n s b h d', h=self.num_heads, n=2)

        s_to = self.neighbors.shape[0]
        k_nb = self.neighbors.shape[1]
        b = kv.shape[2]
        # Each target pixel attends only over its own k neighbours (softmax on dim=1),
        # so the target dim is fully independent -> chunking over it is EXACT, not an
        # approximation. Chunk size auto-scales with batch to bound the TRANSIENT gather
        # (see the MAJ-4 note in __init__: this does NOT reduce saved-activation memory
        # under training; interp_grad_checkpoint is the lever that does).
        per_row = max(1, k_nb * b * self.num_heads * self.head_dim)
        chunk = max(1, int(self.interp_chunk_budget_elems) // per_row)

        def _attend(sl):
            q_c = q[sl]                                   # (cs, k, 1, h, d)
            k_c, v_c = kv[:, self.neighbors[sl]]          # each (cs, k, b, h, d)
            scores = (q_c * k_c).sum(dim=-1, keepdim=True) * self.scale
            w = torch.softmax(scores, dim=1, dtype=torch.float32).type_as(k_c)
            return (w * v_c).sum(dim=1)                   # (cs, b, h, d)

        def _run(sl):
            # MAJ-4: recompute the gather in backward instead of saving it, when opted in.
            if self.interp_grad_checkpoint and torch.is_grad_enabled():
                return _grad_checkpoint(_attend, sl, use_reentrant=False)
            return _attend(sl)

        if s_to <= chunk:
            out = _run(slice(0, s_to))
        else:
            out = torch.cat(
                [_run(slice(i, min(i + chunk, s_to))) for i in range(0, s_to, chunk)],
                dim=0,
            )

        out = rearrange(out, 's b h d -> s b (h d)')
        out = self.to_o(out)
        return out


class NoiseGenerator(nn.Module):
    def __init__(self, noise_dim: int, seed: int):
        super().__init__()
        self.seed = seed
        self.to_noise = nn.Linear(noise_dim, noise_dim, bias=False)
        self.generator = None

    def forward(self, num_samples: int, device: torch.device, dtype: torch.dtype):
        if self.generator is None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(self.seed)

        noise = torch.randn((num_samples, self.to_noise.in_features),
                            generator=self.generator, device=device, dtype=dtype)
        noise = self.to_noise(noise)
        return noise


class HEALPixDownsample(nn.Module):
    space_dim: int = 3

    def __init__(self, in_dim, out_dim, nside_before, nside_after,
                 rmsnorm_elementwise_affine=True):
        super().__init__()
        self.factor = (nside_before // nside_after) ** 2

        self.proj_x = nn.Linear(self.factor * in_dim, out_dim, bias=False)
        self.proj_pos = nn.Linear(self.factor * self.space_dim, out_dim, bias=False)
        self.norm = RMSNorm(out_dim, elementwise_affine=rmsnorm_elementwise_affine)

        hp_grid_fine_xyz = rad_to_xyz(torch.deg2rad(get_healpix_grid(nside_before)))
        hp_grid_coarse_xyz = rad_to_xyz(torch.deg2rad(get_healpix_grid(nside_after)))

        pos = rearrange(hp_grid_fine_xyz, '(n f) d -> n f d', f=self.factor)
        rel_pos = rearrange(pos - hp_grid_coarse_xyz[:, None], 'n f d -> n (f d)')
        rel_pos = (rel_pos - rel_pos.mean(dim=0, keepdim=True)) / (rel_pos.std(dim=0, keepdim=True) + 1e-6)

        self.register_buffer('rel_pos', rel_pos.contiguous(), persistent=True)

    def forward(self, x: torch.Tensor):
        x = rearrange(x, '(n f) b c -> n b (f c)', f=self.factor)
        x = self.proj_x(x) + self.proj_pos(self.rel_pos).unsqueeze(1)
        x = self.norm(x)
        return x


class HEALPixUpsample(nn.Module):
    space_dim: int = 3

    def __init__(self, in_dim, out_dim, nside_before, nside_after,
                 rmsnorm_elementwise_affine=True):
        super().__init__()
        self.factor = (nside_after // nside_before) ** 2

        self.proj_x = nn.Linear(in_dim, out_dim * self.factor, bias=False)
        self.proj_pos = nn.Linear(self.factor * self.space_dim, out_dim * self.factor, bias=False)
        self.norm = RMSNorm(out_dim, elementwise_affine=rmsnorm_elementwise_affine)

        hp_grid_fine_xyz = rad_to_xyz(torch.deg2rad(get_healpix_grid(nside_after)))
        hp_grid_coarse_xyz = rad_to_xyz(torch.deg2rad(get_healpix_grid(nside_before)))

        children_pos_reshaped = rearrange(hp_grid_fine_xyz, '(n f) d -> n f d', f=self.factor)
        rel_pos = rearrange(children_pos_reshaped - hp_grid_coarse_xyz[:, None], 'n f d -> n (f d)')
        rel_pos = (rel_pos - rel_pos.mean(dim=0, keepdim=True)) / (rel_pos.std(dim=0, keepdim=True) + 1e-6)

        self.register_buffer('rel_pos', rel_pos.contiguous(), persistent=True)

    def forward(self, x: torch.Tensor, shortcut: torch.Tensor):
        x = self.proj_x(x) + self.proj_pos(self.rel_pos).unsqueeze(1)
        x = rearrange(x, 'n b (f d) -> (n f) b d', f=self.factor)
        x = x + shortcut
        x = self.norm(x)
        return x
