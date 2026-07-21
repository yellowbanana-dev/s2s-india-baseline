"""Block-sparse attention PyTorch-reference guardrails (lever f / ADR-0007).

Torch tests (run on the cluster). Cover the sparse path that replaces the upstream
Triton ops.py: local block + compressed-global, combined by the 3-way strategy gate.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from s2s.models.mosaic.primitives import MosaicAttention


def _cfg(dim=32, num_heads=4, block_attn_size=16, sparse_block_size=8, sparse_block_count=2):
    return SimpleNamespace(
        dim=dim, num_heads=num_heads, gqa_ratio=1, qkv_compress_ratio=1,
        block_attn_size=block_attn_size, sparse_block_size=sparse_block_size,
        sparse_block_count=sparse_block_count, rope=False, rope_theta=10000,
        rmsnorm_elementwise_affine=True,
    )


def test_sparse_forward_shape_and_finite():
    attn = MosaicAttention(_cfg(), block_attn_only=False).eval()
    x = torch.randn(64, 2, 32)  # (seq, batch, dim); 64 divisible by block(16) and sparse(8)
    with torch.no_grad():
        o = attn(x)
    assert o.shape == (64, 2, 32)
    assert torch.isfinite(o).all()


def test_block_only_matches_dense_when_one_block():
    """block_attn_only with block_attn_size == seq == 1 block == full dense attention."""
    cfg = _cfg(block_attn_size=64)
    attn = MosaicAttention(cfg, block_attn_only=True).eval()
    x = torch.randn(64, 1, 32)
    with torch.no_grad():
        o = attn(x)
    assert o.shape == (64, 1, 32) and torch.isfinite(o).all()


def test_strategy_gate_is_convex_combination():
    """generate_strategy_weights softmaxes over the 3 branches -> weights sum to 1."""
    attn = MosaicAttention(_cfg(), block_attn_only=False).eval()
    x = torch.randn(64, 2, 32)
    w = attn.generate_strategy_weights(x)  # (3, b, t, h, 1)
    assert w.shape[0] == 3
    torch.testing.assert_close(w.sum(dim=0), torch.ones_like(w.sum(dim=0)), rtol=1e-4, atol=1e-4)


def test_sparse_requires_divisible_seq():
    # block_attn_size (48) divides seq (48) so block_attention passes; sparse_block_size
    # (32) does NOT divide 48 -> the sparse guard must raise a clean ValueError.
    attn = MosaicAttention(_cfg(block_attn_size=48, sparse_block_size=32), block_attn_only=False).eval()
    x = torch.randn(48, 1, 32)
    with pytest.raises(ValueError):
        attn(x)


def test_block_attn_size_divisibility_guard():
    # seq (50) not divisible by block_attn_size (16) -> clean ValueError, not raw einops.
    attn = MosaicAttention(_cfg(block_attn_size=16), block_attn_only=False).eval()
    x = torch.randn(50, 1, 32)
    with pytest.raises(ValueError):
        attn(x)


def test_interp_chunking_is_numerically_exact():
    """CrossAttentionInterpolate: chunking the target dim must equal the single-shot
    result exactly (each target pixel's softmax is over its own neighbours only)."""
    pytest.importorskip("sklearn")
    from s2s.models.mosaic.primitives import CrossAttentionInterpolate

    cfg = SimpleNamespace(k_neighbors=4, dim=32, num_heads=4, rmsnorm_elementwise_affine=True)
    interp = CrossAttentionInterpolate(cfg).eval()
    torch.manual_seed(0)
    pos_from = torch.rand(40, 2)   # (n_from, 2) lon/lat radians
    pos_to = torch.rand(50, 2)     # (n_to, 2)
    interp.initialize_interpolation_scheme(pos_from, pos_to)
    x = torch.randn(40, 3, 32)     # (n_from, batch, dim)

    with torch.no_grad():
        interp.interp_chunk_budget_elems = 10 ** 12   # single shot (no chunking)
        o_full = interp(x)
        interp.interp_chunk_budget_elems = 1          # force maximal chunking
        o_chunk = interp(x)
    assert o_full.shape == (50, 3, 32)
    torch.testing.assert_close(o_full, o_chunk, rtol=1e-5, atol=1e-5)


def test_interp_grad_checkpoint_matches_and_grads_agree():
    """MAJ-4: opt-in gradient checkpointing must be a pure memory/compute trade -- identical
    forward AND identical gradients. Default stays False (bit-identical to pre-MAJ-4)."""
    pytest.importorskip("sklearn")
    from s2s.models.mosaic.primitives import CrossAttentionInterpolate

    cfg = SimpleNamespace(k_neighbors=4, dim=32, num_heads=4, rmsnorm_elementwise_affine=True)
    interp = CrossAttentionInterpolate(cfg)
    assert interp.interp_grad_checkpoint is False      # default off => no behaviour change
    torch.manual_seed(0)
    pos_from, pos_to = torch.rand(40, 2), torch.rand(50, 2)
    interp.initialize_interpolation_scheme(pos_from, pos_to)

    # The SAME input for every configuration. (A previous version of this test built a fresh
    # random x inside run(), so it compared unrelated forward passes and failed with ~100% of
    # elements differing -- a test bug, not a kernel bug.)
    x0 = torch.randn(40, 3, 32)

    def run(use_ckpt, chunked):
        for p in interp.parameters():
            if p.grad is not None:
                p.grad = None
        interp.interp_grad_checkpoint = use_ckpt
        interp.interp_chunk_budget_elems = 1 if chunked else 10 ** 12
        x = x0.clone().requires_grad_(True)
        o = interp(x)
        o.pow(2).sum().backward()
        return o.detach().clone(), x.grad.detach().clone()

    o_ref, g_ref = run(False, False)

    # Guard against the failure mode above: the same config twice must agree. NOT bit-exact --
    # the backward of the advanced-index gather kv[:, neighbors[sl]] is a scatter_add, whose
    # reduction order is non-deterministic on CUDA and on multithreaded CPU, so gradients vary
    # by ~1 float32 ULP (1.19e-07 measured) run to run. The tolerance below is still five orders
    # of magnitude tighter than the bug this guards against (comparing unrelated inputs gives
    # diffs of ~1.0), so it keeps its teeth.
    o_again, g_again = run(False, False)
    torch.testing.assert_close(o_again, o_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g_again, g_ref, rtol=1e-5, atol=1e-5)

    for use_ckpt, chunked in ((True, False), (True, True), (False, True)):
        o, g = run(use_ckpt, chunked)
        torch.testing.assert_close(o, o_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(g, g_ref, rtol=1e-5, atol=1e-5)
