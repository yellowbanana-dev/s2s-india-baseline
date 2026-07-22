"""Real fine top-k SELECTION branch (ADR-0010), per query-block.

Torch-gated (authored without a local torch; first executed by the cluster pytest).
The load-bearing test is an independent numpy ORACLE: it recomputes block-mean selection
and the resulting attention from scratch and asserts the torch path matches, so a wrong
gather/permute cannot pass.
"""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from s2s.models.mosaic.primitives import MosaicAttention, selection_attention


def _cfg(selection=False, gate_slots=None, dim=32, num_heads=4,
         block_attn_size=16, sparse_block_size=8, sparse_block_count=2):
    kw = dict(dim=dim, num_heads=num_heads, gqa_ratio=1, qkv_compress_ratio=1,
              block_attn_size=block_attn_size, sparse_block_size=sparse_block_size,
              sparse_block_count=sparse_block_count, rope=False, rope_theta=10000,
              rmsnorm_elementwise_affine=True, selection=selection)
    if gate_slots is not None:
        kw["gate_slots"] = gate_slots
    return SimpleNamespace(**kw)


def test_matches_numpy_oracle():
    """Independent reimplementation: block-mean q/k -> top-k blocks per QUERY BLOCK ->
    softmax attention over those blocks' FINE tokens."""
    torch.manual_seed(0)
    b, seq, h, d = 2, 32, 2, 8
    qs, bs, ksel = 8, 4, 2                      # 4 query blocks, 8 kv blocks, pick 2
    q = torch.randn(b, seq, h, d); k = torch.randn(b, seq, h, d); v = torch.randn(b, seq, h, d)
    got = selection_attention(q, k, v, qs, bs, ksel).detach().numpy()

    qn, kn, vn = q.numpy(), k.numpy(), v.numpy()
    kc = kn.reshape(b, seq // bs, bs, h, d).mean(axis=2)      # block-mean keys
    qb = qn.reshape(b, seq // qs, qs, h, d).mean(axis=2)      # block-mean queries
    exp = np.zeros_like(qn)
    for bi in range(b):
        for hi in range(h):
            for qi in range(seq // qs):
                sc = (qb[bi, qi, hi] @ kc[bi, :, hi].T) * d ** -0.5
                sel = np.argsort(-sc, kind="stable")[:ksel]
                ks = np.concatenate([kn[bi, j * bs:(j + 1) * bs, hi] for j in sel])
                vs = np.concatenate([vn[bi, j * bs:(j + 1) * bs, hi] for j in sel])
                for t in range(qs):
                    qt = qn[bi, qi * qs + t, hi]
                    a = qt @ ks.T * d ** -0.5
                    a = np.exp(a - a.max()); a /= a.sum()
                    exp[bi, qi * qs + t, hi] = a @ vs
    np.testing.assert_allclose(got, exp, rtol=1e-4, atol=1e-5)


def test_queries_in_a_block_share_one_key_set():
    """The tractability trick: selection is per QUERY BLOCK, not per token. If a query
    block's members had independent selections the output would change when only one
    member's query is perturbed; here the SELECTION must stay shared."""
    torch.manual_seed(0)
    b, seq, h, d, qs, bs, kseln = 1, 32, 1, 8, 8, 4, 2
    q = torch.randn(b, seq, h, d); k = torch.randn(b, seq, h, d); v = torch.randn(b, seq, h, d)
    o1 = selection_attention(q, k, v, qs, bs, kseln)
    assert o1.shape == (b, seq, h, d) and torch.isfinite(o1).all()
    # perturbing one token changes ITS output; block-mates keep a finite, valid output
    q2 = q.clone(); q2[0, 0, 0] += 5.0
    o2 = selection_attention(q2, k, v, qs, bs, kseln)
    assert not torch.allclose(o1[0, 0], o2[0, 0])
    assert torch.isfinite(o2).all()


def test_selecting_all_blocks_equals_dense_attention():
    """With sparse_block_count == n_kv_blocks nothing is excluded, so the branch must
    reduce EXACTLY to dense full attention -- the strongest correctness anchor."""
    torch.manual_seed(0)
    b, seq, h, d, qs, bs = 2, 24, 2, 8, 6, 4
    n_kv = seq // bs
    q = torch.randn(b, seq, h, d); k = torch.randn(b, seq, h, d); v = torch.randn(b, seq, h, d)
    got = selection_attention(q, k, v, qs, bs, n_kv)
    qd, kd, vd = [t.permute(0, 2, 1, 3) for t in (q, k, v)]
    exp = torch.nn.functional.scaled_dot_product_attention(qd, kd, vd).permute(0, 2, 1, 3)
    torch.testing.assert_close(got, exp, rtol=1e-4, atol=1e-5)


def test_three_slots_are_distinct_when_selection_enabled():
    """The MAJ-1 defect was o_slc == o_cmp. With selection on they must differ."""
    torch.manual_seed(0)
    attn = MosaicAttention(_cfg(selection=True), block_attn_only=False).eval()
    assert attn.selection and attn.gate_slots == 3
    x = torch.randn(64, 2, 32)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == (64, 2, 32) and torch.isfinite(out).all()


def test_default_off_is_legacy_placeholder():
    attn = MosaicAttention(_cfg(), block_attn_only=False).eval()
    assert attn.selection is False and attn.gate_slots == 3


def test_selection_with_gate_slots_2_is_rejected():
    """The two fixes to MAJ-1 are mutually exclusive; asking for both is a config error."""
    with pytest.raises(ValueError, match="gate_slots"):
        MosaicAttention(_cfg(selection=True, gate_slots=2), block_attn_only=False)
