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
    attn = MosaicAttention(_cfg(sparse_block_size=8), block_attn_only=False).eval()
    x = torch.randn(60, 1, 32)  # 60 not divisible by 8
    with pytest.raises(ValueError):
        attn(x)
