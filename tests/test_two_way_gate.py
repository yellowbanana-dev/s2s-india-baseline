"""Two-way strategy gate (ADR-0009): removing the MAJ-1 duplicate-slot init bias.

The legacy 3-slot gate holds (local, compressed, selection) where selection DUPLICATES
compressed, so a uniform softmax starts the model at 1/3 local + 2/3 compressed. These
tests pin the structural claim exactly (by zeroing the gate weights, the softmax is
exactly uniform, so the bias is an exact ratio, not a statistical one).

NOTE: torch-gated -- authored without a local torch (proxy-blocked); first executed by the
cluster pytest.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from s2s.models.mosaic.primitives import MosaicAttention


def _cfg(gate_slots=None, dim=32, num_heads=4):
    kw = dict(dim=dim, num_heads=num_heads, gqa_ratio=1, qkv_compress_ratio=1,
              block_attn_size=16, sparse_block_size=8, sparse_block_count=2,
              rope=False, rope_theta=10000, rmsnorm_elementwise_affine=True)
    if gate_slots is not None:
        kw["gate_slots"] = gate_slots
    return SimpleNamespace(**kw)


def _effective_weights(attn, seq=64, batch=2, dim=32):
    """Zero the gate weights -> logits are exactly 0 -> softmax exactly uniform, so the
    local/compressed split is an EXACT consequence of the slot count."""
    attn.to_strategy_combine_mlp.weight.data.zero_()
    w = attn.generate_strategy_weights(torch.randn(seq, batch, dim))
    local = w[0]
    compressed = w[1] + w[2] if w.shape[0] == 3 else w[1]
    return float(local.mean()), float(compressed.mean())


def test_legacy_three_slot_gate_starts_2_to_1_biased_toward_compressed():
    attn = MosaicAttention(_cfg(), block_attn_only=False).eval()   # default = 3
    assert attn.gate_slots == 3
    local, compressed = _effective_weights(attn)
    assert compressed == pytest.approx(2.0 * local, rel=1e-6), "MAJ-1 bias should be exactly 2:1"


def test_two_way_gate_starts_unbiased():
    attn = MosaicAttention(_cfg(gate_slots=2), block_attn_only=False).eval()
    assert attn.gate_slots == 2
    local, compressed = _effective_weights(attn)
    assert compressed == pytest.approx(local, rel=1e-6), "2-slot gate must start 1/2 - 1/2"


def test_gate_layer_shape_tracks_slot_count():
    """Default must stay 3*heads so existing checkpoints still load."""
    assert MosaicAttention(_cfg(), block_attn_only=False).to_strategy_combine_mlp.out_features == 3 * 4
    assert MosaicAttention(_cfg(gate_slots=2), block_attn_only=False).to_strategy_combine_mlp.out_features == 2 * 4


def test_forward_shape_unchanged_with_two_way_gate():
    attn = MosaicAttention(_cfg(gate_slots=2), block_attn_only=False).eval()
    x = torch.randn(64, 2, 32)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == (64, 2, 32) and torch.isfinite(out).all()


def test_invalid_slot_count_rejected():
    for bad in (0, 1, 4):
        with pytest.raises(ValueError, match="gate_slots"):
            MosaicAttention(_cfg(gate_slots=bad), block_attn_only=False)
