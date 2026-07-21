"""MIN-1 / MIN-2 / MIN-3 guards (review 2026-07-14).

Assertion-shaped only: dataloader flags and raise-behaviour. No numerical tolerances.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import TensorDataset

from s2s.data.datamodule import S2SDataModule
from s2s.models.mosaic.primitives import (
    MosaicAttention,
    _sdpa_cross,
    block_attention,
)


def _cfg(dim=32, num_heads=4, block_attn_size=16, sparse_block_size=8, sparse_block_count=2):
    return SimpleNamespace(
        dim=dim, num_heads=num_heads, gqa_ratio=1, qkv_compress_ratio=1,
        block_attn_size=block_attn_size, sparse_block_size=sparse_block_size,
        sparse_block_count=sparse_block_count, rope=False, rope_theta=10000,
        rmsnorm_elementwise_affine=True,
    )


# ---------------- MIN-1: drop_last on train only ----------------

def _dm():
    # Only the loader wiring is under test; bypass __init__ so no zarr/channel bookkeeping runs.
    dm = S2SDataModule.__new__(S2SDataModule)
    dm.cfg = SimpleNamespace(train=SimpleNamespace(batch_size=4, num_workers=0))
    ds = TensorDataset(torch.zeros(10, 2), torch.zeros(10, 2))
    dm.train_dataset = dm.val_dataset = dm.test_dataset = ds
    return dm


def test_train_loader_drops_last_batch():
    assert _dm().train_dataloader().drop_last is True


def test_val_and_test_loaders_keep_every_sample():
    # dropping eval samples would silently change reported metrics
    dm = _dm()
    assert dm.val_dataloader().drop_last is False
    assert dm.test_dataloader().drop_last is False


def test_loader_default_is_drop_last_false():
    dm = _dm()
    assert dm._loader(dm.train_dataset, shuffle=False).drop_last is False


# ---------------- MIN-2: no_compression must not be silently ignored ----------------

def test_no_compression_raises_not_implemented():
    attn = MosaicAttention(_cfg(), block_attn_only=False, no_compression=True).eval()
    x = torch.randn(64, 2, 32)
    with pytest.raises(NotImplementedError):
        attn(x)


def test_no_compression_false_still_works():
    attn = MosaicAttention(_cfg(), block_attn_only=False, no_compression=False).eval()
    with torch.no_grad():
        o = attn(torch.randn(64, 2, 32))
    assert o.shape == (64, 2, 32)


# ---------------- MIN-3: q/kv head-count guards ----------------

def test_sdpa_cross_rejects_head_mismatch():
    q = torch.randn(2, 16, 4, 8)     # 4 q heads
    k = torch.randn(2, 8, 2, 8)      # 2 kv heads -> gqa_ratio=2
    v = torch.randn(2, 8, 2, 8)
    with pytest.raises(ValueError, match="gqa_ratio"):
        _sdpa_cross(q, k, v)


def test_block_attention_rejects_head_mismatch():
    q = torch.randn(2, 16, 4, 8)
    k = torch.randn(2, 16, 2, 8)
    v = torch.randn(2, 16, 2, 8)
    with pytest.raises(ValueError, match="gqa_ratio"):
        block_attention(q, k, v, block_size=8)


def test_equal_head_counts_still_pass():
    q = torch.randn(2, 16, 4, 8)
    k = torch.randn(2, 16, 4, 8)
    v = torch.randn(2, 16, 4, 8)
    with torch.no_grad():
        assert block_attention(q, k, v, block_size=8).shape == (2, 16, 4, 8)
        assert _sdpa_cross(q, k, v).shape == (2, 16, 4, 8)
