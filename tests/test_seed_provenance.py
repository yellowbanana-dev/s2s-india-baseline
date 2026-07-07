"""Fix 8 (M3): noise-seed threading + adapter n==1 guard. Needs torch (cluster).

Self-contained (mirrors tests/test_mosaic_backbone.py config). Guards:
  1. cfg.seed threads to the Mosaic NoiseGenerator (was hardcoded 42 -> shared
     across 'independent' seeds).
  2. Different seeds actually change the drawn members; same seed reproduces.
  3. The vendored adapter asserts n==1 (members packed (b s n), unpacked (b n s);
     they agree only when n==1).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

_LAT = np.linspace(87.1875, -87.1875, 32).astype(np.float32)
_LON = np.linspace(0.0, 358.125, 64).astype(np.float32)


def _make_mosaic_cfg(**overrides):
    defaults = dict(
        name="mosaic", nside=4, dim=32, num_heads=4, encoder_depth=1, decoder_depth=1,
        block_attn_size=48, sparse_block_size=16, sparse_block_count=0, mlp_ratio=2.0,
        gqa_ratio=1, bottleneck_nside=2, bottleneck_dim=64, bottleneck_num_heads=4,
        bottleneck_depth=1, bottleneck_block_attn_size=48, k_neighbors=4, qk_norm=False,
        rope=False, rope_theta=10000, sparse_every=0, qkv_compress_ratio=1,
        no_compression=False, noise_dim=8, ortho_init=False,
    )
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def _backbone(seed=42, cfg=None):
    from s2s.models.mosaic_backbone import MosaicBackbone
    return MosaicBackbone(
        in_channels=13, out_channels=2, lead=6,
        cfg=cfg or _make_mosaic_cfg(noise_dim=8),
        latitude=_LAT, longitude=_LON, seed=seed,
    )


def test_seed_threads_to_noise_generator():
    assert _backbone(seed=0).transformer.noise_generator.seed == 0
    assert _backbone(seed=7).transformer.noise_generator.seed == 7
    assert _backbone().transformer.noise_generator.seed == 42  # default preserved


def test_different_seeds_change_members_same_seed_reproduces():
    x = torch.zeros(2, 13, 32, 64)
    m0 = _backbone(seed=0).eval()
    m0b = _backbone(seed=0).eval()
    m1 = _backbone(seed=1).eval()
    # copy weights so ONLY the noise seed differs
    m0b.load_state_dict(m0.state_dict())
    m1.load_state_dict(m0.state_dict())
    with torch.no_grad():
        a = m0(x, num_noise_samples=4)
        a_rep = m0b(x, num_noise_samples=4)
        b = m1(x, num_noise_samples=4)
    torch.testing.assert_close(a, a_rep)   # same seed => identical draws
    assert not torch.allclose(a, b)        # different seed => different members


def test_adapter_asserts_single_history_step():
    tr = _backbone(seed=0).transformer
    bad = torch.zeros(1, 2, 1, 64, 32, 13)  # n=2 -> must raise before compute
    dyt = torch.zeros(1, 1, 2)
    with pytest.raises(AssertionError):
        tr(bad, dyt, num_noise_samples=1)
