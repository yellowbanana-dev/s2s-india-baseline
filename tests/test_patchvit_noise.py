"""Fix 6 (C3): PatchViT stochastic member mechanism. Needs torch (cluster).

Guards:
  1. noise_dim>0, M>1 -> M DISTINCT members, shape (B,M,lead,C,lat,lon).
  2. same seed reproduces; different seed differs.
  3. M==1 stays deterministic 5-D (back-compat with the MSE path/eval/tests).
  4. noise_dim==0 legacy path unchanged (M>1 tiles identical members).
  5. param count matched to the slim-Mosaic band (~3.5-3.7M) for a fair ablation.
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from s2s.models.patch_vit import PatchViT

_IN, _OUT, _LEAD = 13, 2, 6


def _cfg(noise_dim=8, embed_dim=224, depth=6):
    return OmegaConf.create(dict(
        name="patch_vit", patch_size=2, embed_dim=embed_dim, depth=depth,
        num_heads=8, mlp_ratio=4.0, drop_rate=0.0, noise_dim=noise_dim,
    ))


def _x(b=2):
    return torch.zeros(b, _IN, 32, 64)


def test_noise_members_distinct_and_shaped():
    m = PatchViT(_IN, _OUT, _LEAD, _cfg(noise_dim=8), seed=0).eval()
    # perturb to_film so members diverge (zero-init would keep them identical)
    with torch.no_grad():
        m.to_film.weight.normal_(0, 0.5)
        m.to_film.bias.normal_(0, 0.5)
    with torch.no_grad():
        out = m(_x(), num_noise_samples=4)
    assert out.shape == (2, 4, _LEAD, _OUT, 32, 64)
    # members differ from each other
    assert not torch.allclose(out[:, 0], out[:, 1])


def test_same_seed_reproduces_diff_seed_differs():
    cfg = _cfg(noise_dim=8)
    a = PatchViT(_IN, _OUT, _LEAD, cfg, seed=0).eval()
    b = PatchViT(_IN, _OUT, _LEAD, cfg, seed=0).eval()
    c = PatchViT(_IN, _OUT, _LEAD, cfg, seed=1).eval()
    with torch.no_grad():
        for mdl in (a, b, c):
            mdl.to_film.weight.normal_(0, 0.5); mdl.to_film.bias.normal_(0, 0.5)
        b.load_state_dict(a.state_dict()); c.load_state_dict(a.state_dict())
        oa = a(_x(), num_noise_samples=4)
        ob = b(_x(), num_noise_samples=4)
        oc = c(_x(), num_noise_samples=4)
    torch.testing.assert_close(oa, ob)      # same seed => identical draws
    assert not torch.allclose(oa, oc)       # different seed => different members


def test_single_member_deterministic_backcompat():
    m = PatchViT(_IN, _OUT, _LEAD, _cfg(noise_dim=8), seed=0).eval()
    with torch.no_grad():
        out = m(_x(), num_noise_samples=1)
    assert out.shape == (2, _LEAD, _OUT, 32, 64)  # 5-D, no member axis


def test_noise_dim_zero_tiles_identically():
    m = PatchViT(_IN, _OUT, _LEAD, _cfg(noise_dim=0), seed=0).eval()
    with torch.no_grad():
        out = m(_x(), num_noise_samples=4)
    assert out.shape == (2, 4, _LEAD, _OUT, 32, 64)
    assert torch.allclose(out[:, 0], out[:, 1])   # identical members (legacy behavior)


def test_param_count_matches_mosaic_band():
    n = sum(p.numel() for p in PatchViT(_IN, _OUT, _LEAD, _cfg()).parameters())
    print(f"\nPatchViT+noise param count: {n:,}")
    assert 3_000_000 < n < 4_300_000, f"param count {n:,} outside slim-Mosaic band"
