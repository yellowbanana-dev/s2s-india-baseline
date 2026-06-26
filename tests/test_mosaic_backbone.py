"""Tests for the MosaicBackbone adapter (Phase-B step 3).

Four guardrails:
  1. Forward shape:   (2, 13, 32, 64) → (2, 6, 2, 32, 64)
  2. Param count:     reported and within documented budget (<= 15M)
  3. Gradients:       flow through preprocess, attention, cSwiGLU, postprocess
  4. No flash_attn:   model runs with SDPA fallback when flash_attn is absent

No network access, no checkpoint loading.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_LAT = np.linspace(87.1875, -87.1875, 32).astype(np.float32)  # 32 lat points, 5.625 deg
_LON = np.linspace(0.0, 358.125, 64).astype(np.float32)        # 64 lon points, 5.625 deg


def _make_mosaic_cfg(**overrides):
    defaults = dict(
        name="mosaic",
        nside=4,               # tiny nside for fast tests (npix=192)
        dim=32,
        num_heads=4,
        encoder_depth=1,
        decoder_depth=1,
        block_attn_size=48,    # 192/48=4 blocks (divides nside=4 npix=192)
        sparse_block_size=16,
        sparse_block_count=0,
        mlp_ratio=2.0,
        gqa_ratio=1,
        bottleneck_nside=2,    # npix=48
        bottleneck_dim=64,
        bottleneck_num_heads=4,
        bottleneck_depth=1,
        bottleneck_block_attn_size=48,  # 48/48=1 block
        k_neighbors=4,
        qk_norm=False,
        rope=False,
        rope_theta=10000,
        sparse_every=0,
        qkv_compress_ratio=1,
        no_compression=False,
        noise_dim=8,
        ortho_init=False,
    )
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def _build_backbone(cfg=None):
    from s2s.models.mosaic_backbone import MosaicBackbone
    if cfg is None:
        cfg = _make_mosaic_cfg()
    return MosaicBackbone(
        in_channels=13,
        out_channels=2,
        lead=6,
        cfg=cfg,
        latitude=_LAT,
        longitude=_LON,
    )


# ---------------------------------------------------------------------------
# 1. Forward shape
# ---------------------------------------------------------------------------

def test_forward_shape():
    """(2, 13, 32, 64) → (2, 6, 2, 32, 64)."""
    model = _build_backbone()
    model.eval()
    x = torch.randn(2, 13, 32, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 6, 2, 32, 64), f"Unexpected output shape: {out.shape}"


# ---------------------------------------------------------------------------
# 2. Param count within documented budget
# ---------------------------------------------------------------------------

def test_param_count():
    """Param count is reported and within budget (<= 15M)."""
    model = _build_backbone()
    n = sum(p.numel() for p in model.parameters())
    print(f"\nMosaicBackbone (nside=4 test cfg) param count: {n:,}")
    # Sanity: tiny test cfg should still be > 10k (non-trivial model).
    assert n > 10_000, f"Suspiciously small param count: {n}"

    # Full-scale cfg as in configs/model/mosaic.yaml (nside=16).
    full_cfg = _make_mosaic_cfg(
        nside=16,
        dim=256,
        num_heads=8,
        encoder_depth=2,
        decoder_depth=2,
        block_attn_size=512,
        bottleneck_nside=8,
        bottleneck_dim=512,
        bottleneck_num_heads=8,
        bottleneck_depth=2,
        bottleneck_block_attn_size=768,
        noise_dim=32,
        k_neighbors=8,
    )
    full_model = _build_backbone(full_cfg)
    n_full = sum(p.numel() for p in full_model.parameters())
    print(f"MosaicBackbone (full nside=16 cfg) param count: {n_full:,}")
    assert n_full <= 15_000_000, f"Full model exceeds 15M param budget: {n_full:,}"
    assert n_full >= 1_000_000, f"Full model suspiciously small: {n_full:,}"


# ---------------------------------------------------------------------------
# 3. Gradients flow end-to-end
# ---------------------------------------------------------------------------

def test_gradients_flow():
    """Gradients reach preprocess, attention (to_q), cSwiGLU (w2), and postprocess."""
    model = _build_backbone()
    x = torch.randn(1, 13, 32, 64, requires_grad=False)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Check gradient on key components.
    transformer = model.transformer

    # preprocess: first Linear
    pre_w = transformer.preprocess[0].weight
    assert pre_w.grad is not None, "No grad on preprocess[0].weight"
    assert pre_w.grad.abs().sum() > 0, "Zero grad on preprocess[0].weight"

    # encoder stage: attention query projection
    enc_q = transformer.encoder_stages[0].blocks[0].attention.to_q.weight
    assert enc_q.grad is not None, "No grad on encoder attention.to_q"
    assert enc_q.grad.abs().sum() > 0, "Zero grad on encoder attention.to_q"

    # cSwiGLU output projection
    enc_w2 = transformer.encoder_stages[0].blocks[0].ffn.w2.weight
    assert enc_w2.grad is not None, "No grad on encoder cSwiGLU.w2"
    assert enc_w2.grad.abs().sum() > 0, "Zero grad on encoder cSwiGLU.w2"

    # postprocess head (replaced linear)
    post_w = transformer.postprocess[-1].weight
    assert post_w.grad is not None, "No grad on postprocess[-1].weight"
    assert post_w.grad.abs().sum() > 0, "Zero grad on postprocess[-1].weight"


# ---------------------------------------------------------------------------
# 4. Runs without flash_attn (SDPA fallback)
# ---------------------------------------------------------------------------

def test_runs_without_flash_attn(monkeypatch):
    """Model runs correctly when flash_attn is not importable (SDPA fallback)."""
    # Block flash_attn and flash_attn_interface from being importable.
    fake_flash = types.ModuleType("flash_attn")
    # Raise ImportError when the submodule is accessed.
    # We patch sys.modules so any `import flash_attn` raises ImportError.
    original_flash = sys.modules.get("flash_attn", None)
    original_fai   = sys.modules.get("flash_attn_interface", None)
    original_prim  = sys.modules.get("s2s.models.mosaic.primitives", None)

    sys.modules["flash_attn"] = None          # causes ImportError on `import flash_attn`
    sys.modules["flash_attn_interface"] = None

    # Force primitives to be re-imported so the try/except at module level re-runs.
    if "s2s.models.mosaic.primitives" in sys.modules:
        del sys.modules["s2s.models.mosaic.primitives"]
    if "s2s.models.mosaic.mosaic" in sys.modules:
        del sys.modules["s2s.models.mosaic.mosaic"]
    if "s2s.models.mosaic" in sys.modules:
        del sys.modules["s2s.models.mosaic"]
    if "s2s.models.mosaic_backbone" in sys.modules:
        del sys.modules["s2s.models.mosaic_backbone"]

    try:
        from s2s.models.mosaic_backbone import MosaicBackbone
        from s2s.models.mosaic.primitives import _FLASH_ATTN_AVAILABLE
        assert not _FLASH_ATTN_AVAILABLE, "Expected SDPA fallback path"

        model = MosaicBackbone(13, 2, 6, _make_mosaic_cfg(), _LAT, _LON)
        x = torch.randn(1, 13, 32, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 6, 2, 32, 64), f"SDPA path bad shape: {out.shape}"
    finally:
        # Restore original sys.modules state.
        if original_flash is None:
            sys.modules.pop("flash_attn", None)
        else:
            sys.modules["flash_attn"] = original_flash

        if original_fai is None:
            sys.modules.pop("flash_attn_interface", None)
        else:
            sys.modules["flash_attn_interface"] = original_fai

        if original_prim is not None:
            sys.modules["s2s.models.mosaic.primitives"] = original_prim
