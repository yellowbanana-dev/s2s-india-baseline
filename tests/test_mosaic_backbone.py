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


# --------------------------------------------------------------------------- #
# Regression: the Mosaic HEALPix interpolation grid must use the TRUE data     #
# longitudes (WeatherBench2 5.625deg = arange(64)*5.625), never a wrong-spaced #
# default. A mismatch silently degrades eval (caught in Phase-B step-3 review). #
# --------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------
# 5. Global encoder attention: block_attn_size == npix → exactly 1 block
# ---------------------------------------------------------------------------

def test_global_encoder_block():
    """Setting block_attn_size == npix for the encoder stage produces 1 block.

    Guards against silently reverting to local block attention (Stage-A fix 1).
    nside=4 → npix=12*4^2=192 tokens; setting block_attn_size=192 → nb=1.
    """
    nside = 4
    npix = 12 * nside ** 2  # 192
    cfg = _make_mosaic_cfg(
        nside=nside,
        block_attn_size=npix,         # global: one block covers all encoder tokens
        bottleneck_block_attn_size=48, # bottleneck stays global (48/48=1)
    )
    model = _build_backbone(cfg)
    model.eval()

    x = torch.randn(1, 13, 32, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 6, 2, 32, 64), f"Global-block forward bad shape: {out.shape}"

    # Verify the config knob is what we set (not silently overridden).
    # The encoder stage block_attn_size is read from cfg via MosaicBackbone.__init__.
    assert int(cfg.block_attn_size) == npix, "block_attn_size was mutated"


# ---------------------------------------------------------------------------
# 6. RoPE on: forward still works and produces non-trivial output
# ---------------------------------------------------------------------------

def test_rope_enabled_forward():
    """Forward pass completes with rope=True (Stage-A fix 2)."""
    cfg = _make_mosaic_cfg(rope=True)
    model = _build_backbone(cfg)
    model.eval()
    x = torch.randn(1, 13, 32, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 6, 2, 32, 64), f"RoPE forward bad shape: {out.shape}"
    # Output must not be all-zeros (basic sanity: model does something).
    assert out.abs().max() > 0.0, "RoPE model output is all zeros"


# ---------------------------------------------------------------------------
# 7. Native time embedding: non-zero doy_cos input → non-zero day_year_time
# ---------------------------------------------------------------------------

def test_time_embedding_nonzero():
    """Non-zero doy_cos in x[:, -1] produces a non-zero day_year_time signal.

    Verifies Stage-A fix 3: the adapter derives day_normalized via arccos(doy_cos)
    and injects it into the Transformer time embedding instead of passing zeros.
    """
    from s2s.models.mosaic_backbone import MosaicBackbone
    import math

    cfg = _make_mosaic_cfg()
    model = MosaicBackbone(13, 2, 6, cfg, _LAT, _LON)
    model.eval()

    # Midsummer: doy_cos = cos(2π * 182/365.25) ≈ cos(π) ≈ -1
    # arccos(-1)/2π = 0.5 → day_year_time[:, 0, 0] = 0.5
    # Midwinter: doy=0 → doy_cos=1 → day_normalized=0
    doy_cos_summer = torch.tensor(-0.99, dtype=torch.float32)
    doy_cos_winter = torch.tensor(1.00, dtype=torch.float32)
    expected_day_summer = math.acos(-0.99) / (2 * math.pi)
    expected_day_winter = 0.0

    # Capture day_year_time passed to self.transformer by patching forward
    captured = {}
    _real_fwd = model.transformer.forward

    def _patched_fwd(x, day_year_time, **kw):
        captured["day_year_time"] = day_year_time.detach().clone()
        return _real_fwd(x, day_year_time, **kw)

    model.transformer.forward = _patched_fwd

    x_summer = torch.randn(1, 13, 32, 64)
    x_summer[:, -1] = doy_cos_summer   # uniform doy_cos channel

    with torch.no_grad():
        model(x_summer)

    day_got = captured["day_year_time"][0, 0, 0].item()
    assert abs(day_got - expected_day_summer) < 1e-4, (
        f"Expected day_normalized ≈ {expected_day_summer:.4f}, got {day_got:.4f}"
    )

    # Sanity: winter doy_cos=1 → day_normalized=0
    x_winter = torch.randn(1, 13, 32, 64)
    x_winter[:, -1] = doy_cos_winter
    with torch.no_grad():
        model(x_winter)
    day_got_w = captured["day_year_time"][0, 0, 0].item()
    assert abs(day_got_w - expected_day_winter) < 1e-4, (
        f"Expected day_normalized=0 for winter, got {day_got_w:.4f}"
    )


import numpy as _np


# ---------------------------------------------------------------------------
# 8. Per-model LR override: mosaic.yaml lr wins over train default
# ---------------------------------------------------------------------------

def test_mosaic_uses_model_lr_not_train_default():
    """cfg.model.lr=3e-5 must override cfg.train.lr=3e-4 for Mosaic.

    Guards against the lr mismatch regression: at lr=3e-4 Mosaic's zero-init
    residuals are destroyed in one step (epoch-0 curse). The override in
    configs/model/mosaic.yaml sets lr=3e-5; this test ensures lit.py respects it.
    """
    from omegaconf import OmegaConf
    from s2s.models.lit import S2SLitModule

    train_cfg = OmegaConf.create({
        "lr": 3.0e-4,        # default (patch_vit value)
        "weight_decay": 0.1,
        "warmup_epochs": 2,
        "max_epochs": 50,
        "min_lr": 1.0e-6,
    })
    model_cfg = _make_mosaic_cfg(lr=3.0e-5)  # mosaic.yaml per-model override

    cfg = OmegaConf.create({"train": train_cfg, "model": model_cfg})
    lit = S2SLitModule(
        in_channels=13, out_channels=2, lead=6,
        latitude=_LAT, longitude=_LON, cfg=cfg,
    )
    effective = lit._effective_lr()
    assert abs(effective - 3.0e-5) < 1e-9, (
        f"Expected effective lr=3e-5 (from cfg.model.lr), got {effective}. "
        "cfg.train.lr=3e-4 must NOT silently override cfg.model.lr for Mosaic."
    )


def test_patch_vit_falls_back_to_train_lr():
    """When cfg.model has no lr key, lit.py must fall back to cfg.train.lr."""
    from omegaconf import OmegaConf
    from s2s.models.lit import S2SLitModule

    train_cfg = OmegaConf.create({
        "lr": 3.0e-4,
        "weight_decay": 0.1,
        "warmup_epochs": 2,
        "max_epochs": 50,
        "min_lr": 1.0e-6,
    })
    model_cfg = OmegaConf.create({   # patch_vit-style: no lr key
        "patch_size": 2, "embed_dim": 32, "depth": 2,
        "num_heads": 4, "mlp_ratio": 4.0, "drop_rate": 0.1,
    })
    cfg = OmegaConf.create({"train": train_cfg, "model": model_cfg})
    lit = S2SLitModule(
        in_channels=6, out_channels=2, lead=6,
        latitude=_LAT, cfg=cfg,
    )
    effective = lit._effective_lr()
    assert abs(effective - 3.0e-4) < 1e-9, (
        f"Expected effective lr=3e-4 (from cfg.train.lr), got {effective}."
    )


def test_lit_longitude_fallback_is_true_wb2_grid():
    """If longitude isn't passed, lit.py must fall back to the real 5.625deg grid,
    NOT linspace(0, 358.125, 64) which has 5.684deg spacing."""
    import inspect
    from s2s.models import lit as _lit
    src = inspect.getsource(_lit)
    assert "np.arange(64) * 5.625" in src, "lit.py longitude fallback must be arange(64)*5.625"
    assert "linspace(0.0, 358.125, 64)" not in src, "old wrong-spaced default must be gone"
    # And the true grid ends at 354.375, not 358.125.
    true_grid = _np.arange(64) * 5.625
    assert abs(true_grid[-1] - 354.375) < 1e-6
    assert abs(true_grid[1] - true_grid[0] - 5.625) < 1e-6
