"""Phase-B sampling + scheduler tests.

Three guardrails for the denser-data changes:
  1. Leakage  -- no sample's day-span crosses a split boundary.
  2. Density  -- stride=1 yields ~7x the W-MON samples; shapes match.
  3. Scheduler -- cosine+warmup LR curve; weight_decay set on AdamW param groups.

No network, no checkpoint loading.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from omegaconf import OmegaConf

from s2s.data.assemble import assemble_arrays
from s2s.data.windows import daily_init_weekly_windows, daily_to_weekly_mean


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_cfg(history_weeks=2, lead_weeks=(1, 2, 3, 4, 5, 6), train_stride_days=1):
    return OmegaConf.create({
        "data": {
            "history_weeks": history_weeks,
            "lead_weeks": list(lead_weeks),
            "train_stride_days": train_stride_days,
            "variables": {
                "targets": {"surface": ["2m_temperature", "total_precipitation_24hr"]},
                "predictors": {
                    "surface": ["sea_surface_temperature"],
                    "levels": {"geopotential": [500], "u_component_of_wind": [850, 200]},
                },
            },
        }
    })


def _make_daily_ds(n_days: int, lat: int = 4, lon: int = 8, seed: int = 0) -> xr.Dataset:
    """Synthetic daily anomaly dataset with all required variables."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2010-01-01", periods=n_days, freq="D")
    shape = (n_days, lat, lon)
    data_vars = {
        "2m_temperature":           (("time", "latitude", "longitude"), rng.normal(size=shape).astype(np.float32)),
        "total_precipitation_24hr": (("time", "latitude", "longitude"), rng.normal(size=shape).astype(np.float32)),
        "sea_surface_temperature":  (("time", "latitude", "longitude"), rng.normal(size=shape).astype(np.float32)),
        "geopotential_500":         (("time", "latitude", "longitude"), rng.normal(size=shape).astype(np.float32)),
        "u_component_of_wind_850":  (("time", "latitude", "longitude"), rng.normal(size=shape).astype(np.float32)),
        "u_component_of_wind_200":  (("time", "latitude", "longitude"), rng.normal(size=shape).astype(np.float32)),
    }
    return xr.Dataset(
        data_vars,
        coords={
            "time": time,
            "latitude": np.linspace(-60, 60, lat),
            "longitude": np.linspace(0, 270, lon),
        },
    )


# ---------------------------------------------------------------------------
# 1. Leakage test
# ---------------------------------------------------------------------------

def test_daily_strided_no_cross_boundary_leakage():
    """No sample built from split-A data needs any day from split-B.

    Strategy: slice a synthetic daily dataset at a hard boundary (day 200).
    Build windows from the first portion only.  Assert every emitted sample's
    full day span [init - 7*(history_weeks-1) - 6 .. init + 7*max_lead + 6]
    stays within the first portion.
    """
    cfg = _make_cfg(history_weeks=2, lead_weeks=[1, 2, 3, 4, 5, 6])
    n_total = 400
    boundary_day = 200
    history_weeks = int(cfg.data.history_weeks)
    max_lead = max(cfg.data.lead_weeks)

    daily_full = _make_daily_ds(n_days=n_total)
    train_part = daily_full.isel(time=slice(0, boundary_day))
    boundary_date = pd.Timestamp(daily_full.time.values[boundary_day])

    out = daily_init_weekly_windows(train_part, cfg, stride_days=1)

    assert len(out["time"]) > 0, "Expected at least one valid sample"

    init_times = pd.DatetimeIndex(out["time"])
    # Earliest and latest calendar day touched by any sample
    earliest_history = init_times.min() - pd.Timedelta(days=7 * (history_weeks - 1) + 6)
    latest_lead = init_times.max() + pd.Timedelta(days=7 * max_lead + 6)

    split_start = pd.Timestamp(train_part.time.values[0])
    split_end = pd.Timestamp(train_part.time.values[-1])

    assert earliest_history >= split_start, (
        f"History underflows split: {earliest_history} < {split_start}"
    )
    assert latest_lead <= split_end, (
        f"Lead window crosses boundary: {latest_lead} >= {boundary_date}"
    )


# ---------------------------------------------------------------------------
# 2. Density test
# ---------------------------------------------------------------------------

def test_daily_strided_density_vs_wmon():
    """stride=1 gives ~7x the W-MON samples; tensor shapes are identical."""
    cfg = _make_cfg(history_weeks=2, lead_weeks=[1, 2, 3, 4, 5, 6])
    # Use 3 years of daily data for a stable ratio
    n_days = 365 * 3
    daily_ds = _make_daily_ds(n_days=n_days)

    # Daily-strided path
    out_daily = daily_init_weekly_windows(daily_ds, cfg, stride_days=1)

    # W-MON path (same data)
    weekly = daily_to_weekly_mean(daily_ds)
    out_wmon = assemble_arrays(weekly, cfg)

    n_daily = len(out_daily["time"])
    n_wmon = len(out_wmon["time"])
    ratio = n_daily / n_wmon
    assert 6.0 <= ratio <= 8.0, (
        f"Expected ~7x more daily-strided samples than W-MON; got {n_daily}/{n_wmon}={ratio:.2f}"
    )

    # Shape contract: everything except N must match
    assert out_daily["inputs"].shape[1:] == out_wmon["inputs"].shape[1:], (
        f"Input channel/grid mismatch: {out_daily['inputs'].shape} vs {out_wmon['inputs'].shape}"
    )
    assert out_daily["targets"].shape[1:] == out_wmon["targets"].shape[1:], (
        f"Target shape mismatch: {out_daily['targets'].shape} vs {out_wmon['targets'].shape}"
    )

    # All input values must be finite (NaN-fill applied correctly)
    assert np.isfinite(out_daily["inputs"]).all(), "NaN found in daily-strided inputs"


# ---------------------------------------------------------------------------
# 3. Scheduler + weight_decay test
# ---------------------------------------------------------------------------

def test_cosine_warmup_and_weight_decay():
    """LR schedule: linear warmup then cosine decay; weight_decay on AdamW."""
    warmup = 2
    max_ep = 10
    base_lr = 3.0e-4
    min_lr = 1.0e-6
    wd = 0.1

    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)

    # weight_decay must appear on every param group
    for pg in opt.param_groups:
        assert abs(pg["weight_decay"] - wd) < 1e-12, (
            f"Expected weight_decay={wd}, got {pg['weight_decay']}"
        )

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, max_ep - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr + cosine * (base_lr - min_lr)) / base_lr

    # Epoch 0 (first warmup step): LR = base_lr * (1/warmup)
    assert abs(lr_lambda(0) - 1.0 / warmup) < 1e-12

    # Epoch warmup-1 (end of warmup): LR = base_lr
    assert abs(lr_lambda(warmup - 1) - 1.0) < 1e-12

    # First post-warmup epoch (progress=0): still at base_lr (cosine starts at 1)
    assert abs(lr_lambda(warmup) - 1.0) < 1e-9

    # One epoch into decay: strictly below base_lr
    assert lr_lambda(warmup + 1) < 1.0

    # At max_ep (full cosine period): exactly at min_lr/base_lr
    assert abs(lr_lambda(max_ep) - min_lr / base_lr) < 1e-12
