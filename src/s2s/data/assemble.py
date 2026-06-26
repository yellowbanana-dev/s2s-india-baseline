"""Stage 2c -- assemble per-sample model tensors from weekly anomalies (task #7).

Single source of truth for channel order/count, so the model's in/out_channels
and the dataset's actual tensors can never silently disagree. Targets are also
fed back as input history (the model gets to see what it's persisting from),
followed by predictors, followed by one cyclical day-of-year channel.

pack_windows() is the shared packing kernel used by both:
  - assemble_arrays()             : W-MON weekly bins  (test split / eval)
  - daily_init_weekly_windows()   : daily-strided rolling 7-day means (train/val)
Both paths produce tensors with identical shape and channel order.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def predictor_vars(cfg) -> list[str]:
    """Flat predictor names, fixed order: surface vars, then `<var>_<level>` per level."""
    v = cfg.data.variables.predictors
    names = list(v.surface) if v.surface else []
    levels = getattr(v, "levels", None) or {}
    for var, plevels in levels.items():
        for p in plevels:
            names.append(f"{var}_{p}")
    return names


def target_vars(cfg) -> list[str]:
    return list(cfg.data.variables.targets.surface)


def input_vars(cfg) -> list[str]:
    """Targets first (fed back as history, like persistence), then predictors."""
    return target_vars(cfg) + predictor_vars(cfg)


def in_out_channels(cfg) -> tuple[int, int]:
    """(in_channels, out_channels) for building the model -- matches assemble_arrays exactly."""
    history_weeks = int(cfg.data.history_weeks)
    in_channels = history_weeks * len(input_vars(cfg)) + 1  # +1 day-of-year encoding
    out_channels = len(target_vars(cfg))
    return in_channels, out_channels


def pack_windows(
    in_hist: np.ndarray,
    out_leads: np.ndarray,
    doy_cos_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack pre-built per-sample history and lead windows into model-ready tensors.

    in_hist:      (N, history_weeks, n_in_vars, lat, lon)  — oldest week first
    out_leads:    (N, n_lead_weeks,  n_out_vars, lat, lon)
    doy_cos_vals: (N,) float32 cosine day-of-year at each init time

    Returns:
      inputs  (N, history_weeks*n_in_vars + 1, lat, lon)  float32
      targets (N, n_lead_weeks, n_out_vars, lat, lon)      float32

    Channel order: [oldest_week_var0, oldest_week_var1, ..., newest_week_var0, ...] + doy_cos.
    This is the ONLY place that defines the channel layout; in_out_channels() counts it.
    """
    N, hw, niv, lat, lon = in_hist.shape
    # (N, history_weeks, n_in_vars, lat, lon) -> (N, history_weeks*n_in_vars, lat, lon)
    flat_hist = in_hist.reshape(N, hw * niv, lat, lon).astype(np.float32)
    inputs = np.empty((N, hw * niv + 1, lat, lon), dtype=np.float32)
    inputs[:, : hw * niv] = flat_hist
    inputs[:, -1] = doy_cos_vals[:, None, None]  # broadcast over lat, lon
    targets = np.asarray(out_leads, dtype=np.float32)
    return inputs, targets


def assemble_arrays(weekly: xr.Dataset, cfg) -> dict:
    """Build (inputs, targets) numpy arrays from a weekly-mean anomaly Dataset.

    inputs  : (N, in_channels,  lat, lon)         float32
    targets : (N, n_lead, out_channels, lat, lon) float32
    time    : (N,) the INIT week of each sample

    Only init weeks with a full history window behind them and a full lead
    window ahead of them are kept -- no zero-padding at the edges, since that
    would silently mix real and fabricated data.

    NaNs in input predictors (e.g. sea_surface_temperature over land) are
    filled with 0 -- the model doesn't need a physically meaningful value
    there, just a finite one. NaNs in targets are left as NaN; the caller's
    loss/eval must mask them (none are currently expected for the target
    variables over the India box, but this is not asserted here).
    """
    history_weeks = int(cfg.data.history_weeks)
    lead_weeks = list(cfg.data.lead_weeks)
    max_lead = max(lead_weeks)
    in_vars = input_vars(cfg)
    out_vars = target_vars(cfg)

    n_time = weekly.sizes["time"]
    lo = history_weeks - 1
    hi = n_time - max_lead
    if hi <= lo:
        raise ValueError(
            f"not enough weeks ({n_time}) for history_weeks={history_weeks} "
            f"and max lead={max_lead}"
        )
    valid_idx = np.arange(lo, hi)
    lat, lon = weekly.sizes["latitude"], weekly.sizes["longitude"]

    def _tlatlon(name):
        return weekly[name].transpose("time", "latitude", "longitude").values

    in_stack = np.nan_to_num(
        np.stack([_tlatlon(v) for v in in_vars], axis=0), nan=0.0
    )  # (n_in_vars, T, lat, lon)
    out_stack = np.stack([_tlatlon(v) for v in out_vars], axis=0)  # (n_out_vars, T, lat, lon)

    doy = weekly.time.dt.dayofyear.values.astype(np.float64)
    doy_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    # Vectorised extraction: build (N, history_weeks, n_in_vars, lat, lon) in one shot.
    # Offsets: 0 = oldest week, history_weeks-1 = most recent (same as the old t-hw+1..t+1 slice)
    hist_offsets = np.arange(history_weeks)  # [0, 1, ..., hw-1]
    hist_idx = valid_idx[:, None] - (history_weeks - 1) + hist_offsets[None, :]  # (N, hw)
    # in_stack[:, hist_idx] -> (n_in_vars, N, hw, lat, lon) -> transpose -> (N, hw, n_in_vars, lat, lon)
    in_hist = in_stack[:, hist_idx, :, :].transpose(1, 2, 0, 3, 4)

    lead_idx = valid_idx[:, None] + np.array(lead_weeks)[None, :]  # (N, n_leads)
    # out_stack[:, lead_idx] -> (n_out_vars, N, n_leads, lat, lon) -> (N, n_leads, n_out_vars, lat, lon)
    out_leads = out_stack[:, lead_idx, :, :].transpose(1, 2, 0, 3, 4)

    inputs, targets = pack_windows(in_hist, out_leads, doy_cos[valid_idx])
    return {"inputs": inputs, "targets": targets, "time": weekly.time.values[valid_idx]}
