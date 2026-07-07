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


# Sea-surface temperature: the slow boundary variable (ADR-0006, lever b).
_SST_VAR = "sea_surface_temperature"


def sst_extra_lags(cfg) -> list[int]:
    """Extra SST-history lags (in weeks) beyond the shared `history_weeks` stack.

    Configured by `data.sst_history_lags_weeks` (ADR-0006). Empty unless SST is a
    predictor AND the list is non-empty, so the default pipeline is unchanged.
    """
    lags = list(getattr(cfg.data, "sst_history_lags_weeks", None) or [])
    if not lags:
        return []
    surface = list(cfg.data.variables.predictors.surface or [])
    if _SST_VAR not in surface:
        return []
    return [int(l) for l in lags]


def n_sst_extra(cfg) -> int:
    return len(sst_extra_lags(cfg))


def in_out_channels(cfg) -> tuple[int, int]:
    """(in_channels, out_channels) for building the model -- matches assemble_arrays exactly."""
    history_weeks = int(cfg.data.history_weeks)
    # history stack + extra SST-lag channels (ADR-0006) + 2 seasonal (doy_cos, doy_sin)
    in_channels = history_weeks * len(input_vars(cfg)) + n_sst_extra(cfg) + 2
    out_channels = len(target_vars(cfg))
    return in_channels, out_channels


def pack_windows(
    in_hist: np.ndarray,
    out_leads: np.ndarray,
    doy_cos_vals: np.ndarray,
    doy_sin_vals: np.ndarray,
    sst_extra: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack pre-built per-sample windows into model-ready tensors.

    in_hist:      (N, history_weeks, n_in_vars, lat, lon)  — oldest week first
    out_leads:    (N, n_lead_weeks,  n_out_vars, lat, lon)
    doy_cos_vals: (N,) float32 cos(day-of-year) at each init time
    doy_sin_vals: (N,) float32 sin(day-of-year) at each init time
    sst_extra:    (N, n_sst_extra, lat, lon) extra SST-lag anomaly channels, or None

    Returns:
      inputs  (N, history_weeks*n_in_vars + n_sst_extra + 2, lat, lon) float32
      targets (N, n_lead_weeks, n_out_vars, lat, lon)                   float32

    Channel order (THE single source of truth; in_out_channels() counts it):
      [ history: oldest_week_var0..newest_week_varK ]
      [ SST extra lags (ADR-0006), in the configured lag order ]
      [ doy_cos, doy_sin ]                      <- always the LAST two channels
    The seasonal pair is kept last so MosaicBackbone can read x[:, -2]=cos,
    x[:, -1]=sin and recover the true day fraction via atan2.
    """
    N, hw, niv, lat, lon = in_hist.shape
    flat_hist = in_hist.reshape(N, hw * niv, lat, lon).astype(np.float32)
    parts = [flat_hist]
    if sst_extra is not None and sst_extra.shape[1] > 0:
        parts.append(np.nan_to_num(sst_extra, nan=0.0).astype(np.float32))
    body = np.concatenate(parts, axis=1) if len(parts) > 1 else flat_hist
    n_body = body.shape[1]
    inputs = np.empty((N, n_body + 2, lat, lon), dtype=np.float32)
    inputs[:, :n_body] = body
    inputs[:, n_body] = doy_cos_vals[:, None, None]      # broadcast over lat, lon
    inputs[:, n_body + 1] = doy_sin_vals[:, None, None]
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

    lags = sst_extra_lags(cfg)
    max_lag = max(lags) if lags else 0

    n_time = weekly.sizes["time"]
    lo = max(history_weeks - 1, max_lag)   # deepest lookback = SST lag or history
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
    doy_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)

    # Extra SST-history channels at the configured week-lags (ADR-0006).
    if lags:
        sst_series = np.nan_to_num(_tlatlon(_SST_VAR), nan=0.0)  # (T, lat, lon)
        sst_idx = valid_idx[:, None] - np.array(lags)[None, :]   # (N, n_lags) weekly indices
        sst_extra = sst_series[sst_idx]                          # (N, n_lags, lat, lon)
    else:
        sst_extra = None

    # Vectorised extraction: build (N, history_weeks, n_in_vars, lat, lon) in one shot.
    # Offsets: 0 = oldest week, history_weeks-1 = most recent (same as the old t-hw+1..t+1 slice)
    hist_offsets = np.arange(history_weeks)  # [0, 1, ..., hw-1]
    hist_idx = valid_idx[:, None] - (history_weeks - 1) + hist_offsets[None, :]  # (N, hw)
    # in_stack[:, hist_idx] -> (n_in_vars, N, hw, lat, lon) -> transpose -> (N, hw, n_in_vars, lat, lon)
    in_hist = in_stack[:, hist_idx, :, :].transpose(1, 2, 0, 3, 4)

    lead_idx = valid_idx[:, None] + np.array(lead_weeks)[None, :]  # (N, n_leads)
    # out_stack[:, lead_idx] -> (n_out_vars, N, n_leads, lat, lon) -> (N, n_leads, n_out_vars, lat, lon)
    out_leads = out_stack[:, lead_idx, :, :].transpose(1, 2, 0, 3, 4)

    inputs, targets = pack_windows(
        in_hist, out_leads, doy_cos[valid_idx], doy_sin[valid_idx], sst_extra
    )
    return {"inputs": inputs, "targets": targets, "time": weekly.time.values[valid_idx]}
