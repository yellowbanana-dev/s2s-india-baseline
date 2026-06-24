"""Stage 2c -- assemble per-sample model tensors from weekly anomalies (task #7).

Single source of truth for channel order/count, so the model's in/out_channels
and the dataset's actual tensors can never silently disagree. Targets are also
fed back as input history (the model gets to see what it's persisting from),
followed by predictors, followed by one cyclical day-of-year channel.
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
    n_samples = len(valid_idx)
    lat, lon = weekly.sizes["latitude"], weekly.sizes["longitude"]

    def _tlatlon(name):
        return weekly[name].transpose("time", "latitude", "longitude").values

    in_stack = np.nan_to_num(
        np.stack([_tlatlon(v) for v in in_vars], axis=0), nan=0.0
    )  # (n_in_vars, time, lat, lon)
    out_stack = np.stack([_tlatlon(v) for v in out_vars], axis=0)  # (n_out_vars, time, lat, lon)

    doy = weekly.time.dt.dayofyear.values.astype(np.float64)
    doy_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    n_in_vars = len(in_vars)
    inputs = np.empty((n_samples, history_weeks * n_in_vars + 1, lat, lon), dtype=np.float32)
    targets = np.empty((n_samples, len(lead_weeks), len(out_vars), lat, lon), dtype=np.float32)

    for i, t in enumerate(valid_idx):
        hist = in_stack[:, t - history_weeks + 1 : t + 1, :, :]  # (n_in_vars, history_weeks, lat, lon)
        hist = hist.transpose(1, 0, 2, 3).reshape(history_weeks * n_in_vars, lat, lon)
        inputs[i, : history_weeks * n_in_vars] = hist
        inputs[i, -1] = doy_cos[t]
        for li, lead in enumerate(lead_weeks):
            targets[i, li] = out_stack[:, t + lead, :, :]

    return {
        "inputs": inputs,
        "targets": targets,
        "time": weekly.time.values[valid_idx],
    }
