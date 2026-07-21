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


# --- SST teleconnection indices (Phase-C lever b) -------------------------------
# Low-dim ENSO/IOD boundary signal as globally-broadcast channels, computed from the
# train-standardized SST anomaly field (leakage-safe; monotone proxy for the
# conventional raw-anomaly indices). Boxes are (lat_min, lat_max, lon_min, lon_max,
# sign); lon in 0-360.
_SST_VAR = "sea_surface_temperature"
_SST_INDEX_BOXES = {
    "nino34": [(-5.0, 5.0, 190.0, 240.0, 1.0)],                       # ENSO, Nino 3.4
    "dmi": [(-10.0, 10.0, 50.0, 70.0, 1.0), (-10.0, 0.0, 90.0, 110.0, -1.0)],  # IOD DMI = W - E
}


def sst_index_names(cfg) -> list[str]:
    names = list(getattr(cfg.data, "sst_indices", None) or [])
    if not names:
        return []
    # MIN-7 (review 2026-07-14): unknown names were silently dropped, so a typo
    # (e.g. "nino_34" for "nino34") disabled the lever with no error and produced a
    # quiet null result. Validate BEFORE the predictor gate so config typos surface
    # even when SST is not wired in as a predictor.
    unknown = [n for n in names if n not in _SST_INDEX_BOXES]
    if unknown:
        raise ValueError(
            f"unknown sst_indices {unknown}; valid names are {sorted(_SST_INDEX_BOXES)}. "
            "Unknown names used to be dropped silently, which turns the lever off without "
            "any error."
        )
    if _SST_VAR not in list(cfg.data.variables.predictors.surface or []):
        return []
    return names


def sst_index_lags(cfg) -> list[int]:
    return [int(l) for l in (getattr(cfg.data, "sst_index_lags_weeks", None) or [0])]


def n_sst_index_channels(cfg) -> int:
    return len(sst_index_names(cfg)) * len(sst_index_lags(cfg))


def _box_mean(sst_da, lat_min, lat_max, lon_min, lon_max):
    """cos(lat)-weighted mean of an SST DataArray over a lat/lon box (skips land NaN)."""
    lat = sst_da["latitude"]
    lon = sst_da["longitude"]
    mask = (
        (lat >= min(lat_min, lat_max)) & (lat <= max(lat_min, lat_max))
        & (lon >= lon_min) & (lon <= lon_max)
    )
    # MIN-7 (review 2026-07-14): an EMPTY box makes where() all-NaN, the weighted mean NaN,
    # and nan_to_num downstream turns it into a silently all-zero index channel -- i.e. a
    # quiet null result. The boxes are defined on a 0-360 longitude convention, so a
    # -180..180 store empties them. Confirmed empirically by the reviewer.
    if int(mask.sum()) == 0:
        raise ValueError(
            f"SST index box (lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}) selects NO "
            f"grid cells. Store longitudes span {float(lon.min())}..{float(lon.max())} and "
            "latitudes span "
            f"{float(lat.min())}..{float(lat.max())}. Index boxes use the 0-360 longitude "
            "convention; a -180..180 store yields an empty box and a silently zero channel."
        )
    return sst_da.where(mask).weighted(np.cos(np.deg2rad(lat))).mean(
        ("latitude", "longitude"), skipna=True
    )


def compute_sst_index_series(ds, cfg) -> np.ndarray:
    """(n_index_names, T) signed area-weighted SST-index time series over `ds`."""
    names = sst_index_names(cfg)
    if not names:
        return np.zeros((0, ds.sizes["time"]), dtype=np.float32)
    sst = ds[_SST_VAR]
    series = []
    for name in names:
        total = None
        for (la0, la1, lo0, lo1, sign) in _SST_INDEX_BOXES[name]:
            bm = sign * _box_mean(sst, la0, la1, lo0, lo1)
            total = bm if total is None else total + bm
        # MIN-7: even with a non-empty box, an all-land (all-NaN) box or a mostly-missing
        # SST field would be flattened to zeros by nan_to_num below. Require the index to be
        # finite for the large majority of timesteps before that flattening hides it.
        vals = total.values.astype(np.float32)
        finite_frac = float(np.isfinite(vals).mean()) if vals.size else 0.0
        if finite_frac < 0.95:
            raise ValueError(
                f"SST index '{name}' is finite for only {finite_frac:.1%} of timesteps "
                "(require >=95%). nan_to_num would turn this into a near-constant zero "
                "channel, silently disabling the lever instead of failing."
            )
        series.append(np.nan_to_num(vals, nan=0.0))
    return np.stack(series, axis=0)


def sst_index_block(idx_series, valid_idx, lags, step, lat, lon):
    """Broadcast per-sample index values to (N, n_names*n_lags, lat, lon).

    `step` = 1 for weekly index units, 7 for daily. Channel order lag-major.
    """
    cols = [idx_series[:, valid_idx - step * lag].T for lag in lags]  # each (N, n_names)
    stacked = np.concatenate(cols, axis=1).astype(np.float32)         # (N, n_names*n_lags)
    N, C = stacked.shape
    return np.broadcast_to(stacked[:, :, None, None], (N, C, lat, lon)).astype(np.float32)


def in_out_channels(cfg) -> tuple[int, int]:
    """(in_channels, out_channels) for building the model -- matches assemble_arrays exactly."""
    history_weeks = int(cfg.data.history_weeks)
    in_channels = history_weeks * len(input_vars(cfg)) + n_sst_index_channels(cfg) + 2  # + SST indices + 2 seasonal (cos, sin)
    out_channels = len(target_vars(cfg))
    return in_channels, out_channels


def pack_windows(
    in_hist: np.ndarray,
    out_leads: np.ndarray,
    doy_cos_vals: np.ndarray,
    doy_sin_vals: np.ndarray,
    extra: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack pre-built per-sample history and lead windows into model-ready tensors.

    in_hist:      (N, history_weeks, n_in_vars, lat, lon)  — oldest week first
    out_leads:    (N, n_lead_weeks,  n_out_vars, lat, lon)
    doy_cos_vals: (N,) float32 cosine day-of-year at each init time
    doy_sin_vals: (N,) float32 sine   day-of-year at each init time

    Returns:
      inputs  (N, history_weeks*n_in_vars + 2, lat, lon)  float32
      targets (N, n_lead_weeks, n_out_vars, lat, lon)      float32

    Channel order: [oldest_week_var0, ..., newest_week_var0, ...] + doy_cos + doy_sin.
    The (cos, sin) PAIR is unambiguous over the year (Fix 5b/M5): cos alone folds
    Jan/Dec together (arccos in [0,pi]); the adapter recovers the true phase via
    atan2(sin, cos). This is the ONLY place that defines the channel layout;
    in_out_channels() counts it.
    """
    N, hw, niv, lat, lon = in_hist.shape
    flat_hist = in_hist.reshape(N, hw * niv, lat, lon).astype(np.float32)
    parts = [flat_hist]
    if extra is not None and extra.shape[1] > 0:
        parts.append(np.nan_to_num(extra, nan=0.0).astype(np.float32))
    body = np.concatenate(parts, axis=1) if len(parts) > 1 else flat_hist
    nb = body.shape[1]
    inputs = np.empty((N, nb + 2, lat, lon), dtype=np.float32)
    inputs[:, :nb] = body
    inputs[:, nb] = doy_cos_vals[:, None, None]      # broadcast over lat, lon
    inputs[:, nb + 1] = doy_sin_vals[:, None, None]  # seasonal pair stays LAST
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

    idx_names = sst_index_names(cfg)
    idx_lags = sst_index_lags(cfg)
    n_time = weekly.sizes["time"]
    lo = max([history_weeks - 1] + (idx_lags if idx_names else []))  # deepest lookback
    hi = n_time - max_lead
    if hi <= lo:
        raise ValueError(
            f"not enough weeks ({n_time}) for history_weeks={history_weeks} "
            f"and max lead={max_lead}"
        )
    valid_idx = np.arange(lo, hi)

    def _tlatlon(name):
        return weekly[name].transpose("time", "latitude", "longitude").values

    in_stack = np.nan_to_num(
        np.stack([_tlatlon(v) for v in in_vars], axis=0), nan=0.0
    )  # (n_in_vars, T, lat, lon)
    out_stack = np.stack([_tlatlon(v) for v in out_vars], axis=0)  # (n_out_vars, T, lat, lon)

    doy = weekly.time.dt.dayofyear.values.astype(np.float64)
    doy_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    doy_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)

    # Vectorised extraction: build (N, history_weeks, n_in_vars, lat, lon) in one shot.
    # Offsets: 0 = oldest week, history_weeks-1 = most recent (same as the old t-hw+1..t+1 slice)
    hist_offsets = np.arange(history_weeks)  # [0, 1, ..., hw-1]
    hist_idx = valid_idx[:, None] - (history_weeks - 1) + hist_offsets[None, :]  # (N, hw)
    # in_stack[:, hist_idx] -> (n_in_vars, N, hw, lat, lon) -> transpose -> (N, hw, n_in_vars, lat, lon)
    in_hist = in_stack[:, hist_idx, :, :].transpose(1, 2, 0, 3, 4)

    lead_idx = valid_idx[:, None] + np.array(lead_weeks)[None, :]  # (N, n_leads)
    # out_stack[:, lead_idx] -> (n_out_vars, N, n_leads, lat, lon) -> (N, n_leads, n_out_vars, lat, lon)
    out_leads = out_stack[:, lead_idx, :, :].transpose(1, 2, 0, 3, 4)

    extra = None
    if idx_names:
        lat, lon = weekly.sizes["latitude"], weekly.sizes["longitude"]
        idx_series = compute_sst_index_series(weekly, cfg)
        extra = sst_index_block(idx_series, valid_idx, idx_lags, 1, lat, lon)

    inputs, targets = pack_windows(
        in_hist, out_leads, doy_cos[valid_idx], doy_sin[valid_idx], extra
    )
    return {"inputs": inputs, "targets": targets, "time": weekly.time.values[valid_idx]}
