"""Stage 2b - Climatology, anomalies, normalization (task #3).

THE CARDINAL RULE LIVES HERE. Every statistic is fit on TRAIN ONLY and applied
to val/test. A climatology that includes test years inflates skill invisibly.

The climatology produced here does triple duty:
  1. de-seasonalize inputs/targets (anomaly = actual - climatology)
  2. reconstruct full fields from predicted anomalies at eval time
  3. IS the climatology baseline (see eval/baselines.py)
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from s2s.data.windows import daily_to_weekly_mean

# Precip is skewed/bounded-at-zero: always log1p before treating it as Gaussian
# anomalies. Hardcoded (not read from cfg.data.precip_transform) because both
# fit_climatology and to_anomaly must apply the IDENTICAL transform regardless
# of call order, and log1p is the only transform this project supports.
_PRECIP_VAR = "total_precipitation_24hr"
_DEFAULT_SMOOTH_DAYS = 11


def _log1p_precip(ds: xr.Dataset) -> xr.Dataset:
    if _PRECIP_VAR not in ds.data_vars:
        return ds
    ds = ds.copy()
    ds[_PRECIP_VAR] = np.log1p(ds[_PRECIP_VAR])
    return ds


def fit_climatology(train_ds: xr.Dataset, cfg) -> xr.Dataset:
    """Smoothed seasonal cycle per location and time-of-year. TRAIN years only.

    Returns a Dataset indexed by `dayofyear` (1..366), circularly smoothed so
    Dec 31 and Jan 1 connect instead of jumping.
    """
    transformed = _log1p_precip(train_ds)
    doy = transformed.time.dt.dayofyear
    raw = transformed.groupby(doy).mean("time")

    window = int(getattr(cfg.data, "climatology_smooth_days", _DEFAULT_SMOOTH_DAYS))
    n = raw.sizes["dayofyear"]
    padded = xr.concat(
        [raw.isel(dayofyear=slice(n - window, n)), raw, raw.isel(dayofyear=slice(0, window))],
        dim="dayofyear",
    )
    smoothed = padded.rolling(dayofyear=window, center=True, min_periods=1).mean()
    smoothed = smoothed.isel(dayofyear=slice(window, window + n))
    return smoothed.assign_coords(dayofyear=raw.dayofyear)


def to_anomaly(ds: xr.Dataset, clim: xr.Dataset) -> xr.Dataset:
    """anomaly = actual - climatology. Precip uses log1p first (see fit_climatology)."""
    transformed = _log1p_precip(ds)
    doy = transformed.time.dt.dayofyear
    # Dec 31 of a leap year is doy 366; climatology has no entry beyond the
    # smoothing window's max day, so clamp to the climatology's own range.
    doy = doy.clip(max=int(clim.dayofyear.max()))
    clim_for_time = clim.sel(dayofyear=doy)
    return transformed - clim_for_time


def fit_normalizer(train_anom: xr.Dataset, cfg) -> dict:
    """Per-variable mean/std from TRAIN anomalies only. Returns stats to apply later."""
    stats = {}
    for var in train_anom.data_vars:
        da = train_anom[var]
        stats[var] = {
            "mean": float(da.mean().values),
            "std": float(da.std().values),
        }
    return stats


def weekly_mean(ds: xr.Dataset, cfg) -> xr.Dataset:
    """Aggregate daily fields to weekly means; define lead-week targets 1..6."""
    return daily_to_weekly_mean(ds)
