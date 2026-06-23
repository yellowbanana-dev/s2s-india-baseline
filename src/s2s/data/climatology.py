"""Stage 2b - Climatology, anomalies, normalization (task #3).

THE CARDINAL RULE LIVES HERE. Every statistic is fit on TRAIN ONLY and applied
to val/test. A climatology that includes test years inflates skill invisibly.

The climatology produced here does triple duty:
  1. de-seasonalize inputs/targets (anomaly = actual - climatology)
  2. reconstruct full fields from predicted anomalies at eval time
  3. IS the climatology baseline (see eval/baselines.py)
"""
from __future__ import annotations
import xarray as xr


def fit_climatology(train_ds: xr.Dataset, cfg) -> xr.Dataset:
    """Smoothed seasonal cycle per location and time-of-year. TRAIN years only."""
    raise NotImplementedError


def to_anomaly(ds: xr.Dataset, clim: xr.Dataset) -> xr.Dataset:
    """anomaly = actual - climatology. Precip uses cfg.data.precip_transform first."""
    raise NotImplementedError


def fit_normalizer(train_anom: xr.Dataset, cfg) -> dict:
    """Per-variable mean/std from TRAIN anomalies only. Returns stats to apply later."""
    raise NotImplementedError


def weekly_mean(ds: xr.Dataset, cfg) -> xr.Dataset:
    """Aggregate daily fields to weekly means; define lead-week targets 1..6."""
    raise NotImplementedError
