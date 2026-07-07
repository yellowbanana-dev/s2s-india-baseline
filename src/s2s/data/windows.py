"""Stage 2 - daily -> weekly windowing, shared by Stage 2c and eval (task #3/#4/#5).

S2S forecasts target WEEKLY MEANS at leads 1-6, but Stage 1 data is daily/6-hourly.
This module is the single place that turns a daily series into the
(init week -> lead-week target) windows, so the datamodule, baselines, and eval
scripts all slice time identically and never disagree about what "week 3" means.

daily_to_weekly_mean / build_lead_targets  -- W-MON bins (eval, baselines, test split)
daily_init_weekly_windows                  -- daily-strided rolling windows (train/val)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def daily_to_weekly_mean(ds: xr.Dataset, anchor: str = "W-MON") -> xr.Dataset:
    """Non-overlapping 7-day means, anchored consistently across the whole project.

    `anchor` fixes which weekday a week starts on so that "week 3" means the same
    7 calendar days no matter which split or script computes it.
    """
    return ds.resample(time=anchor, label="left", closed="left").mean()


def build_lead_targets(weekly: xr.Dataset, lead_weeks: list[int]) -> xr.Dataset:
    """Stack per-lead targets along a new 'lead' dim, indexed by INIT week.

    `targets.sel(lead=L, time=t)` holds the actual week-(t+L) mean -- i.e. exactly
    what a forecast initialized at week `t` for lead `L` must predict. Edge weeks
    with no full lead window available come back as NaN (drop before scoring).
    """
    leads = [weekly.shift(time=-lead) for lead in lead_weeks]
    lead_dim = xr.DataArray(list(lead_weeks), dims="lead", name="lead")
    return xr.concat(leads, dim=lead_dim)


def valid_init_weeks(weekly_time: xr.DataArray, max_lead: int) -> xr.DataArray:
    """Init weeks that have a full lead window before the series ends."""
    n = weekly_time.sizes["time"]
    if max_lead <= 0:
        return weekly_time
    return weekly_time.isel(time=slice(0, n - max_lead))


def daily_init_weekly_windows(
    daily_ds: xr.Dataset,
    cfg,
    stride_days: int = 1,
) -> dict:
    """Dense daily-strided sample builder for TRAIN/VAL splits only.

    For each valid init date t (stepped by stride_days), computes:
      History week h (h=0 oldest, h=history_weeks-1 most recent):
          rolling 7-day backward mean ending at t - 7*h
          i.e. mean of days [t - 7*h - 6 .. t - 7*h]
      Lead week L  (L from cfg.data.lead_weeks):
          rolling 7-day backward mean ending at t + 7*L + 6
          i.e. mean of days [t + 7*L .. t + 7*L + 6]

    Uses rolling(time=7, min_periods=7): the first 6 positions of the rolled
    array are NaN, which naturally prevents any sample from using days that
    predate the supplied split's data.  The caller MUST pass a single split's
    daily data -- cross-split leakage is structurally impossible because
    daily_ds contains only that split's days.

    LEAKAGE GUARD: assertions verify that the valid-index bounds never reach
    into the rolled-NaN prefix (oldest history) or past the array end (latest
    lead).  Fail loudly if violated.

    Shape contract is identical to assemble_arrays():
      inputs  (N, history_weeks*n_in_vars+1, lat, lon) float32
      targets (N, n_lead_weeks, n_out_vars, lat, lon)   float32
      time    (N,) datetime64  -- init dates
    """
    from s2s.data.assemble import _SST_VAR, input_vars, pack_windows, sst_extra_lags, target_vars

    history_weeks = int(cfg.data.history_weeks)
    lead_weeks = list(cfg.data.lead_weeks)
    max_lead = max(lead_weeks)
    in_vars = input_vars(cfg)
    out_vars = target_vars(cfg)
    lags = sst_extra_lags(cfg)
    max_lag_w = max(lags) if lags else 0  # deepest extra SST lag, in weeks

    # 7-day backward rolling mean: rolled[d] = mean(days d-6 .. d).
    # NaN for d < 6 (< 7 days available from the split start).
    rolled = daily_ds.rolling(time=7, min_periods=7).mean()
    times = pd.DatetimeIndex(rolled.time.values)
    n_time = len(times)
    lat = rolled.sizes["latitude"]
    lon = rolled.sizes["longitude"]

    def _tlatlon(name: str) -> np.ndarray:
        return rolled[name].transpose("time", "latitude", "longitude").values  # (T, lat, lon)

    in_stack = np.nan_to_num(
        np.stack([_tlatlon(v) for v in in_vars], axis=0), nan=0.0
    )  # (n_in_vars, T, lat, lon)
    out_stack = np.stack([_tlatlon(v) for v in out_vars], axis=0)  # (n_out_vars, T, lat, lon)

    doy_vals = times.dayofyear.values.astype(np.float64)
    doy_cos = np.cos(2 * np.pi * doy_vals / 365.25).astype(np.float32)
    doy_sin = np.sin(2 * np.pi * doy_vals / 365.25).astype(np.float32)

    # ---- valid init-index range ----------------------------------------
    # Oldest history window: rolled index t - 7*(history_weeks-1).
    #   Needs to be >= 6 so the 7-day rolling mean is not NaN.
    #   => t >= 7*(history_weeks-1) + 6
    # Latest lead window: rolled index t + 7*max_lead + 6.
    #   Needs to be < n_time.
    #   => t <= n_time - 7*max_lead - 7
    t_min = 7 * max(history_weeks - 1, max_lag_w) + 6  # deepest lookback: history or SST lag
    t_max = n_time - 7 * max_lead - 7

    n_in_vars = len(in_vars)
    n_out_vars = len(out_vars)
    n_leads = len(lead_weeks)

    if t_max < t_min:
        # Split too short to produce any sample (e.g. tiny dev subset).
        return {
            "inputs": np.empty((0, history_weeks * n_in_vars + len(lags) + 2, lat, lon), dtype=np.float32),
            "targets": np.empty((0, n_leads, n_out_vars, lat, lon), dtype=np.float32),
            "time": np.array([], dtype="datetime64[ns]"),
        }

    valid_idx = np.arange(t_min, t_max + 1, stride_days)

    # ---- LEAKAGE ASSERTIONS -------------------------------------------
    # oldest lookback index for first valid sample (history OR deepest SST lag)
    oldest_idx = int(valid_idx[0]) - 7 * max(history_weeks - 1, max_lag_w)
    assert oldest_idx >= 6, (
        f"daily_init_weekly_windows: oldest lookback rolled index={oldest_idx} < 6 "
        f"(rolling mean would be NaN -- split boundary, embargo, or SST lag misconfigured)"
    )
    # latest lead index for last valid sample
    latest_idx = int(valid_idx[-1]) + 7 * max_lead + 6
    assert latest_idx < n_time, (
        f"daily_init_weekly_windows: latest lead rolled index={latest_idx} >= n_time={n_time} "
        f"(lead window exceeds split data -- cross-boundary leakage)"
    )

    # ---- build history windows (N, history_weeks, n_in_vars, lat, lon) ----
    # h=0: oldest, offset = -7*(history_weeks-1);  h=hw-1: most recent, offset=0
    hist_offsets = -7 * (history_weeks - 1) + 7 * np.arange(history_weeks)  # (hw,)
    hist_idx = valid_idx[:, None] + hist_offsets[None, :]                    # (N, hw)
    # in_stack[:, hist_idx] -> (n_in_vars, N, hw, lat, lon)
    in_hist = in_stack[:, hist_idx, :, :].transpose(1, 2, 0, 3, 4).astype(np.float32)

    # ---- build lead windows (N, n_leads, n_out_vars, lat, lon) -----------
    # Lead L: rolled index t + 7*L + 6  (mean of days t+7L .. t+7L+6)
    lead_offsets = 7 * np.array(lead_weeks) + 6                              # (n_leads,)
    lead_idx = valid_idx[:, None] + lead_offsets[None, :]                    # (N, n_leads)
    out_leads = out_stack[:, lead_idx, :, :].transpose(1, 2, 0, 3, 4).astype(np.float32)

    # ---- extra SST-history channels at week-lags (ADR-0006) --------------
    if lags:
        sst_series = np.nan_to_num(_tlatlon(_SST_VAR), nan=0.0)     # (T, lat, lon)
        sst_idx = valid_idx[:, None] - 7 * np.array(lags)[None, :]  # (N, n_lags) daily indices
        sst_extra = sst_series[sst_idx].astype(np.float32)         # (N, n_lags, lat, lon)
    else:
        sst_extra = None

    inputs, targets = pack_windows(
        in_hist, out_leads, doy_cos[valid_idx], doy_sin[valid_idx], sst_extra
    )
    return {"inputs": inputs, "targets": targets, "time": np.array(times[valid_idx])}
