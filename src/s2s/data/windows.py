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
          rolling 7-day backward mean ending at t + 7*L
          i.e. mean of days [t + 7*L - 6 .. t + 7*L]

    ALIGNMENT (Fix 5a/M4): the lead window ENDS at t+7*L, not t+7*L+6. This makes
    the daily-strided TRAIN target the same 7 calendar days the W-MON TEST bins use:
    for an init whose most-recent history day is t, test lead-L is the weekly bin
    [t-6+7L .. t+7L] (0-day gap between history end and target start). The previous
    +6 offset made train targets end 6 days later than test -- a silent train/test
    lead mismatch.

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
    from s2s.data.assemble import (
        compute_sst_index_series, input_vars, pack_windows,
        sst_index_block, sst_index_names, sst_index_lags, target_vars,
    )

    history_weeks = int(cfg.data.history_weeks)
    lead_weeks = list(cfg.data.lead_weeks)
    max_lead = max(lead_weeks)
    idx_names = sst_index_names(cfg)
    idx_lags = sst_index_lags(cfg)
    max_idx_lag = max(idx_lags) if idx_names else 0
    in_vars = input_vars(cfg)
    out_vars = target_vars(cfg)

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
    # Latest lead window: rolled index t + 7*max_lead (ends at t+7*max_lead; Fix 5a).
    #   Needs to be < n_time.
    #   => t <= n_time - 7*max_lead - 1
    t_min = 7 * max(history_weeks - 1, max_idx_lag) + 6  # deepest lookback: history or SST-index lag
    t_max = n_time - 7 * max_lead - 1

    n_in_vars = len(in_vars)
    n_out_vars = len(out_vars)
    n_leads = len(lead_weeks)

    if t_max < t_min:
        # Split too short to produce any sample (e.g. tiny dev subset).
        return {
            "inputs": np.empty((0, history_weeks * n_in_vars + len(idx_names) * len(idx_lags) + 2, lat, lon), dtype=np.float32),
            "targets": np.empty((0, n_leads, n_out_vars, lat, lon), dtype=np.float32),
            "time": np.array([], dtype="datetime64[ns]"),
        }

    valid_idx = np.arange(t_min, t_max + 1, stride_days)

    # ---- LEAKAGE ASSERTIONS -------------------------------------------
    # oldest history index for first valid sample
    oldest_idx = int(valid_idx[0]) - 7 * max(history_weeks - 1, max_idx_lag)
    assert oldest_idx >= 6, (
        f"daily_init_weekly_windows: oldest lookback rolled index={oldest_idx} < 6 "
        f"(rolling mean would be NaN -- split boundary, embargo, or SST-index lag misconfigured)"
    )
    # latest lead index for last valid sample
    latest_idx = int(valid_idx[-1]) + 7 * max_lead
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
    # Lead L: rolled index t + 7*L  (mean of days t+7L-6 .. t+7L); aligned to the
    # W-MON test bins (Fix 5a). Ends at t+7L, NOT t+7L+6.
    lead_offsets = 7 * np.array(lead_weeks)                                  # (n_leads,)
    lead_idx = valid_idx[:, None] + lead_offsets[None, :]                    # (N, n_leads)
    out_leads = out_stack[:, lead_idx, :, :].transpose(1, 2, 0, 3, 4).astype(np.float32)

    extra = None
    if idx_names:
        idx_series = compute_sst_index_series(rolled, cfg)   # rolled = 7-day-mean SST
        extra = sst_index_block(idx_series, valid_idx, idx_lags, 7, lat, lon)

    inputs, targets = pack_windows(
        in_hist, out_leads, doy_cos[valid_idx], doy_sin[valid_idx], extra
    )
    return {"inputs": inputs, "targets": targets, "time": np.array(times[valid_idx])}
