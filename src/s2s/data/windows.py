"""Stage 2 - daily -> weekly windowing, shared by Stage 2c and eval (task #3/#4/#5).

S2S forecasts target WEEKLY MEANS at leads 1-6, but Stage 1 data is daily/6-hourly.
This module is the single place that turns a daily series into the
(init week -> lead-week target) windows, so the datamodule, baselines, and eval
scripts all slice time identically and never disagree about what "week 3" means.
"""
from __future__ import annotations

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
