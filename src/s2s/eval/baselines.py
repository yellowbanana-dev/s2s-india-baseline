"""Reference forecasts (task #4). Every model result must beat these.

  climatology : predict the seasonal cycle => zero anomaly. The HARD baseline.
  persistence : carry the most recent observed anomaly forward to all leads.

Beating persistence but not climatology is the classic false win — report both.
"""
from __future__ import annotations
import xarray as xr


def _lead_dim(cfg) -> xr.DataArray:
    return xr.DataArray(list(cfg.data.lead_weeks), dims="lead", name="lead")


def climatology_forecast(init_time, clim, cfg) -> xr.Dataset:
    """Zero-anomaly forecast for weeks 1-6 (i.e. predict the climatology).

    `clim` only supplies the variable set/template (its `dayofyear` dim is
    dropped) -- in anomaly space, "predict the climatology" IS "predict zero",
    by construction of `to_anomaly`.
    """
    template = clim.isel(dayofyear=0, drop=True)
    zero = xr.zeros_like(template)
    forecast = xr.concat([zero for _ in cfg.data.lead_weeks], dim=_lead_dim(cfg))
    return forecast.expand_dims(time=[init_time])


def persistence_forecast(init_anomaly, cfg) -> xr.Dataset:
    """Hold the latest observed anomaly constant across all lead weeks."""
    forecast = xr.concat([init_anomaly for _ in cfg.data.lead_weeks], dim=_lead_dim(cfg))
    return forecast


def _week_of_year(times):
    """ISO week index in [1, 53] from a datetime64 array."""
    import numpy as np
    return xr.DataArray(times).dt.isocalendar().week.values.astype(np.int64)


def climatology_woy_ensemble(train_weekly, target_woy: int, window: int = 3,
                             member_dim: str = "member"):
    """Seasonality-aware probabilistic climatology for ONE target week-of-year.

    The honest Phase-B reference (ADR-0001 / phase-b-plan): pools TRAIN weekly
    anomalies whose week-of-year lies within +/-`window` weeks (circular over the
    ~52-week year) of the target's week-of-year, returned along `member`.

    Why windowed, not all-pooled: even in de-seasonalized anomaly space the
    *variance* of weekly anomalies is itself seasonal (monsoon precip spread >>
    dry-season). A +/-3-week window keeps the reference local in season -- an
    honest, harder bar -- while retaining ~7 weeks x #train-years members for a
    stable CRPS. Train-only => no leakage.

    train_weekly : (time, lat, lon) DataArray of TRAIN weekly anomalies (one var).
    target_woy   : int week-of-year (1..53) of the verification target.
    """
    import numpy as np
    woy = _week_of_year(train_weekly["time"].values)
    d = np.abs(woy - int(target_woy))
    d = np.minimum(d, 52 - d)            # circular distance on a 52-week ring
    sel = np.where(d <= window)[0]
    pool = train_weekly.isel(time=sel)
    return pool.rename({"time": member_dim})
