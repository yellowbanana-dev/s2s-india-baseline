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
