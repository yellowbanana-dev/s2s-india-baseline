"""Reference forecasts (task #4). Every model result must beat these.

  climatology : predict the seasonal cycle => zero anomaly. The HARD baseline.
  persistence : carry the most recent observed anomaly forward to all leads.

Beating persistence but not climatology is the classic false win — report both.
"""
from __future__ import annotations
import xarray as xr


def climatology_forecast(init_time, clim, cfg) -> xr.Dataset:
    """Zero-anomaly forecast for weeks 1-6 (i.e. predict the climatology)."""
    raise NotImplementedError


def persistence_forecast(init_anomaly, cfg) -> xr.Dataset:
    """Hold the latest observed anomaly constant across all lead weeks."""
    raise NotImplementedError
