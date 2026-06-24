"""Phase-A eval layer tests (task #4/#5 guardrails).

VERIFY EACH METRIC against a hand-computed tiny example before trusting it --
a CRPS/ACC off by a normalization makes everything look great and never errors.
No network: everything runs on small synthetic arrays/datasets.
"""
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf

from s2s.eval.baselines import climatology_forecast, persistence_forecast
from s2s.eval.metrics import acc, crps_ensemble


def _make_cfg():
    return OmegaConf.create({"data": {"lead_weeks": [1, 2, 3]}})


def test_crps_perfect_ensemble_is_zero():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(5, 5))
    # Every member equals the truth -> zero spread, zero error -> CRPS == 0.
    forecast_members = np.broadcast_to(truth, (8, *truth.shape))
    out = crps_ensemble(forecast_members, truth)
    np.testing.assert_allclose(out, 0.0, atol=1e-10)


def test_crps_deterministic_reduces_to_abs_error():
    truth = np.array([[1.0, -2.0], [3.0, 0.0]])
    forecast = np.array([[0.5, -1.0], [3.0, 1.0]])
    forecast_members = forecast[np.newaxis, ...]  # single member
    out = crps_ensemble(forecast_members, truth)
    np.testing.assert_allclose(out, np.abs(forecast - truth))


def test_acc_perfect_correlation_is_one():
    truth = np.array([1.0, 2.0, -1.0, 3.0, -2.0])
    out = acc(truth, truth)
    assert abs(out - 1.0) < 1e-10


def test_climatology_forecast_is_zero_anomaly():
    cfg = _make_cfg()
    lat = np.linspace(-10, 10, 3)
    lon = np.linspace(0, 20, 4)
    doy = np.arange(1, 367)
    clim = xr.Dataset(
        {"2m_temperature": (("dayofyear", "latitude", "longitude"), np.ones((366, 3, 4)))},
        coords={"dayofyear": doy, "latitude": lat, "longitude": lon},
    )
    out = climatology_forecast(pd.Timestamp("2018-01-07"), clim, cfg)
    assert set(out.dims) >= {"lead", "time", "latitude", "longitude"}
    assert out.sizes["lead"] == 3
    np.testing.assert_allclose(out["2m_temperature"].values, 0.0)


def test_persistence_forecast_holds_latest_value():
    cfg = _make_cfg()
    lat = np.linspace(-10, 10, 3)
    lon = np.linspace(0, 20, 4)
    values = np.arange(12).reshape(3, 4).astype(float)
    init_anomaly = xr.Dataset(
        {"2m_temperature": (("latitude", "longitude"), values)},
        coords={"latitude": lat, "longitude": lon},
    )
    out = persistence_forecast(init_anomaly, cfg)
    assert out.sizes["lead"] == 3
    for lead in cfg.data.lead_weeks:
        np.testing.assert_allclose(out["2m_temperature"].sel(lead=lead).values, values)
