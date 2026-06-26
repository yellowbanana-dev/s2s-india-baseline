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


# --------------------------------------------------------------------------- #
# Phase-B eval upgrade: CRPSS, calibration (rank histogram / spread-error),    #
# reliability, week-of-year climatology pooling. Hand-verified, numpy API.     #
# --------------------------------------------------------------------------- #

from s2s.eval.baselines import climatology_woy_ensemble
from s2s.eval.metrics import (
    crpss,
    event_probability,
    rank_histogram,
    reliability_curve,
    spread_error_ratio,
)


def test_crpss_known_value():
    assert abs(crpss(0.5, 1.0) - 0.5) < 1e-12       # 1 - 0.5/1.0
    assert abs(crpss(0.8, 0.8) - 0.0) < 1e-12       # equal -> zero skill
    assert crpss(1.2, 1.0) < 0.0                    # worse -> negative (precip case)
    assert np.isnan(crpss(0.5, 0.0))                # guard against /0


def test_rank_histogram_calibrated_is_flat():
    rng = np.random.default_rng(0)
    M, N = 20, 4000
    members = rng.normal(size=(M, N))
    truth = rng.normal(size=N)               # truth ~ same dist as members
    counts = rank_histogram(members, truth)
    assert counts.shape == (M + 1,)
    freq = counts / counts.sum()
    assert freq.max() < 2.0 / (M + 1)        # no bin wildly over uniform
    assert freq.min() > 0.3 / (M + 1)


def test_rank_histogram_underdispersed_is_u_shaped():
    rng = np.random.default_rng(1)
    M, N = 20, 4000
    members = 0.1 * rng.normal(size=(M, N))  # tight members
    truth = rng.normal(size=N)               # wide truth
    counts = rank_histogram(members, truth)
    edges = counts[0] + counts[-1]
    middle = counts[1:-1].sum()
    assert edges > middle                    # U-shape: ends dominate interior


def test_spread_error_ratio_underdispersed_below_one():
    rng = np.random.default_rng(2)
    M = 15
    members = 0.01 * rng.normal(size=(M, 50))
    truth = np.ones(50)
    assert spread_error_ratio(members, truth) < 0.5


def test_reliability_curve_perfect_event():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    centers, obs, counts = reliability_curve(p, y, n_bins=10)
    assert np.isnan(obs[5])                  # empty middle bin
    assert obs[0] == 0.0
    assert obs[-1] == 1.0


def test_event_probability_fraction():
    members = np.array([0.0, 2.0, 4.0, 6.0])
    assert abs(float(event_probability(members, 3.0, "gt")) - 0.5) < 1e-12
    assert abs(float(event_probability(members, 3.0, "lt")) - 0.5) < 1e-12


def test_climatology_woy_window_selects_in_season():
    time = pd.date_range("2001-01-01", periods=104, freq="7D")
    da = xr.DataArray(np.arange(104, dtype=float), dims=["time"], coords={"time": time})
    ens = climatology_woy_ensemble(da, target_woy=26, window=2)
    assert ens.sizes["member"] > 0
    member_times = ens["member"].values
    woy = xr.DataArray(member_times).dt.isocalendar().week.values
    d = np.abs(woy - 26)
    d = np.minimum(d, 52 - d)
    assert (d <= 2).all()                    # every pooled member in-season
