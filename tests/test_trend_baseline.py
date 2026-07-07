"""Fix 3 (C1): trend-aware probabilistic climatology reference. Synthetic, torch-free."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from s2s.eval.baselines import (
    climatology_woy_ensemble,
    climatology_woy_trend_ensemble,
    fit_linear_trend,
)
from s2s.eval.metrics import crps_ensemble


def _weekly(years, lat=2, lon=3, slope=0.0, noise=0.0, seed=0):
    """Weekly TRAIN series over `years`, anomaly = slope*(decimal_year - y0) + noise.
    Constant across gridpoints; no seasonality so every week-of-year is poolable."""
    rng = np.random.default_rng(seed)
    time = pd.date_range(f"{years[0]}-01-07", f"{years[1]}-12-31", freq="7D")
    dy = time.year + (time.dayofyear - 1) / 365.25
    base = slope * (dy.values - years[0])
    field = base[:, None, None] + rng.normal(0, noise, size=(len(time), lat, lon))
    return xr.DataArray(
        field, dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": np.linspace(5, 35, lat),
                "longitude": np.linspace(70, 95, lon)},
    )


def test_fit_linear_trend_recovers_known_slope():
    tw = _weekly((1979, 2012), slope=0.05, noise=0.0)
    tr = fit_linear_trend(tw)
    assert tr.dims == ("latitude", "longitude")
    np.testing.assert_allclose(tr.values, 0.05, atol=1e-6)


def test_zero_slope_trend_reduces_to_woy_ensemble():
    """A zero trend field (explicit, or a noise-free zero-slope fit) => the trend
    ensemble is identical to the plain WOY ensemble. Isolates the shift logic from
    the OLS fit (which on noisy data legitimately finds a small non-zero slope)."""
    target_time = np.datetime64("2020-07-06")
    woy = int(pd.Timestamp(target_time).isocalendar().week)

    # (a) explicit zero-trend field on noisy data
    tw = _weekly((1979, 2012), slope=0.0, noise=0.5, seed=1)
    plain = climatology_woy_ensemble(tw, woy, window=3)
    zero_trend = xr.zeros_like(fit_linear_trend(tw))
    trended = climatology_woy_trend_ensemble(tw, woy, target_time, window=3, trend=zero_trend)
    np.testing.assert_allclose(trended.transpose(*plain.dims).values, plain.values, atol=1e-9)

    # (b) truly noise-free, zero-slope data: the fitted slope is exactly 0
    tw0 = _weekly((1979, 2012), slope=0.0, noise=0.0)
    plain0 = climatology_woy_ensemble(tw0, woy, window=3)
    trended0 = climatology_woy_trend_ensemble(tw0, woy, target_time, window=3)
    np.testing.assert_allclose(trended0.transpose(*plain0.dims).values, plain0.values, atol=1e-9)


def test_members_shifted_by_exact_trend_delta():
    """Each member m must be shifted by slope*(t_target - t_member)."""
    slope = 0.03
    tw = _weekly((1979, 2012), slope=slope, noise=0.0)
    target_time = np.datetime64("2021-06-01")
    woy = int(pd.Timestamp(target_time).isocalendar().week)
    plain = climatology_woy_ensemble(tw, woy, window=2)
    trended = climatology_woy_trend_ensemble(tw, woy, target_time, window=2)

    def _dy(t):
        i = pd.DatetimeIndex(np.atleast_1d(np.asarray(t, "datetime64[ns]")))
        return (i.year + (i.dayofyear - 1) / 365.25).values
    t_tar = _dy(target_time)[0]
    t_mem = _dy(plain["member"].values)
    expected_shift = slope * (t_tar - t_mem)  # (member,)
    got_shift = (trended.transpose(*plain.dims).values - plain.values)
    # shift is constant across lat/lon, varies by member
    np.testing.assert_allclose(got_shift[:, 0, 0], expected_shift, atol=1e-6)
    assert np.allclose(got_shift, got_shift[:, :1, :1])  # gridpoint-constant here


def test_detrended_reference_is_harder_against_trend_following_truth():
    """The decisive property: when the ONLY signal is a warming trend, the raw WOY
    pool (centred ~1995) is biased low vs a 2020 target, so a trend-capturing
    forecast beats it trivially. Detrending recentres the pool onto 2020, making it
    a HARDER bar => lower CRPS against the trend-consistent truth."""
    slope = 0.1
    tw = _weekly((1979, 2012), slope=slope, noise=0.05, seed=2)
    target_time = np.datetime64("2020-07-06")
    woy = int(pd.Timestamp(target_time).isocalendar().week)
    # truth = the trend value extrapolated to 2020 (what a warming-aware obs shows)
    dy_target = 2020 + (pd.Timestamp(target_time).dayofyear - 1) / 365.25
    truth = np.full((1, 2, 3), slope * (dy_target - 1979))

    plain = climatology_woy_ensemble(tw, woy, window=3).transpose("member", "latitude", "longitude")
    trended = climatology_woy_trend_ensemble(tw, woy, target_time, window=3).transpose(
        "member", "latitude", "longitude"
    )
    crps_plain = float(np.mean(crps_ensemble(plain.values, truth, fair=True)))
    crps_trend = float(np.mean(crps_ensemble(trended.values, truth, fair=True)))
    assert crps_trend < crps_plain  # detrended reference fits the trend truth better


def test_precomputed_trend_matches_internal_fit():
    tw = _weekly((1979, 2012), slope=0.04, noise=0.3, seed=3)
    target_time = np.datetime64("2019-09-01")
    woy = int(pd.Timestamp(target_time).isocalendar().week)
    tr = fit_linear_trend(tw)
    a = climatology_woy_trend_ensemble(tw, woy, target_time, window=3, trend=tr)
    b = climatology_woy_trend_ensemble(tw, woy, target_time, window=3, trend=None)
    np.testing.assert_allclose(a.values, b.values, atol=1e-9)
