"""Spread-calibration attribution helpers (torch-free).

Pins the properties 03_evaluate.py relies on when it reports crpss_vs_prob_cal:
inflation leaves the ensemble MEAN fixed (so ACC/RMSE are untouched), scales the
spread-error ratio EXACTLY by alpha, and commutes with spatial subsetting.
"""
import numpy as np
import pytest

from s2s.eval.metrics import (
    inflate_ensemble_spread,
    spread_error_ratio,
    spread_inflation_factor,
)


def _case(seed=0, m=16, n=5, lat=6, lon=6, spread=0.4):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(n, lat, lon))
    # deliberately under-dispersed: members tight around a biased mean
    mean = truth + rng.normal(scale=0.8, size=(n, lat, lon))
    members = mean[None, ...] + rng.normal(scale=spread, size=(m, n, lat, lon))
    return members, truth


def test_alpha_one_is_identity():
    members, _ = _case()
    np.testing.assert_allclose(inflate_ensemble_spread(members, 1.0), members, rtol=0, atol=0)


def test_ensemble_mean_is_preserved():
    """The mean must not move -- otherwise ACC/RMSE would change and the attribution
    would no longer hold 'deterministic skill fixed'."""
    members, _ = _case()
    out = inflate_ensemble_spread(members, 1.7)
    np.testing.assert_allclose(out.mean(axis=0), members.mean(axis=0), rtol=1e-12, atol=1e-12)


def test_ser_scales_exactly_by_alpha():
    members, truth = _case()
    ser0 = spread_error_ratio(members, truth)
    for a in (0.5, 1.3, 2.0):
        ser_a = spread_error_ratio(inflate_ensemble_spread(members, a), truth)
        assert np.isclose(ser_a, a * ser0, rtol=1e-10, atol=1e-12)


def test_inflation_factor_round_trip_hits_target():
    """The property the diagnostic depends on: inflating by spread_inflation_factor
    makes the recomputed SER equal the requested target."""
    members, truth = _case()
    assert spread_error_ratio(members, truth) < 1.0          # under-dispersed by construction
    for target in (1.0, 0.8):
        a = spread_inflation_factor(members, truth, target)
        assert np.isfinite(a)
        ser_cal = spread_error_ratio(inflate_ensemble_spread(members, a), truth)
        assert np.isclose(ser_cal, target, rtol=1e-10, atol=1e-12)


def test_commutes_with_spatial_subsetting():
    """03_evaluate inflates the FULL field then boxes it, while alpha is derived from the
    BOX. That is only valid because inflation is pointwise over the member axis."""
    members, _ = _case()
    box = (slice(None), slice(None), slice(1, 4), slice(2, 5))
    a = 1.42
    np.testing.assert_allclose(
        inflate_ensemble_spread(members, a)[box],
        inflate_ensemble_spread(members[box], a),
        rtol=1e-12, atol=1e-12,
    )


def test_single_member_and_nonfinite_alpha_are_no_ops():
    members, truth = _case(m=1)
    np.testing.assert_allclose(inflate_ensemble_spread(members, 3.0), members)
    assert not np.isfinite(spread_inflation_factor(members, truth))   # SER undefined at M=1
    members2, _ = _case()
    np.testing.assert_allclose(inflate_ensemble_spread(members2, np.nan), members2)
