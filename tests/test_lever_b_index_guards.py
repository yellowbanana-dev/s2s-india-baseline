"""MIN-7 (review 2026-07-14): lever-b SST-index silent-failure guards.

The hazard: an empty index box (or an unknown index name) produced a quiet null result
instead of an error. These reproduce both failure modes on synthetic data.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from s2s.data.assemble import (
    _SST_VAR,
    compute_sst_index_series,
    sst_index_names,
)


def _cfg(indices, sst_predictor=True):
    return SimpleNamespace(
        data=SimpleNamespace(
            sst_indices=indices,
            variables=SimpleNamespace(
                predictors=SimpleNamespace(
                    surface=[_SST_VAR] if sst_predictor else ["geopotential"],
                    levels={},
                )
            ),
        )
    )


def _sst_ds(lon_0_360=True, nt=12, nlat=32, nlon=64, all_nan=False):
    lat = -87.1875 + 5.625 * np.arange(nlat)
    lon = 5.625 * np.arange(nlon)          # 0..354.375
    if not lon_0_360:
        lon = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(lon)
        lon = lon[order]
    rng = np.random.default_rng(0)
    data = rng.standard_normal((nt, nlat, nlon)).astype("float32")
    if all_nan:
        data[:] = np.nan
    t = np.datetime64("2000-01-01", "ns") + np.arange(nt) * np.timedelta64(7, "D")
    return xr.Dataset(
        {_SST_VAR: (("time", "latitude", "longitude"), data)},
        coords={"time": t, "latitude": lat, "longitude": lon},
    )


# ---------------- unknown index names must not be silently dropped ----------------

def test_unknown_index_name_raises():
    with pytest.raises(ValueError, match="unknown sst_indices"):
        sst_index_names(_cfg(["nino_34"]))          # typo for "nino34"


def test_unknown_name_raises_even_without_sst_predictor():
    # a config typo is a config error regardless of whether SST is wired in
    with pytest.raises(ValueError, match="unknown sst_indices"):
        sst_index_names(_cfg(["nino_34"], sst_predictor=False))


def test_known_names_pass_through_in_order():
    assert sst_index_names(_cfg(["nino34", "dmi"])) == ["nino34", "dmi"]


def test_empty_list_is_still_off():
    assert sst_index_names(_cfg([])) == []


def test_sst_not_a_predictor_disables_lever():
    assert sst_index_names(_cfg(["nino34"], sst_predictor=False)) == []


# ---------------- empty box must raise, not yield a zero channel ----------------

def test_empty_box_on_minus180_store_raises():
    ds = _sst_ds(lon_0_360=False)
    with pytest.raises(ValueError, match="selects NO grid cells"):
        compute_sst_index_series(ds, _cfg(["nino34"]))


def test_normal_0_360_store_produces_finite_series():
    ds = _sst_ds(lon_0_360=True)
    out = compute_sst_index_series(ds, _cfg(["nino34", "dmi"]))
    assert out.shape == (2, ds.sizes["time"])
    assert np.isfinite(out).all()
    assert not np.allclose(out, 0.0)      # the silent-null signature


def test_all_nan_sst_raises_instead_of_zero_channel():
    ds = _sst_ds(lon_0_360=True, all_nan=True)
    with pytest.raises(ValueError, match="finite for only"):
        compute_sst_index_series(ds, _cfg(["nino34"]))
