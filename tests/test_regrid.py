import numpy as np
import xarray as xr

from s2s.eval.regrid import (
    conservative_matrices,
    equiangular_grid,
    latitude_area_weights,
    regrid_conservative,
    regrid_conservative_da,
)

SRC = equiangular_grid(1.5, with_poles=True)      # 121 x 240 (WB2 1.5deg)
DST = equiangular_grid(5.625, with_poles=False)   # 32 x 64  (WB2 5.625deg)


def test_grid_shapes_match_wb2_stores():
    assert SRC[0].size == 121 and SRC[1].size == 240
    assert DST[0].size == 32 and DST[1].size == 64
    assert np.isclose(SRC[0][0], -90.0) and np.isclose(SRC[0][-1], 90.0)      # with poles
    assert DST[0][0] > -90.0 and DST[0][-1] < 90.0                            # no poles


def test_partition_of_unity():
    w_lat, w_lon = conservative_matrices(SRC[0], SRC[1], DST[0], DST[1])
    assert np.allclose(w_lat.sum(axis=1), 1.0)
    assert np.allclose(w_lon.sum(axis=1), 1.0)


def test_constant_preserved():
    f = np.full((SRC[0].size, SRC[1].size), 3.14159)
    out = regrid_conservative(f, SRC[0], SRC[1], DST[0], DST[1])
    assert np.allclose(out, 3.14159)


def test_identity_when_grids_match():
    lat, lon = DST
    rng = np.random.default_rng(0)
    f = rng.standard_normal((lat.size, lon.size))
    out = regrid_conservative(f, lat, lon, lat, lon)
    assert np.allclose(out, f, atol=1e-10)


def test_area_weighted_mean_conserved():
    rng = np.random.default_rng(1)
    f = rng.standard_normal((SRC[0].size, SRC[1].size))
    out = regrid_conservative(f, SRC[0], SRC[1], DST[0], DST[1])
    a_src = latitude_area_weights(SRC[0])
    a_dst = latitude_area_weights(DST[0])
    m_src = (f.mean(axis=-1) * a_src).sum() / a_src.sum()
    m_dst = (out.mean(axis=-1) * a_dst).sum() / a_dst.sum()
    assert abs(m_src - m_dst) < 1e-10


def test_da_preserves_leading_dim():
    rng = np.random.default_rng(2)
    da = xr.DataArray(
        rng.standard_normal((4, SRC[0].size, SRC[1].size)),
        dims=("time", "latitude", "longitude"),
        coords={"latitude": SRC[0], "longitude": SRC[1]},
    )
    out = regrid_conservative_da(da, DST[0], DST[1])
    assert out.dims == ("time", "latitude", "longitude")
    assert out.shape == (4, DST[0].size, DST[1].size)
    assert np.allclose(out.latitude.values, DST[0])


def test_da_handles_dayofyear_lonlat_order():
    # climatology.zarr stores (dayofyear, longitude, latitude) -- non-standard axis order.
    rng = np.random.default_rng(3)
    da = xr.DataArray(
        rng.standard_normal((5, SRC[1].size, SRC[0].size)),
        dims=("dayofyear", "longitude", "latitude"),
        coords={"latitude": SRC[0], "longitude": SRC[1]},
    )
    out = regrid_conservative_da(da, DST[0], DST[1])
    assert out.shape == (5, DST[0].size, DST[1].size)


# --- NaN-safety (regression for the MAJ-3 identity-control failure) ---
# A plain matmul regrid propagates NaN through ZERO weights (0.0*NaN==NaN), so one missing
# cell turned an entire eval field NaN and every decision gate reported FAIL. These lock the
# NaN-aware renormalised behaviour in.

def test_single_nan_does_not_contaminate_identity():
    lat, lon = DST
    rng = np.random.default_rng(7)
    f = rng.standard_normal((lat.size, lon.size))
    f[3, 5] = np.nan
    out = regrid_conservative(f, lat, lon, lat, lon)
    assert np.isnan(out[3, 5])                       # the missing cell stays missing
    assert np.isnan(out).sum() == 1                  # and NOTHING else is poisoned
    m = ~np.isnan(f)
    assert np.allclose(out[m], f[m], atol=1e-10)     # every finite cell is bit-preserved


def test_nan_source_cell_is_skipped_not_propagated_when_coarsening():
    slat, slon = SRC
    rng = np.random.default_rng(8)
    f = rng.standard_normal((slat.size, slon.size))
    f[60, 100] = np.nan
    out = regrid_conservative(f, slat, slon, DST[0], DST[1])
    assert np.isfinite(out).all()                    # one missing fine cell must not blank a coarse cell


def test_target_with_no_finite_source_is_nan():
    lat, lon = DST
    f = np.full((lat.size, lon.size), np.nan)
    out = regrid_conservative(f, lat, lon, lat, lon)
    assert np.isnan(out).all()


def test_nan_free_fast_path_unchanged():
    # the finite fast path must stay bit-identical to the pre-fix behaviour
    slat, slon = SRC
    rng = np.random.default_rng(9)
    f = rng.standard_normal((slat.size, slon.size))
    a = regrid_conservative(f, slat, slon, DST[0], DST[1])
    b = regrid_conservative(f, slat, slon, DST[0], DST[1], block=1)   # force blocking
    assert np.allclose(a, b, atol=1e-12)
