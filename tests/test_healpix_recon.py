"""Phase-C lever (a): weight-free HEALPix round-trip geometry guardrails.

Torch-free (numpy + healpy + sklearn). Pins the geometric-floor invariants the
decomposition diagnostic (scripts/healpix_recon_decomp.py) relies on.
"""
import numpy as np
import pytest

from s2s.eval.healpix_recon import (
    build_lonlat_grid,
    healpix_lonlat_deg,
    idw_round_trip,
    india_box_masks,
    latw_rmse,
    sample_fields,
)

LON, LAT = build_lonlat_grid(5.625)


def test_grid_shape_wb2_convention():
    assert LON.shape == (64,) and LAT.shape == (32,)
    assert LON.min() == 0.0 and abs(LON.max() - 354.375) < 1e-9
    assert abs(LAT.min() + 87.1875) < 1e-9 and abs(LAT.max() - 87.1875) < 1e-9


def test_healpix_npix_and_nested_power_of_two():
    assert healpix_lonlat_deg(16).shape == (12 * 16 * 16, 2)
    with pytest.raises(ValueError):
        healpix_lonlat_deg(24)  # not a power of two -> invalid NESTED


def test_constant_field_roundtrip_is_lossless():
    const = np.full((32, 64), 2.71828, dtype=np.float64)
    recon = idw_round_trip(const, LON, LAT, nside=16)
    assert np.abs(recon - const).max() < 1e-9


def test_smooth_field_floor_is_small_over_india_box():
    latm, lonm = india_box_masks(LAT, LON, 5.0, 40.0, 65.0, 100.0)
    f = sample_fields(LON, LAT)["planetary_wave_k3"]
    rng = float(f.max() - f.min())
    recon = idw_round_trip(f, LON, LAT, nside=16)
    india_rel = latw_rmse(recon, f, LAT, latm, lonm) / rng
    # Weight-free floor for a smooth field must be far below the untrained
    # cross-attention 0.307 -> the geometry is not the bottleneck.
    assert india_rel < 0.1


def test_finer_mesh_does_not_worsen_floor():
    f = sample_fields(LON, LAT)["white_noise"]
    g16 = latw_rmse(idw_round_trip(f, LON, LAT, 16), f, LAT)
    g32 = latw_rmse(idw_round_trip(f, LON, LAT, 32), f, LAT)
    assert g32 <= g16 + 1e-9


def test_smooth_interpolates_better_than_noise():
    fields = sample_fields(LON, LAT)
    smooth = fields["planetary_wave_k3"]
    noise = fields["white_noise"]
    s = latw_rmse(idw_round_trip(smooth, LON, LAT, 16), smooth, LAT) / (smooth.max() - smooth.min())
    n = latw_rmse(idw_round_trip(noise, LON, LAT, 16), noise, LAT) / (noise.max() - noise.min())
    assert s < n
