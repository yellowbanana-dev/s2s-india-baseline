"""Weight-free HEALPix round-trip geometry (Phase-C lever (a) diagnostic).

Pure-geometry reference for the lon/lat <-> HEALPix interpolation loss. This
module is deliberately torch-free (numpy + healpy + sklearn only) so it can be
unit-tested cheaply and used as the GEOMETRIC FLOOR against which the model's
learned CrossAttentionInterpolate (trained and untrained) is compared.

The Mosaic adapter maps our lon/lat grid onto a HEALPix mesh and back via two
learned cross-attention interpolators. scripts/healpix_recon_check.py measures
that round-trip with UNTRAINED weights, which confounds mesh geometry with random
projections. The inverse-distance round-trip here isolates the geometry: it uses
the same haversine k-nearest-neighbour graph the model uses, but fixed 1/d^p
weights instead of learned attention. If even this weight-free floor is large over
the India box, the mesh is genuinely lossy there; if it is small, the ~0.307 in
the untrained check is mostly a measurement artifact, not headroom.
"""
from __future__ import annotations

import healpy as hp
import numpy as np
from sklearn.neighbors import BallTree


def build_lonlat_grid(resolution_deg: float = 5.625) -> tuple[np.ndarray, np.ndarray]:
    """Equiangular cell-centre lon/lat (deg) for a global grid, WB2 convention.

    5.625 deg -> lon (64,) in [0, 354.375], lat (32,) in [-87.1875, 87.1875].
    On the cluster the caller should prefer the datamodule's real coordinates;
    this is a data-free fallback so the geometry path runs without ERA5.
    """
    nlon = int(round(360.0 / resolution_deg))
    nlat = int(round(180.0 / resolution_deg))
    lon = np.arange(nlon) * resolution_deg
    lat = -90.0 + resolution_deg / 2.0 + np.arange(nlat) * resolution_deg
    return lon.astype(np.float64), lat.astype(np.float64)


def healpix_lonlat_deg(nside: int) -> np.ndarray:
    """HEALPix (NESTED) pixel centres as (npix, 2) array of [lon_deg, lat_deg].

    nside must be a power of two for NESTED ordering (matches the model's
    utils.get_healpix_grid, nest=True).
    """
    if nside < 1 or (nside & (nside - 1)) != 0:
        raise ValueError(f"nside must be a power of two for NESTED HEALPix, got {nside}")
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=True)
    lon = np.rad2deg(phi)
    lat = 90.0 - np.rad2deg(theta)
    return np.stack([lon, lat], axis=-1)


def knn_haversine(
    from_lonlat_deg: np.ndarray, to_lonlat_deg: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """k nearest neighbours (great-circle) of each `to` point among `from` points.

    Returns (dist_rad, idx). Distances are the haversine angle in radians. Mirrors
    the model's utils.get_neighbors coordinate handling (BallTree on [lat, lon]).
    """
    k = min(k, from_lonlat_deg.shape[0])
    from_latlon_rad = np.deg2rad(from_lonlat_deg[:, ::-1])
    to_latlon_rad = np.deg2rad(to_lonlat_deg[:, ::-1])
    tree = BallTree(from_latlon_rad, metric="haversine")
    dist, idx = tree.query(to_latlon_rad, k=k)
    return dist, idx


def _idw_interp(
    values_from: np.ndarray,
    from_lonlat_deg: np.ndarray,
    to_lonlat_deg: np.ndarray,
    k: int,
    power: float,
    eps: float = 1e-12,
) -> np.ndarray:
    dist, idx = knn_haversine(from_lonlat_deg, to_lonlat_deg, k)
    w = 1.0 / (dist ** power + eps)
    w /= w.sum(axis=1, keepdims=True)
    return (w * values_from[idx]).sum(axis=1)


def idw_round_trip(
    field_latlon: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    nside: int,
    k: int = 8,
    power: float = 2.0,
) -> np.ndarray:
    """lon/lat -> HEALPix -> lon/lat identity reconstruction via inverse-distance.

    `field_latlon` has shape (lat, lon); the return has the same shape.
    """
    lon_g, lat_g = np.meshgrid(lon_deg, lat_deg)  # (lat, lon)
    ll = np.stack([lon_g.ravel(), lat_g.ravel()], axis=-1)  # (Nll, 2) [lon, lat]
    hpll = healpix_lonlat_deg(nside)  # (Nhp, 2)
    f = field_latlon.ravel().astype(np.float64)
    on_hp = _idw_interp(f, ll, hpll, k, power)
    back = _idw_interp(on_hp, hpll, ll, k, power)
    return back.reshape(field_latlon.shape)


def latw_rmse(
    pred: np.ndarray,
    truth: np.ndarray,
    lat_deg: np.ndarray,
    lat_mask: np.ndarray | None = None,
    lon_mask: np.ndarray | None = None,
) -> float:
    """Latitude-weighted RMSE over (lat, lon), optionally restricted by masks."""
    w = np.cos(np.deg2rad(lat_deg)).reshape(-1, 1)  # (lat, 1)
    err2 = (pred - truth) ** 2
    if lat_mask is not None:
        err2 = err2[lat_mask, :]
        w = w[lat_mask, :]
    if lon_mask is not None:
        err2 = err2[:, lon_mask]
    return float(np.sqrt((err2 * w).sum() / (w.sum() * err2.shape[1])))


def india_box_masks(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    lat_mask = (lat_deg >= min(lat_min, lat_max)) & (lat_deg <= max(lat_min, lat_max))
    lon_mask = (lon_deg >= lon_min) & (lon_deg <= lon_max)
    return lat_mask, lon_mask


def sample_fields(lon_deg: np.ndarray, lat_deg: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """Controlled (lat, lon) fields: smooth planetary waves + white-noise worst case."""
    lon_r, lat_r = np.meshgrid(np.deg2rad(lon_deg), np.deg2rad(lat_deg))  # (lat, lon)
    rng = np.random.default_rng(seed)
    return {
        "planetary_wave_k3": (np.cos(3 * lon_r) * np.cos(lat_r)).astype(np.float64),
        "planetary_wave_k6": (np.cos(6 * lon_r) * np.cos(lat_r)).astype(np.float64),
        "white_noise": rng.normal(size=lon_r.shape).astype(np.float64),
    }
