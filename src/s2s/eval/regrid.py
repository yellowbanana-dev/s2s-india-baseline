"""Area-conservative regridding between equiangular lat-lon grids (MAJ-3, review 2026-07-14).

Cross-resolution CRPSS comparison (ADR-0007 f3) requires scoring the 1.5deg and 5.625deg
models on a COMMON grid: CRPS magnitude scales with grid resolution, so native-grid CRPSS is
not comparable across resolutions. We area-conservatively coarsen the finer forecast/truth/
climatology to the coarse grid before scoring.

Standard separable first-order-conservative scheme for equiangular lat-lon grids: cell area =
dlon * (sin(lat_hi) - sin(lat_lo)) separates into independent 1-D overlap-weight matrices
along latitude (weighted by sin of cell edges) and longitude (periodic length overlap). A
target cell's value is the area-weighted mean of the source cells it overlaps. This preserves
the global area-weighted mean (conservation) and maps a constant field to itself (partition of
unity). Cell edges are inferred from coordinate centers, so 'with poles' (points at +/-90,
half-width polar cells) and 'without poles' grids are both handled. Latitudes must be ASCENDING.
"""
from __future__ import annotations

import numpy as np


def equiangular_grid(resolution_deg, with_poles=False):
    """Return (lat, lon) centers for a global equiangular grid at resolution_deg."""
    d = float(resolution_deg)
    nlat = int(round(180.0 / d)) + (1 if with_poles else 0)
    nlon = int(round(360.0 / d))
    if with_poles:
        lat = np.linspace(-90.0, 90.0, nlat)
    else:
        lat = -90.0 + d / 2.0 + d * np.arange(nlat)
    lon = d * np.arange(nlon)
    return lat, lon


def _centers_to_edges_lat(centers):
    c = np.asarray(centers, dtype=np.float64)
    if c.size > 1 and c[0] > c[-1]:
        raise ValueError("latitude must be ascending (south->north)")
    mid = 0.5 * (c[:-1] + c[1:])
    first = c[0] - (mid[0] - c[0]) if c.size > 1 else c[0] - 0.5
    last = c[-1] + (c[-1] - mid[-1]) if c.size > 1 else c[-1] + 0.5
    edges = np.concatenate([[first], mid, [last]])
    return np.clip(edges, -90.0, 90.0)


def _centers_to_edges_lon(centers):
    c = np.asarray(centers, dtype=np.float64)
    d = float(np.median(np.diff(c))) if c.size > 1 else 360.0
    edges = np.empty(c.size + 1)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - d / 2.0
    edges[-1] = c[-1] + d / 2.0
    return edges


def latitude_area_weights(lat):
    """Per-cell area measure along latitude: sin(edge_hi) - sin(edge_lo)."""
    s = np.sin(np.deg2rad(_centers_to_edges_lat(lat)))
    return s[1:] - s[:-1]


def _overlap_lat(src_edges, dst_edges):
    s_src = np.sin(np.deg2rad(src_edges))
    s_dst = np.sin(np.deg2rad(dst_edges))
    lo = np.maximum(s_dst[:-1][:, None], s_src[:-1][None, :])
    hi = np.minimum(s_dst[1:][:, None], s_src[1:][None, :])
    return np.clip(hi - lo, 0.0, None)


def _overlap_lon(src_edges, dst_edges, period=360.0):
    src_lo, src_hi = src_edges[:-1], src_edges[1:]
    dst_lo, dst_hi = dst_edges[:-1], dst_edges[1:]
    w = np.zeros((dst_lo.size, src_lo.size))
    for shift in (-period, 0.0, period):
        lo = np.maximum(dst_lo[:, None], src_lo[None, :] + shift)
        hi = np.minimum(dst_hi[:, None], src_hi[None, :] + shift)
        w += np.clip(hi - lo, 0.0, None)
    return w


def _row_normalize(w):
    s = w.sum(axis=1, keepdims=True)
    if np.any(s <= 0):
        raise ValueError("target cell with no source overlap; grids incompatible")
    return w / s


def conservative_matrices(src_lat, src_lon, dst_lat, dst_lon):
    w_lat = _row_normalize(_overlap_lat(_centers_to_edges_lat(src_lat),
                                        _centers_to_edges_lat(dst_lat)))
    w_lon = _row_normalize(_overlap_lon(_centers_to_edges_lon(src_lon),
                                        _centers_to_edges_lon(dst_lon)))
    return w_lat, w_lon


def regrid_conservative(field, src_lat, src_lon, dst_lat, dst_lon):
    """field (..., n_src_lat, n_src_lon) -> (..., n_dst_lat, n_dst_lon)."""
    w_lat, w_lon = conservative_matrices(src_lat, src_lon, dst_lat, dst_lon)
    a = np.asarray(field, dtype=np.float64)
    lead = a.shape[:-2]
    a2 = a.reshape(-1, a.shape[-2], a.shape[-1])
    tmp = np.einsum('ij,bjk->bik', w_lat, a2, optimize=True)
    out = np.einsum('bik,lk->bil', tmp, w_lon, optimize=True)
    out = out.reshape(*lead, w_lat.shape[0], w_lon.shape[0])
    return out.astype(np.asarray(field).dtype, copy=False)


def regrid_conservative_da(da, dst_lat, dst_lon, lat_name="latitude", lon_name="longitude"):
    """Conservatively regrid an xarray DataArray, preserving all non-spatial dims/coords."""
    import xarray as xr
    src_lat = da[lat_name].values
    src_lon = da[lon_name].values
    other = [d for d in da.dims if d not in (lat_name, lon_name)]
    da_t = da.transpose(*other, lat_name, lon_name)
    out = regrid_conservative(da_t.values, src_lat, src_lon, dst_lat, dst_lon)
    coords = {k: da_t.coords[k] for k in other if k in da_t.coords}
    coords[lat_name] = np.asarray(dst_lat)
    coords[lon_name] = np.asarray(dst_lon)
    return xr.DataArray(out, dims=(*other, lat_name, lon_name), coords=coords)
