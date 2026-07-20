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


def _apply(w_lat, w_lon, chunk):
    """NaN-aware separable conservative apply on a (b, lat, lon) float64 block.

    NaN-AWARENESS IS LOAD-BEARING, not defensive polish. A plain matmul regrid computes
    out[i] = sum_j w[i,j]*a[j]; since 0.0*NaN == NaN in IEEE-754, a single missing source
    cell poisons every target cell sharing its row/column -- even when w is the exact
    identity. The eval fields carry NaN (the scorer uses nanmean/skipna throughout), so an
    unguarded regrid turns the whole field NaN and every gate silently reports FAIL. We
    therefore average over the FINITE source cells only and renormalise by the weight mass
    that actually landed on them; a target cell with no finite source overlap is NaN.
    """
    finite = np.isfinite(chunk)
    if finite.all():                                        # fast path: exact, single pass
        tmp = np.einsum('ij,bjk->bik', w_lat, chunk, optimize=True)
        return np.einsum('bik,lk->bil', tmp, w_lon, optimize=True)
    vals = np.where(finite, chunk, 0.0)
    num = np.einsum('bik,lk->bil',
                    np.einsum('ij,bjk->bik', w_lat, vals, optimize=True), w_lon, optimize=True)
    den = np.einsum('bik,lk->bil',
                    np.einsum('ij,bjk->bik', w_lat, finite.astype(np.float64), optimize=True),
                    w_lon, optimize=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def regrid_conservative(field, src_lat, src_lon, dst_lat, dst_lon, block=None):
    """field (..., n_src_lat, n_src_lon) -> (..., n_dst_lat, n_dst_lon).

    NaN-aware (see _apply). Processed in blocks over the flattened leading dims so the
    float64 working copy stays bounded at high resolution (a full 1.5deg member ensemble
    would otherwise materialise ~11 GB in one go).
    """
    w_lat, w_lon = conservative_matrices(src_lat, src_lon, dst_lat, dst_lon)
    a = np.asarray(field)
    lead = a.shape[:-2]
    n_lat, n_lon = a.shape[-2], a.shape[-1]
    a2 = a.reshape(-1, n_lat, n_lon)
    nb = a2.shape[0]
    if block is None:
        block = max(1, int(64_000_000 // max(1, n_lat * n_lon)))
    out = np.empty((nb, w_lat.shape[0], w_lon.shape[0]), dtype=np.float64)
    for s in range(0, nb, block):
        out[s:s + block] = _apply(w_lat, w_lon, a2[s:s + block].astype(np.float64))
    return out.reshape(*lead, w_lat.shape[0], w_lon.shape[0]).astype(a.dtype, copy=False)


def regrid_conservative_da(da, dst_lat, dst_lon, lat_name="latitude", lon_name="longitude"):
    """Conservatively regrid an xarray DataArray, preserving dims, order, coords and name.

    DIM ORDER IS LOAD-BEARING. The processed store (daily_anom.zarr) is lon-major --
    (time, longitude, latitude) -- while the internal regrid works lat-major. Downstream
    scorers (crps_ensemble) consume `.values` POSITIONALLY, so returning a transposed array
    silently swaps the spatial axes: values, coords and even the India-box shape all still
    look correct (the box is 6x6), but the spatial correspondence is scrambled and
    crps_clim_prob inflates. This is what made both models fail the common-grid gate through
    an otherwise numerically-exact identity regrid. We therefore transpose back to the
    caller's original dim order before returning.
    """
    import xarray as xr
    src_lat = da[lat_name].values
    src_lon = da[lon_name].values
    other = [d for d in da.dims if d not in (lat_name, lon_name)]
    da_t = da.transpose(*other, lat_name, lon_name)
    out = regrid_conservative(da_t.values, src_lat, src_lon, dst_lat, dst_lon)
    coords = {k: da_t.coords[k] for k in other if k in da_t.coords}
    coords[lat_name] = np.asarray(dst_lat)
    coords[lon_name] = np.asarray(dst_lon)
    result = xr.DataArray(
        out, dims=(*other, lat_name, lon_name), coords=coords,
        name=da.name, attrs=dict(da.attrs),
    )
    return result.transpose(*da.dims)
