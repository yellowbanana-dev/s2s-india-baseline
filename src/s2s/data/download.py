"""Stage 1 - Pull ERA5 from WeatherBench2 (task #2).

GLOBAL input domain at 5.625 deg (64x32, no poles). Opens the public WB2 Zarr
store lazily over the network (anonymous access) and selects the variables/levels
from the data config. No local download is required for 5.625 deg.

HUMAN-OWNED CHECKS (verify_pull): units, continuous 6h time axis, grid shape,
physical value ranges. Precip is an ACCUMULATION; SST is NaN over land (expected).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

# Fallback if cfg.data.zarr_path is missing.
_DEFAULT_STORE = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
)


def pull_era5(cfg) -> xr.Dataset:
    """Open the WB2 ERA5 Zarr store and select target + predictor variables.

    Leveled predictors are selected per pressure level and flattened to
    `<var>_<level>` (e.g. geopotential_500, u_component_of_wind_850). Returns a
    lazily-loaded Dataset with dims (time, latitude, longitude). Stays lazy.
    """
    store = getattr(cfg.data, "zarr_path", None) or _DEFAULT_STORE
    ds = xr.open_zarr(store, storage_options={"token": "anon"}, chunks={"time": 100})

    v = cfg.data.variables
    surface = list(v.targets.surface) + list(v.predictors.surface)

    missing = [s for s in surface if s not in ds.data_vars]
    if missing:
        raise KeyError(
            f"surface variables not in store: {missing}. "
            f"First available: {list(ds.data_vars)[:25]}"
        )

    out = {name: ds[name] for name in surface}

    levels_cfg = getattr(v.predictors, "levels", {}) or {}
    for var, levels in levels_cfg.items():
        if var not in ds.data_vars:
            raise KeyError(f"leveled variable {var!r} not in store")
        for lev in levels:
            out[f"{var}_{lev}"] = ds[var].sel(level=lev).drop_vars("level")

    result = xr.Dataset(out)

    dev_years = getattr(cfg.data, "dev_years", None)
    if dev_years:
        lo, hi = dev_years
        result = result.sel(time=slice(f"{lo}", f"{hi}"))

    return result


def verify_pull(ds: xr.Dataset, cfg) -> None:
    """Loud sanity checks. Raises on anything that would silently corrupt training.

    This is the human-owned gate: read the printed summary and the plotted field
    before trusting the data.
    """
    # --- grid shape: 64 lon x 32 lat, poles excluded ---
    assert ds.sizes.get("longitude") == 64, f"longitude != 64: {ds.sizes}"
    assert ds.sizes.get("latitude") == 32, f"latitude != 32: {ds.sizes}"
    assert abs(float(ds.latitude.max())) < 90, "grid should NOT include the poles"

    # --- continuous 6-hourly time axis, no gaps ---
    dt_h = np.diff(ds.time.values).astype("timedelta64[h]").astype(int)
    uniq = np.unique(dt_h)
    assert set(uniq.tolist()) <= {6}, f"non-6h gaps in time axis: {uniq}"

    # --- physical ranges (catches unit mistakes) ---
    t2m = ds["2m_temperature"]
    tmin, tmax = float(t2m.min()), float(t2m.max())
    assert 180 < tmin and tmax < 340, f"2m_temperature not in Kelvin? [{tmin:.1f},{tmax:.1f}]"

    tp = ds["total_precipitation_24hr"]
    assert float(tp.min()) >= -1e-6, "precip accumulation should be non-negative"

    print("verify_pull OK")
    print(f"  vars      : {list(ds.data_vars)}")
    print(f"  grid      : {ds.sizes.get('latitude')} lat x {ds.sizes.get('longitude')} lon")
    print(f"  time span : {str(ds.time.values[0])[:10]} -> {str(ds.time.values[-1])[:10]}")
    print(f"  t2m range : [{tmin:.1f}, {tmax:.1f}] K")
