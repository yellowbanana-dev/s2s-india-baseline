"""MAJ-3 diagnostic B: why does crps_clim_prob change under an identity common-grid regrid?

crps_clim_prob is built ONLY from train_weekly (daily_anom.zarr, the PROCESSED store).
Diagnostic A verified the RAW store grid (dm.latitude/dm.lon) matches equiangular_grid();
it never checked the PROCESSED store's grid or axis order. If those differ, regridding
train_weekly onto the constructed grid is a real remap, not a no-op.

No GPU, no checkpoint.
    PYTHONPATH="$PWD/src:$PYTHONPATH" python scripts/_diag_maj3b.py data=era5_india
"""
from pathlib import Path

import hydra
import numpy as np
import xarray as xr
from omegaconf import DictConfig

from s2s.data.windows import daily_to_weekly_mean
from s2s.eval.regrid import equiangular_grid, regrid_conservative_da


def _india_box(da, cfg):
    box = cfg.data.eval_box
    return da.sel(
        latitude=slice(min(box.lat_min, box.lat_max), max(box.lat_min, box.lat_max)),
        longitude=slice(box.lon_min, box.lon_max),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from s2s.data.datamodule import S2SDataModule
    dm = S2SDataModule(cfg); dm.prepare_data(); dm.setup()
    raw_lat = np.asarray(dm.latitude, dtype=np.float64)
    raw_lon = np.asarray(dm.lon, dtype=np.float64)
    d_lat, d_lon = equiangular_grid(5.625, with_poles=False)

    var = dm.target_vars[0]
    processed = Path(cfg.data.paths.processed)
    anom = xr.open_zarr(processed / "daily_anom.zarr")
    split = anom["split"].astype(str)
    train = anom.sel(time=split == "train").drop_vars("split")
    tw = daily_to_weekly_mean(train[[var]])[var].load()

    print("\n=== TRAIN_WEEKLY (processed daily_anom.zarr) ===")
    print(f"var        : {var}")
    print(f"dims       : {tw.dims}")
    print(f"shape      : {tw.shape}")
    print(f"value dtype: {tw.dtype}")
    p_lat = np.asarray(tw['latitude'].values, dtype=np.float64)
    p_lon = np.asarray(tw['longitude'].values, dtype=np.float64)
    print(f"lat dtype {tw['latitude'].dtype}  n={p_lat.size}  ascending={bool(p_lat[0] < p_lat[-1])}")
    print(f"lat[:3] {p_lat[:3]}  lat[-3:] {p_lat[-3:]}")
    print(f"lon dtype {tw['longitude'].dtype}  n={p_lon.size}  ascending={bool(p_lon[0] < p_lon[-1])}")
    print(f"lon[:3] {p_lon[:3]}  lon[-3:] {p_lon[-3:]}")
    print(f"non-dim coords: {[c for c in tw.coords if c not in tw.dims]}")

    print("\n=== GRID AGREEMENT (processed vs raw vs constructed) ===")
    for nm, a, b in (("processed vs raw   lat", p_lat, raw_lat),
                     ("processed vs raw   lon", p_lon, raw_lon),
                     ("processed vs ctor  lat", p_lat, d_lat),
                     ("processed vs ctor  lon", p_lon, d_lon)):
        if a.size != b.size:
            print(f"{nm}: SIZE MISMATCH {a.size} vs {b.size}   <-- REMAP, not identity")
        else:
            print(f"{nm}: MAX|diff| = {np.abs(a - b).max():.3e}")

    print("\n=== REGRID ROUND-TRIP ON train_weekly (should be a no-op) ===")
    tw_rg = regrid_conservative_da(tw, d_lat, d_lon)
    print(f"dims  before {tw.dims} -> after {tw_rg.dims}")
    print(f"shape before {tw.shape} -> after {tw_rg.shape}")
    ref = tw.transpose(*tw_rg.dims)
    if ref.shape == tw_rg.shape:
        diff = np.abs(np.asarray(ref.values, dtype=np.float64)
                      - np.asarray(tw_rg.values, dtype=np.float64))
        print(f"MAX|train_weekly diff| = {np.nanmax(diff):.6e}   <-- must be ~0")
        print(f"mean|diff|             = {np.nanmean(diff):.6e}")
    else:
        print("shape changed -> genuine remap")
    print(f"NaN frac before {float(np.isnan(tw.values).mean()):.4f} -> after {float(np.isnan(tw_rg.values).mean()):.4f}")

    print("\n=== INDIA BOX (drives crps_clim_prob) ===")
    b0, b1 = _india_box(tw, cfg), _india_box(tw_rg, cfg)
    print(f"native box shape {b0.shape} dims {b0.dims}")
    print(f"regrid box shape {b1.shape} dims {b1.dims}")
    print(f"native box lat {np.asarray(b0['latitude'].values)}")
    print(f"regrid box lat {np.asarray(b1['latitude'].values)}")
    print(f"native box lon {np.asarray(b0['longitude'].values)}")
    print(f"regrid box lon {np.asarray(b1['longitude'].values)}")
    print(f"native box std {float(b0.std()):.6f} | regrid box std {float(b1.std()):.6f}")
    print(f"native box mean {float(b0.mean()):.6f} | regrid box mean {float(b1.mean()):.6f}")
    print()


if __name__ == "__main__":
    main()
