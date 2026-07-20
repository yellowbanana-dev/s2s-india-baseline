"""MAJ-3 diagnostic: is the common-grid regrid an identity on the REAL store grid?

Throwaway verification script (no GPU, no checkpoint). Compares the datamodule's actual
latitude/longitude against the grid equiangular_grid() constructs, and measures how far
the conservative regrid is from an exact identity when source == target resolution.

Run with the 5.625 deg config:
    PYTHONPATH="$PWD/src:$PYTHONPATH" python scripts/_diag_maj3.py data=era5_india
"""
import hydra
import numpy as np
from omegaconf import DictConfig

from s2s.data.datamodule import S2SDataModule
from s2s.eval.regrid import (
    conservative_matrices,
    equiangular_grid,
    regrid_conservative,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    dm = S2SDataModule(cfg)
    dm.prepare_data()
    dm.setup()
    lat = np.asarray(dm.latitude, dtype=np.float64)
    lon = np.asarray(dm.lon, dtype=np.float64)
    d_lat, d_lon = equiangular_grid(5.625, with_poles=False)

    print("\n=== GRID COMPARISON ===")
    print(f"store nlat/nlon : {lat.size} x {lon.size}")
    print(f"ctor  nlat/nlon : {d_lat.size} x {d_lon.size}")
    print(f"store lat ascending: {bool(lat[0] < lat[-1])}")
    print(f"store lat[:3] {lat[:3]}   lat[-3:] {lat[-3:]}")
    print(f"ctor  lat[:3] {d_lat[:3]}   lat[-3:] {d_lat[-3:]}")
    print(f"store lon[:3] {lon[:3]}   lon[-3:] {lon[-3:]}")
    print(f"ctor  lon[:3] {d_lon[:3]}   lon[-3:] {d_lon[-3:]}")
    if lat.size == d_lat.size:
        print(f"MAX|lat diff| = {np.abs(lat - d_lat).max():.3e}")
    else:
        print("MAX|lat diff| = SIZE MISMATCH  <-- grids differ")
    if lon.size == d_lon.size:
        print(f"MAX|lon diff| = {np.abs(lon - d_lon).max():.3e}")
    else:
        print("MAX|lon diff| = SIZE MISMATCH  <-- grids differ")

    print("\n=== IDENTITY CHECK (store grid -> 5.625 deg ctor grid) ===")
    w_lat, w_lon = conservative_matrices(lat, lon, d_lat, d_lon)
    print(f"lat weight-matrix deviation from I : {np.abs(w_lat - np.eye(*w_lat.shape)).max():.3e}")
    print(f"lon weight-matrix deviation from I : {np.abs(w_lon - np.eye(*w_lon.shape)).max():.3e}")
    rng = np.random.default_rng(0)
    f = rng.standard_normal((lat.size, lon.size))
    out = regrid_conservative(f, lat, lon, d_lat, d_lon)
    if out.shape == f.shape:
        print(f"MAX|regrid(f) - f| = {np.abs(out - f).max():.3e}   <-- must be ~0 for identity")
    else:
        print(f"shape changed {f.shape} -> {out.shape}; not an identity by construction")

    print("\n=== INDIA-BOX CELL SELECTION (store vs ctor) ===")
    box = cfg.data.eval_box
    for name, la, lo in (("store", lat, lon), ("ctor", d_lat, d_lon)):
        sel_la = la[(la >= min(box.lat_min, box.lat_max)) & (la <= max(box.lat_min, box.lat_max))]
        sel_lo = lo[(lo >= box.lon_min) & (lo <= box.lon_max)]
        print(f"{name:5s}: {sel_la.size} lat x {sel_lo.size} lon  "
              f"lat[{sel_la.min() if sel_la.size else float('nan')}..{sel_la.max() if sel_la.size else float('nan')}] "
              f"lon[{sel_lo.min() if sel_lo.size else float('nan')}..{sel_lo.max() if sel_lo.size else float('nan')}]")
    print()


if __name__ == "__main__":
    main()
