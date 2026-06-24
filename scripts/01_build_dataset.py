"""Stage 2 entrypoint: split -> climatology -> anomalies -> weekly means (task #3).

Cardinal rule: fit climatology + normalizer on TRAIN ONLY, then apply to val/test.

Caches data/processed/daily_anom.zarr (all years, tagged with a `split` coord:
train/val/test/embargo) and data/processed/climatology.zarr (train-only seasonal
cycle, needed to reconstruct full fields and to score the climatology baseline).
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import xarray as xr
from omegaconf import DictConfig

from s2s.data.climatology import fit_climatology, fit_normalizer, to_anomaly
from s2s.data.download import pull_era5
from s2s.data.splits import split_by_year


def _tag_split(ds: xr.Dataset, label: str) -> xr.Dataset:
    return ds.assign_coords(split=("time", np.full(ds.sizes["time"], label, dtype=object)))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    processed = Path(cfg.data.paths.processed)
    processed.mkdir(parents=True, exist_ok=True)

    print("Pulling raw ERA5 (Stage 1) and resampling to daily means...")
    raw_6h = pull_era5(cfg)
    daily = raw_6h.resample(time="1D").mean()
    # dev_years subsets are small (one dev window is tens to hundreds of MB) --
    # load eagerly so the chained groupby/rolling/concat in fit_climatology runs
    # on numpy instead of building a huge, slow dask task graph against the
    # network store. Full-record runs would need a chunked rewrite of this step.
    print("Loading daily-resampled subset into memory...")
    daily = daily.compute()
    print(f"  loaded {daily.sizes['time']} days, "
          f"~{sum(v.nbytes for v in daily.data_vars.values()) / 1e6:.1f} MB")

    print("Splitting by year (with embargo)...")
    splits = split_by_year(daily, cfg)
    for name, sub in splits.items():
        print(f"  {name:5s}: {sub.sizes['time']:5d} days  "
              f"[{str(sub.time.values.min())[:10]} -> {str(sub.time.values.max())[:10]}]"
              if sub.sizes["time"] else f"  {name:5s}: 0 days (empty for this dev_years window)")

    print("Fitting climatology on TRAIN years only...")
    clim = fit_climatology(splits["train"], cfg)

    print("Computing anomalies per split...")
    anom_parts = []
    for name, sub in splits.items():
        if sub.sizes["time"] == 0:
            continue
        anom = to_anomaly(sub, clim)
        anom_parts.append(_tag_split(anom, name))

    daily_anom = xr.concat(anom_parts, dim="time").sortby("time")

    normalizer = fit_normalizer(to_anomaly(splits["train"], clim), cfg)
    print("Normalizer stats (train-only):")
    for var, stats in normalizer.items():
        print(f"  {var:40s} mean={stats['mean']:.4f}  std={stats['std']:.4f}")

    clim_path = processed / "climatology.zarr"
    anom_path = processed / "daily_anom.zarr"
    print(f"Writing climatology -> {clim_path}")
    clim.to_zarr(clim_path, mode="w", zarr_format=2)
    print(f"Writing daily anomalies -> {anom_path}")
    daily_anom.drop_vars("split").assign_coords(
        split=("time", daily_anom.split.values.astype(str))
    ).to_zarr(anom_path, mode="w", zarr_format=2)

    # --- split census ---
    print("\nSplit census (days):")
    for name in ("train", "val", "test"):
        n = int((daily_anom.split == name).sum())
        print(f"  {name:5s}: {n}")

    # --- monsoon-season sanity check (JJAS daily precip total, India box) ---
    box = cfg.data.eval_box
    india = daily.sel(
        latitude=slice(min(box.lat_min, box.lat_max), max(box.lat_min, box.lat_max)),
        longitude=slice(box.lon_min, box.lon_max),
    )
    jjas = india["total_precipitation_24hr"].sel(time=india.time.dt.month.isin([6, 7, 8, 9]))
    jjas_mean = float(jjas.mean())
    print(f"\nMonsoon (JJAS) mean daily total_precipitation_24hr over India box: "
          f"{jjas_mean:.6f} m/day")

    print("\nStage 2 build complete.")


if __name__ == "__main__":
    main()
