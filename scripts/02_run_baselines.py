"""Compute climatology + persistence baselines and score them (task #4 + #5).

Produces the numbers every model must beat, over the India box, weeks 1-6.
Reads data/processed/daily_anom.zarr (built by scripts/01_build_dataset.py) and
scores the TEST split, since that's what the decision gate (configs/eval/default.yaml)
is defined against.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig

from s2s.data.windows import build_lead_targets, daily_to_weekly_mean
from s2s.eval.baselines import climatology_forecast, persistence_forecast
from s2s.eval.metrics import acc, crps_ensemble, rmse


def _india_box(ds: xr.Dataset, cfg) -> xr.Dataset:
    box = cfg.data.eval_box
    return ds.sel(
        latitude=slice(min(box.lat_min, box.lat_max), max(box.lat_min, box.lat_max)),
        longitude=slice(box.lon_min, box.lon_max),
    )


def _latweighted_spatial_mean(da: xr.DataArray) -> float:
    """Weighted mean of a per-gridpoint field over (time, lon) then lat, weight=cos(lat)."""
    w = np.cos(np.deg2rad(da["latitude"]))
    flat = da.mean(["time", "longitude"], skipna=True)
    return float((flat * w).sum("latitude") / w.sum())


def _apply_lat_weight(da: xr.DataArray) -> xr.DataArray:
    """Scale by sqrt(cos(lat) / mean(cos(lat))) so flattened ACC/RMSE are lat-weighted."""
    w = np.cos(np.deg2rad(da["latitude"]))
    w = w / w.mean()
    return da * np.sqrt(w)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    processed = Path(cfg.data.paths.processed)
    anom_path = processed / "daily_anom.zarr"
    if not anom_path.exists():
        raise FileNotFoundError(f"{anom_path} not found -- run scripts/01_build_dataset.py first")

    daily_anom = xr.open_zarr(anom_path)
    test = daily_anom.sel(time=daily_anom.split == "test").drop_vars("split")
    print(f"Scoring TEST split: {test.sizes['time']} days "
          f"[{str(test.time.values.min())[:10]} -> {str(test.time.values.max())[:10]}]")

    weekly = daily_to_weekly_mean(test)
    lead_weeks = list(cfg.data.lead_weeks)
    targets = build_lead_targets(weekly, lead_weeks)

    weekly_box = _india_box(weekly, cfg)
    targets_box = _india_box(targets, cfg)

    target_vars = list(cfg.data.variables.targets.surface)
    rows = []

    for var in target_vars:
        for lead in lead_weeks:
            truth_da = targets_box[var].sel(lead=lead)
            persist_da = weekly_box[var]
            valid = truth_da.notnull() & persist_da.notnull()
            truth_da = truth_da.where(valid)
            persist_da = persist_da.where(valid)

            # --- CRPS: deterministic (single-member) forecasts, weighted spatial mean ---
            truth_np = truth_da.values
            zero_np = np.zeros_like(truth_np)
            persist_np = persist_da.values

            crps_clim_field = truth_da.copy(
                data=crps_ensemble(zero_np[np.newaxis, ...], truth_np)
            )
            crps_pers_field = truth_da.copy(
                data=crps_ensemble(persist_np[np.newaxis, ...], truth_np)
            )
            crps_clim = _latweighted_spatial_mean(crps_clim_field)
            crps_pers = _latweighted_spatial_mean(crps_pers_field)

            # --- ACC: persistence only (climatology forecast has zero variance -> undefined) ---
            truth_w = _apply_lat_weight(truth_da)
            persist_w = _apply_lat_weight(persist_da)
            acc_pers = acc(persist_w.values, truth_w.values)
            rmse_pers = rmse(persist_w.values, truth_w.values)
            rmse_clim = rmse(np.zeros_like(truth_w.values), truth_w.values)

            rows.append(
                {
                    "variable": var,
                    "lead_week": lead,
                    "crps_climatology": crps_clim,
                    "crps_persistence": crps_pers,
                    "rmse_climatology": rmse_clim,
                    "rmse_persistence": rmse_pers,
                    "acc_persistence": acc_pers,
                }
            )

    table = pd.DataFrame(rows)

    print("\n=== Baseline scores over India box (test split) ===\n")
    for var in target_vars:
        sub = table[table["variable"] == var].drop(columns="variable").set_index("lead_week")
        print(f"--- {var} ---")
        print(sub.to_string(float_format=lambda x: f"{x:.5f}"))
        print()

    out_path = processed / "baseline_scores.csv"
    table.to_csv(out_path, index=False)
    print(f"Saved score table -> {out_path}")


if __name__ == "__main__":
    main()
