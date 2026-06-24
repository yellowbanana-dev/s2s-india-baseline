"""Phase-A ensemble evaluation / gate smoke test (task #8).

Loads ONE trained checkpoint (eval.checkpoint), wraps it in a P2 ensemble via
mosaic-corner IC perturbation (ADR 0001), scores CRPS over the India box per
lead week against the climatology baseline, and reports the decision gate
(configs/eval/default.yaml: gate.lead_week, gate.metric) as PASS/FAIL.

THIS IS A SMOKE TEST OF THE EVAL PATH, NOT THE REAL GATE: the real gate needs a
full-record-trained ensemble of independently-trained members, not IC
perturbation around a single few-epoch dev-subset checkpoint. A FAIL here is
expected and acceptable for that reason.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import DictConfig

from s2s.data.datamodule import S2SDataModule
from s2s.eval.metrics import crps_ensemble
from s2s.models.ensemble import P2Ensemble
from s2s.models.lit import S2SLitModule


def _india_box(da: xr.DataArray, cfg) -> xr.DataArray:
    box = cfg.data.eval_box
    return da.sel(
        latitude=slice(min(box.lat_min, box.lat_max), max(box.lat_min, box.lat_max)),
        longitude=slice(box.lon_min, box.lon_max),
    )


def _latweighted_spatial_mean(da: xr.DataArray) -> float:
    w = np.cos(np.deg2rad(da["latitude"]))
    flat = da.mean(["time", "longitude"], skipna=True)
    return float((flat * w).sum("latitude") / w.sum())


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    ckpt_path = cfg.eval.checkpoint
    if not ckpt_path:
        raise ValueError("eval.checkpoint must be set, e.g. eval.checkpoint=/path/to/model.ckpt")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    dm = S2SDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    lead = len(cfg.data.lead_weeks)
    lit = S2SLitModule(
        in_channels=dm.in_channels,
        out_channels=dm.out_channels,
        lead=lead,
        latitude=dm.latitude,
        cfg=cfg,
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lit.load_state_dict(checkpoint["state_dict"])
    lit.eval()

    ensemble = P2Ensemble([lit], cfg)
    n_members = len(ensemble.seeds)

    test_ds = dm.test_dataset
    x = test_ds.inputs  # (N, C_in, lat, lon)
    y_true = test_ds.targets.numpy()  # (N, lead, C_out, lat, lon)
    n_samples = x.shape[0]
    print(f"Evaluating ensemble: n test samples={n_samples}  ensemble members={n_members}")

    preds = ensemble.forecast(x).numpy()  # (M, N, lead, C_out, lat, lon)

    lats, lons = dm.latitude, dm.lon
    lead_weeks = list(cfg.data.lead_weeks)
    out_vars = dm.target_vars

    rows = []
    for ci, var in enumerate(out_vars):
        for li, lead_week in enumerate(lead_weeks):
            truth = y_true[:, li, ci]  # (N, lat, lon)
            members = preds[:, :, li, ci]  # (M, N, lat, lon)
            zero = np.zeros_like(truth)

            truth_da = xr.DataArray(
                truth,
                dims=("time", "latitude", "longitude"),
                coords={"latitude": lats, "longitude": lons},
            )
            crps_model_field = truth_da.copy(data=crps_ensemble(members, truth))
            crps_clim_field = truth_da.copy(data=crps_ensemble(zero[np.newaxis, ...], truth))

            crps_model_box = _latweighted_spatial_mean(_india_box(crps_model_field, cfg))
            crps_clim_box = _latweighted_spatial_mean(_india_box(crps_clim_field, cfg))
            skill_pct = 100.0 * (1.0 - crps_model_box / crps_clim_box) if crps_clim_box else float("nan")

            rows.append(
                {
                    "variable": var,
                    "lead_week": lead_week,
                    "crps_model": crps_model_box,
                    "crps_climatology": crps_clim_box,
                    "skill_pct": skill_pct,
                    "pass": crps_model_box < crps_clim_box,
                }
            )

    table = pd.DataFrame(rows)

    gate_leads = list(cfg.eval.gate.lead_week)
    gate_rows = table[table["lead_week"].isin(gate_leads)]
    gate_passed = bool(gate_rows["pass"].all()) if len(gate_rows) else False

    print("\n=== Ensemble eval -- CRPS vs climatology (test split, India box) ===\n")
    for var in out_vars:
        sub = table[table["variable"] == var].drop(columns="variable").set_index("lead_week")
        print(f"--- {var} ---")
        print(sub.to_string(float_format=lambda v: f"{v:.5f}"))
        print()

    print(
        f"Decision gate (lead weeks {gate_leads}, metric={cfg.eval.gate.metric}): "
        f"{'PASS' if gate_passed else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
