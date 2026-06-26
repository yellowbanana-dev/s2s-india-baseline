"""Phase-B honest evaluation (eval upgrade) -- extends the Phase-A smoke test.

Loads ONE trained checkpoint (eval.checkpoint), wraps it in the P2 IC-perturbation
ensemble (ADR-0001), converts model output + truth back to PHYSICAL units, and
scores over the India box on the TEST split:

  * CRPSS vs the PROBABILISTIC climatology -- the honest Phase-B bar. The reference
    is a week-of-year-windowed (+/- eval.clim_woy_window) pool of TRAIN weekly
    anomalies (s2s.eval.baselines.climatology_woy_ensemble), so it is local in
    season. Train-only => no leakage.
  * raw CRPS for the model + the deterministic (zero-anomaly) climatology, kept for
    continuity with Phase A.
  * deterministic ACC and RMSE of the ensemble MEAN.
  * rank histogram (Talagrand) + spread-error ratio -- ensemble calibration
    (detects the Phase-A under-dispersion).
  * reliability diagrams for P(t2m anom>0), P(precip anom>0), and the India-context
    absolute WEEKLY-MEAN events P(t2m>40 C) and P(precip>50 mm/day) -- the absolute
    ones scored on RECONSTRUCTED weekly-mean physical fields (train clim added back;
    precip un-log1p'd). These are weekly-mean exceedances, NOT daily extremes.

Outputs (eval.results_dir): metrics.csv, rank_hist_<var>_wk<k>.png,
reliability_<event>.png. Decision gate: CRPSS > eval.gate.threshold at the gate
lead weeks, both target variables, vs eval.gate.reference.

STILL A SMOKE TEST OF THE EVAL PATH when run on a single IC-perturbation checkpoint:
the real gate needs an ensemble of independently-trained members. A CRPSS<=0 here
(under-dispersed) is the expected, honest Phase-A baseline we want on record.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import DictConfig

from s2s.data.datamodule import S2SDataModule
from s2s.data.windows import daily_to_weekly_mean
from s2s.eval.baselines import climatology_woy_ensemble
from s2s.eval.metrics import (
    acc,
    crps_ensemble,
    crpss,
    event_probability,
    rank_histogram,
    reliability_curve,
    rmse,
    spread_error_ratio,
)
from s2s.models.ensemble import P2Ensemble
from s2s.models.lit import S2SLitModule


def _india_box(da: xr.DataArray, cfg) -> xr.DataArray:
    box = cfg.data.eval_box
    return da.sel(
        latitude=slice(min(box.lat_min, box.lat_max), max(box.lat_min, box.lat_max)),
        longitude=slice(box.lon_min, box.lon_max),
    )


def _latw(da: xr.DataArray) -> np.ndarray:
    """sqrt(cos(lat)) weights broadcast to da's (..., lat, lon) shape, for acc/rmse."""
    w = np.sqrt(np.cos(np.deg2rad(da["latitude"].values)))
    shape = [1] * da.ndim
    shape[list(da.dims).index("latitude")] = w.size
    return w.reshape(shape)


def _latweighted_spatial_mean(da: xr.DataArray) -> float:
    w = np.cos(np.deg2rad(da["latitude"]))
    flat = da.mean(["time", "longitude"], skipna=True)
    return float((flat * w).sum("latitude") / w.sum())


def _train_weekly_anom(cfg, var):
    """TRAIN weekly-mean anomalies (physical-unit anomalies) for one var, with time."""
    processed = Path(cfg.data.paths.processed)
    anom = xr.open_zarr(processed / "daily_anom.zarr")
    split = anom["split"].astype(str)
    train = anom.sel(time=split == "train").drop_vars("split")
    return daily_to_weekly_mean(train[[var]])[var].load()


def _clim_doy(cfg, var):
    """Train day-of-year climatology for one var (log1p space for precip)."""
    processed = Path(cfg.data.paths.processed)
    clim = xr.open_zarr(processed / "climatology.zarr")
    return clim[var].load()


def _reconstruct_physical(anom_phys, clim_doy, var, target_times):
    """anomaly(physical) + train clim(day-of-year) -> weekly-mean physical field.

    anom_phys: (..., n_samples, lat, lon) with sample as the SECOND-LAST-but-one
    axis is awkward; here we accept (n, lat, lon) or (m, n, lat, lon) and broadcast
    the per-sample climatology over the leading member axis if present. Precip is
    un-log1p'd after adding the climatology back (clim lives in log1p space).
    """
    doy = pd.DatetimeIndex(target_times).dayofyear.values
    clim_sel = clim_doy.sel(dayofyear=xr.DataArray(doy, dims="sample")).values  # (n, lat, lon)
    if anom_phys.ndim == 4:        # (m, n, lat, lon)
        phys = anom_phys + clim_sel[np.newaxis, ...]
    else:                          # (n, lat, lon)
        phys = anom_phys + clim_sel
    if var.startswith("total_precipitation"):
        phys = np.expm1(phys)
    return phys


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    ckpt_path = cfg.eval.checkpoint
    if not ckpt_path:
        raise ValueError("eval.checkpoint must be set, e.g. eval.checkpoint=/path/to/model.ckpt")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    results_dir = Path(cfg.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

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
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    lit.load_state_dict(checkpoint["state_dict"])
    lit.eval()
    lit.to(device)

    ensemble = P2Ensemble([lit], cfg)
    n_members = len(ensemble.seeds)

    test_ds = dm.test_dataset
    x = test_ds.inputs.to(device)
    y_true = test_ds.targets.numpy()           # (N, lead, C, lat, lon) standardized
    n_samples = x.shape[0]
    print(f"Evaluating ensemble: n test samples={n_samples}  members={n_members}  device={device}")

    with torch.no_grad():
        preds = ensemble.forecast(x).detach().cpu().numpy()  # (M, N, lead, C, lat, lon) standardized

    lats, lons = dm.latitude, dm.lon
    lead_weeks = list(cfg.data.lead_weeks)
    out_vars = dm.target_vars
    woy_window = int(cfg.eval.clim_woy_window)

    target_mean = np.array([dm.normalizer[v]["mean"] for v in out_vars], dtype=np.float32)
    target_std = np.array([dm.normalizer[v]["std"] for v in out_vars], dtype=np.float32)

    def _to_physical(arr, channel_axis):
        shape = [1] * arr.ndim
        shape[channel_axis] = len(out_vars)
        return arr * target_std.reshape(shape) + target_mean.reshape(shape)

    y_true = _to_physical(y_true, channel_axis=2)
    preds = _to_physical(preds, channel_axis=3)

    # --- reconstruct the TEST weekly time axis so each (sample, lead) has a date,
    # for week-of-year pooling + absolute-event physical reconstruction. ---
    processed = Path(cfg.data.paths.processed)
    anom_all = xr.open_zarr(processed / "daily_anom.zarr")
    split = anom_all["split"].astype(str)
    test_daily = anom_all.sel(time=split == "test").drop_vars("split")
    test_weekly_time = daily_to_weekly_mean(test_daily[[out_vars[0]]]).time.values
    # assemble_arrays drops edge weeks lacking a full lead window; align by trimming
    # to the last N init weeks that produced samples (max_lead trimmed off the end).
    max_lead = max(lead_weeks)
    init_times = test_weekly_time[: len(test_weekly_time) - max_lead]
    if len(init_times) != n_samples:
        # be robust to any off-by-one in edge handling: take the trailing n_samples.
        init_times = init_times[-n_samples:]

    rows = []
    rank_store = {}
    reliab = {ev["name"]: {"p": [], "y": []} for ev in cfg.eval.reliability.events}

    for ci, var in enumerate(out_vars):
        clim_doy = _clim_doy(cfg, var)
        train_weekly = _train_weekly_anom(cfg, var)  # physical-unit anomalies + time

        for li, lead_week in enumerate(lead_weeks):
            truth = y_true[:, li, ci]            # (N, lat, lon) physical anomaly
            members = preds[:, :, li, ci]        # (M, N, lat, lon) physical anomaly
            zero = np.zeros_like(truth)

            truth_da = xr.DataArray(
                truth, dims=("time", "latitude", "longitude"),
                coords={"latitude": lats, "longitude": lons},
            )

            # ---- CRPS: model + deterministic clim ----
            crps_model_f = truth_da.copy(data=crps_ensemble(members, truth))
            crps_detclim_f = truth_da.copy(data=crps_ensemble(zero[np.newaxis, ...], truth))
            crps_model = _latweighted_spatial_mean(_india_box(crps_model_f, cfg))
            crps_detclim = _latweighted_spatial_mean(_india_box(crps_detclim_f, cfg))

            # ---- CRPS: probabilistic clim (woy-windowed train pool), per sample ----
            target_times = pd.DatetimeIndex(init_times) + pd.to_timedelta(7 * lead_week, unit="D")
            woy = target_times.isocalendar().week.values.astype(int)
            crps_prob_samples = np.empty(n_samples)
            train_box = _india_box(train_weekly, cfg)
            for s in range(n_samples):
                clim_ens = climatology_woy_ensemble(train_box, int(woy[s]), window=woy_window)
                t_box = _india_box(truth_da.isel(time=s), cfg)
                cf = crps_ensemble(clim_ens.values, t_box.values)  # (lat, lon)
                w = np.cos(np.deg2rad(t_box["latitude"].values))
                crps_prob_samples[s] = float(
                    np.average(np.nanmean(cf, axis=1), weights=w)
                )
            crps_probclim = float(np.mean(crps_prob_samples))

            skill_det = crpss(crps_model, crps_detclim)
            skill_prob = crpss(crps_model, crps_probclim)

            # ---- deterministic ACC / RMSE of the ensemble mean (lat-weighted) ----
            ens_mean = members.mean(axis=0)      # (N, lat, lon)
            em_box = _india_box(truth_da.copy(data=ens_mean), cfg)
            tr_box = _india_box(truth_da, cfg)
            w = _latw(em_box)
            acc_box = acc((em_box.values * w), (tr_box.values * w))
            rmse_box = rmse((em_box.values * w), (tr_box.values * w))

            # ---- rank histogram + spread-error (box) ----
            mem_box = _india_box(
                xr.DataArray(
                    members, dims=("member", "time", "latitude", "longitude"),
                    coords={"latitude": lats, "longitude": lons},
                ), cfg,
            ).values
            tr_box_v = tr_box.values
            rank_store[(var, lead_week)] = rank_histogram(mem_box, tr_box_v)
            ser = spread_error_ratio(mem_box, tr_box_v)

            # ---- reliability events ----
            for ev in cfg.eval.reliability.events:
                if ev["variable"] != var:
                    continue
                thr = float(ev["threshold"])
                if ev["space"] == "anomaly":
                    mem_ev, truth_ev = mem_box, tr_box_v
                else:
                    mem_ev = _india_box(
                        xr.DataArray(
                            _reconstruct_physical(members, clim_doy, var, init_times if lead_week == 0 else
                                                  (pd.DatetimeIndex(init_times) + pd.to_timedelta(7 * lead_week, unit="D"))),
                            dims=("member", "time", "latitude", "longitude"),
                            coords={"latitude": lats, "longitude": lons},
                        ), cfg,
                    ).values
                    truth_ev = _india_box(
                        xr.DataArray(
                            _reconstruct_physical(truth, clim_doy, var,
                                                  pd.DatetimeIndex(init_times) + pd.to_timedelta(7 * lead_week, unit="D")),
                            dims=("time", "latitude", "longitude"),
                            coords={"latitude": lats, "longitude": lons},
                        ), cfg,
                    ).values
                pf = event_probability(mem_ev, thr, ev["comparison"])
                yt = (truth_ev > thr).astype(float)
                reliab[ev["name"]]["p"].append(pf.reshape(-1))
                reliab[ev["name"]]["y"].append(yt.reshape(-1))

            rows.append({
                "variable": var, "lead_week": lead_week,
                "crps_model": crps_model,
                "crps_clim_det": crps_detclim,
                "crps_clim_prob": crps_probclim,
                "crpss_vs_det": skill_det,
                "crpss_vs_prob": skill_prob,
                "acc_mean": acc_box,
                "rmse_mean": rmse_box,
                "spread_error_ratio": ser,
            })

    table = pd.DataFrame(rows)
    csv_path = results_dir / "metrics.csv"
    table.to_csv(csv_path, index=False)

    # ---- rank-hist plots ----
    for (var, k), counts in rank_store.items():
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(np.arange(len(counts)), counts / counts.sum(), width=0.9)
        ax.axhline(1.0 / len(counts), color="k", ls="--", lw=1, label="calibrated")
        ax.set_title(f"Rank histogram -- {var} wk{k}")
        ax.set_xlabel("rank"); ax.set_ylabel("frequency"); ax.legend()
        fig.tight_layout(); fig.savefig(results_dir / f"rank_hist_{var}_wk{k}.png", dpi=120)
        plt.close(fig)

    # ---- reliability plots ----
    for name, store in reliab.items():
        if not store["p"]:
            continue
        p = np.concatenate(store["p"]); y = np.concatenate(store["y"])
        centers, obs, counts = reliability_curve(p, y, n_bins=int(cfg.eval.reliability.n_bins))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(centers, obs, "o-", label="model")
        ax.set_title(f"Reliability -- {name}")
        ax.set_xlabel("forecast probability"); ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
        fig.tight_layout(); fig.savefig(results_dir / f"reliability_{name}.png", dpi=120)
        plt.close(fig)

    print("\n=== Honest eval -- India box, test split, PHYSICAL units ===\n")
    for var in out_vars:
        sub = table[table["variable"] == var].drop(columns="variable").set_index("lead_week")
        print(f"--- {var} ---")
        print(sub.to_string(float_format=lambda v: f"{v:.5f}"))
        print()

    gate_leads = list(cfg.eval.gate.lead_week)
    gate_thr = float(cfg.eval.gate.get("threshold", 0.0))
    g = table[table["lead_week"].isin(gate_leads)]
    gate_pass = bool((g["crpss_vs_prob"] > gate_thr).all()) if len(g) else False
    print(
        f"Decision gate (lead weeks {gate_leads}, metric={cfg.eval.gate.metric} "
        f"vs {cfg.eval.gate.reference}, threshold {gate_thr}): "
        f"{'PASS' if gate_pass else 'FAIL'}"
    )
    print(f"\nSaved: {csv_path} + rank-hist/reliability PNGs in {results_dir}")


if __name__ == "__main__":
    main()
