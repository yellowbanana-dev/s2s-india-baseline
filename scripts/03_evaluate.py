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
    precip un-log1p'd to mm/day; see Jensen-bias note in _reconstruct_physical).
    These are weekly-mean exceedances, NOT daily extremes.

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
from s2s.eval.baselines import (
    climatology_woy_ensemble,
    climatology_woy_polytrend_ensemble,
    climatology_woy_trend_ensemble,
    fit_linear_trend,
    fit_poly_trend,
)
from s2s.eval.bootstrap import (
    block_bootstrap_crpss,
    crpss_by_year,
    year_bootstrap_crpss,
)
from s2s.eval.time_align import reconstruct_init_times
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
from s2s.eval.provenance import provenance_columns
from s2s.models.mosaic.primitives import NoiseGenerator


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


def _latweighted_per_sample(da: xr.DataArray) -> np.ndarray:
    """Lat-weighted spatial mean per time step -> (n_time,). Same recipe as the
    probabilistic-clim per-sample reference: nanmean over longitude, then
    cos(lat)-weighted average over latitude. mean() of this equals
    _latweighted_spatial_mean(da) up to NaN handling."""
    arr = da.transpose("time", "latitude", "longitude").values  # (N, lat, lon)
    w = np.cos(np.deg2rad(da["latitude"].values))
    per_lat = np.nanmean(arr, axis=2)                           # (N, lat)
    return np.average(per_lat, axis=1, weights=w)               # (N,)


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
    un-log1p'd after adding the climatology back (clim lives in log1p space); the
    result is mm/day (Fix 4/M2).

    JENSEN BIAS (Fix 4/M2): this reconstruction is expm1(weekly_mean(log1p(daily))),
    but the physical weekly mean is weekly_mean(expm1(log1p(daily))). Since expm1 is
    convex, expm1(mean(.)) <= mean(expm1(.)), so reconstructed weekly-mean precip is
    biased LOW. It affects ONLY the physical-threshold reliability diagrams (a
    secondary diagnostic); the headline CRPS/CRPSS are scored in transformed-anomaly
    space and are unaffected. A future fix reconstructs in daily space then averages.
    """
    doy = pd.DatetimeIndex(target_times).dayofyear.values
    # climatology.zarr stores (dayofyear, longitude, latitude); transpose to
    # (sample, latitude, longitude) to match assemble_arrays' lat-first layout.
    clim_sel = (
        clim_doy.sel(dayofyear=xr.DataArray(doy, dims="sample"))
        .transpose("sample", "latitude", "longitude")
        .values
    )  # (n, lat, lon)
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
        longitude=dm.lon,  # REQUIRED for model=mosaic: HEALPix interp grid must match
        cfg=cfg,           # the data's true longitudes (dm.lon), not lit.py's fallback.
    )
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    lit.load_state_dict(checkpoint["state_dict"])
    lit.eval()
    lit.to(device)

    test_ds = dm.test_dataset
    x = test_ds.inputs.to(device)
    y_true = test_ds.targets.numpy()           # (N, lead, C, lat, lon) standardized
    n_samples = x.shape[0]

    # Member source. "internal": draw members from the model's own noise mechanism
    # (Mosaic NoiseGenerator / cSwiGLU -- the Phase-B Stage-B calibrated ensemble).
    # "p2": external IC-perturbation wrapper (deterministic backbones, e.g. patch_vit).
    # "auto" picks internal for Mosaic, p2 otherwise.
    member_source = str(cfg.eval.get("member_source", "auto"))
    if member_source == "auto":
        # internal = draw members from the model's own noise mechanism.
        # Mosaic always has one; patch-ViT has one only when noise_dim>0 (Fix 6/C3).
        _name = str(getattr(cfg.model, "name", ""))
        _has_noise = int(getattr(cfg.model, "noise_dim", 0)) > 0
        member_source = "internal" if (_name == "mosaic" or _has_noise) else "p2"
    member_seed = None  # set for the internal-noise path below; None for p2

    if member_source == "internal":
        # Eval member draws come solely from the NoiseGenerator seed (eval sets no
        # global seed). cfg.seed now threads into training (Fix 8), so pin the EVAL
        # draw seed explicitly here — default 42 reproduces the pre-Fix-8 baseline
        # bit-for-bit; override eval.member_seed to vary the ensemble draw.
        member_seed = int(cfg.eval.get("member_seed", 42))
        for _mod in lit.model.modules():
            if isinstance(_mod, NoiseGenerator):
                _mod.seed = member_seed
                _mod.generator = None
        n_members = int(cfg.eval.get("members", getattr(cfg.train, "eval_members", 16)))
        chunk = int(cfg.eval.get("batch", 16))
        print(f"Evaluating INTERNAL-noise ensemble: n={n_samples}  members={n_members}  "
              f"chunk={chunk}  device={device}")
        parts = []
        with torch.no_grad():
            for i in range(0, n_samples, chunk):
                xb = x[i:i + chunk]
                # adapter returns (b, M, lead, C, lat, lon) for M>1
                ob = lit.model(xb, num_noise_samples=n_members)
                parts.append(ob.permute(1, 0, 2, 3, 4, 5).detach().cpu())  # (M, b, ...)
        preds = torch.cat(parts, dim=1).numpy()   # (M, N, lead, C, lat, lon) standardized
    else:
        ensemble = P2Ensemble([lit], cfg)
        n_members = len(ensemble.seeds)
        print(f"Evaluating P2 (IC-perturbation) ensemble: n={n_samples}  members={n_members}  device={device}")
        with torch.no_grad():
            preds = ensemble.forecast(x).detach().cpu().numpy()  # (M, N, lead, C, lat, lon) standardized

    lats, lons = dm.latitude, dm.lon
    lead_weeks = list(cfg.data.lead_weeks)
    out_vars = dm.target_vars
    woy_window = int(cfg.eval.clim_woy_window)
    crps_fair = bool(cfg.eval.get("crps_fair", True))  # unbiased CRPS for equal footing
    provenance = provenance_columns(
        checkpoint_path=ckpt_path, crps_fair=crps_fair,
        n_members=n_members, member_seed=member_seed,
    )  # constant per-row stamp for metrics.csv (commit/checkpoint/crps_fair/n_members)

    target_mean = np.array([dm.normalizer[v]["mean"] for v in out_vars], dtype=np.float32)
    target_std = np.array([dm.normalizer[v]["std"] for v in out_vars], dtype=np.float32)

    def _to_physical(arr, channel_axis):
        shape = [1] * arr.ndim
        shape[channel_axis] = len(out_vars)
        return arr * target_std.reshape(shape) + target_mean.reshape(shape)

    y_true = _to_physical(y_true, channel_axis=2)
    preds = _to_physical(preds, channel_axis=3)

    # --- MAJ-3 (review 2026-07-14): optional COMMON-GRID coarsening for the
    # cross-resolution CRPSS comparison (ADR-0007 f3). Native-grid CRPSS is NOT
    # comparable across resolutions (CRPS magnitude scales with grid); when
    # eval.common_grid is set, area-conservatively coarsen the forecast members,
    # truth, and (below) the train climatology pool to the common grid BEFORE
    # India-box scoring. Default null => OFF => native-grid path unchanged.
    common_grid = cfg.eval.get("common_grid", None)
    if common_grid:
        from s2s.eval.regrid import equiangular_grid, regrid_conservative
        _dst_res = float(common_grid["resolution_deg"])
        _dst_lats, _dst_lons = equiangular_grid(
            _dst_res, with_poles=bool(common_grid.get("with_poles", False))
        )
        _nan_before = float(np.isnan(preds).mean()), float(np.isnan(y_true).mean())
        preds = regrid_conservative(preds, lats, lons, _dst_lats, _dst_lons)
        y_true = regrid_conservative(y_true, lats, lons, _dst_lats, _dst_lons)
        _nan_after = float(np.isnan(preds).mean()), float(np.isnan(y_true).mean())
        print(f"COMMON-GRID eval (MAJ-3): {len(lats)}x{len(lons)} -> "
              f"{len(_dst_lats)}x{len(_dst_lons)} @ {_dst_res} deg conservative")
        # NaN fractions must not blow up across the regrid; a jump to ~1.0 was the bug that
        # made every gate silently FAIL (0.0*NaN==NaN propagates through zero weights).
        print(f"  NaN frac preds {_nan_before[0]:.4f} -> {_nan_after[0]:.4f} | "
              f"truth {_nan_before[1]:.4f} -> {_nan_after[1]:.4f}")
        lats, lons = _dst_lats, _dst_lons

    # --- reconstruct the TEST weekly time axis so each (sample, lead) has a date,
    # for week-of-year pooling + absolute-event physical reconstruction. ---
    processed = Path(cfg.data.paths.processed)
    anom_all = xr.open_zarr(processed / "daily_anom.zarr")
    split = anom_all["split"].astype(str)
    test_daily = anom_all.sel(time=split == "test").drop_vars("split")
    test_weekly_time = daily_to_weekly_mean(test_daily[[out_vars[0]]]).time.values
    # Reconstruct the test init-week axis EXACTLY as assemble_arrays does, via the
    # shared helper (src/s2s/eval/time_align.py): valid_idx = arange(history_weeks-1,
    # n_time - max_lead). The previous code trimmed only max_lead off the END and
    # leaned on an init_times[-n_samples:] fallback to silently repair the missing
    # (history_weeks - 1) leading drop; that masked any real eval/assemble divergence.
    # Now we hard-assert length and endpoints so a mismatch fails loudly.
    max_lead = max(lead_weeks)
    history_weeks = int(cfg.data.history_weeks)
    init_times = reconstruct_init_times(test_weekly_time, history_weeks, max_lead)
    assert len(init_times) == n_samples, (
        f"init-time reconstruction length {len(init_times)} != n_samples {n_samples} "
        f"(history_weeks={history_weeks}, max_lead={max_lead}, "
        f"n_weekly={len(test_weekly_time)}); 03_evaluate and assemble_arrays disagree "
        f"on which test weeks are valid inits."
    )
    ds_time = getattr(test_ds, "time", None)
    if ds_time is not None:
        ds_time = np.asarray(ds_time)
        assert np.array_equal(init_times, ds_time), (
            "reconstructed init_times do not match the assembled test dataset times: "
            f"got [{init_times[0]} .. {init_times[-1]}], "
            f"assembled [{ds_time[0]} .. {ds_time[-1]}]."
        )

    rows = []
    year_rows = []  # per-(variable, lead, calendar-year) CRPSS for the separate CSV
    rank_store = {}
    reliab = {ev["name"]: {"p": [], "y": []} for ev in cfg.eval.reliability.events}

    for ci, var in enumerate(out_vars):
        clim_doy = _clim_doy(cfg, var)
        train_weekly = _train_weekly_anom(cfg, var)  # physical-unit anomalies + time
        if common_grid:
            from s2s.eval.regrid import regrid_conservative_da
            train_weekly = regrid_conservative_da(train_weekly, lats, lons)
            clim_doy = regrid_conservative_da(clim_doy, lats, lons)

        for li, lead_week in enumerate(lead_weeks):
            truth = y_true[:, li, ci]            # (N, lat, lon) physical anomaly
            members = preds[:, :, li, ci]        # (M, N, lat, lon) physical anomaly
            zero = np.zeros_like(truth)

            truth_da = xr.DataArray(
                truth, dims=("time", "latitude", "longitude"),
                coords={"latitude": lats, "longitude": lons},
            )

            # ---- CRPS: model + deterministic clim ----
            crps_model_f = truth_da.copy(data=crps_ensemble(members, truth, fair=crps_fair))
            crps_detclim_f = truth_da.copy(data=crps_ensemble(zero[np.newaxis, ...], truth))
            crps_model_box = _india_box(crps_model_f, cfg)
            crps_model = _latweighted_spatial_mean(crps_model_box)
            crps_model_samples = _latweighted_per_sample(crps_model_box)  # (N,) for bootstrap
            crps_detclim = _latweighted_spatial_mean(_india_box(crps_detclim_f, cfg))

            # ---- CRPS: probabilistic clim (woy-windowed train pool), per sample ----
            target_times = pd.DatetimeIndex(init_times) + pd.to_timedelta(7 * lead_week, unit="D")
            woy = target_times.isocalendar().week.values.astype(int)
            crps_prob_samples = np.empty(n_samples)
            crps_trend_samples = np.empty(n_samples)  # trend-aware reference (Fix 3 / C1)
            crps_trend2_samples = np.empty(n_samples)  # quadratic-trend sensitivity
            train_box = _india_box(train_weekly, cfg)
            trend_box = fit_linear_trend(train_box)   # per-gridpoint TRAIN-only slope
            poly_box = fit_poly_trend(train_box, degree=2)  # quadratic (train-only)
            for s in range(n_samples):
                t_box = _india_box(truth_da.isel(time=s), cfg)
                w = np.cos(np.deg2rad(t_box["latitude"].values))
                # (i) plain week-of-year probabilistic climatology
                clim_ens = climatology_woy_ensemble(train_box, int(woy[s]), window=woy_window)
                cf = crps_ensemble(clim_ens.values, t_box.values, fair=crps_fair)  # (lat, lon)
                crps_prob_samples[s] = float(np.average(np.nanmean(cf, axis=1), weights=w))
                # (ii) SAME pool, detrended onto this target date (trend null)
                clim_trend = climatology_woy_trend_ensemble(
                    train_box, int(woy[s]), target_times[s], window=woy_window, trend=trend_box
                )
                cft = crps_ensemble(clim_trend.values, t_box.values, fair=crps_fair)
                crps_trend_samples[s] = float(np.average(np.nanmean(cft, axis=1), weights=w))
                # (iii) SAME pool, quadratic detrend (residual-nonlinear-trend probe)
                clim_trend2 = climatology_woy_polytrend_ensemble(
                    train_box, int(woy[s]), target_times[s], window=woy_window, poly=poly_box
                )
                cft2 = crps_ensemble(clim_trend2.values, t_box.values, fair=crps_fair)
                crps_trend2_samples[s] = float(np.average(np.nanmean(cft2, axis=1), weights=w))
            crps_probclim = float(np.mean(crps_prob_samples))
            crps_climtrend = float(np.mean(crps_trend_samples))
            crps_climtrend2 = float(np.mean(crps_trend2_samples))

            skill_det = crpss(crps_model, crps_detclim)
            skill_prob = crpss(crps_model, crps_probclim)
            skill_trend = crpss(crps_model, crps_climtrend)
            skill_trend2 = crpss(crps_model, crps_climtrend2)

            # ---- paired bootstrap 95% CIs on crpss_vs_prob (Fix 2 / C2) ----
            # Resample paired per-sample CRPS (model vs prob-clim reference); no retrain.
            boot_cfg = cfg.eval.get("bootstrap", {}) or {}
            block_len = int(boot_cfg.get("block_len", 8))
            n_boot = int(boot_cfg.get("n_boot", 5000))
            boot_seed = int(boot_cfg.get("seed", 0))
            init_years = pd.DatetimeIndex(init_times).year.values
            blk = block_bootstrap_crpss(
                crps_model_samples, crps_prob_samples,
                block_len=block_len, n_boot=n_boot, seed=boot_seed,
            )
            yr = year_bootstrap_crpss(
                crps_model_samples, crps_prob_samples,
                init_years, n_boot=n_boot, seed=boot_seed,
            )
            # same machinery, trend-aware reference (Fix 3 / C1)
            blk_tr = block_bootstrap_crpss(
                crps_model_samples, crps_trend_samples,
                block_len=block_len, n_boot=n_boot, seed=boot_seed,
            )
            yr_tr = year_bootstrap_crpss(
                crps_model_samples, crps_trend_samples,
                init_years, n_boot=n_boot, seed=boot_seed,
            )
            blk_tr2 = block_bootstrap_crpss(
                crps_model_samples, crps_trend2_samples,
                block_len=block_len, n_boot=n_boot, seed=boot_seed,
            )
            _by_prob = crpss_by_year(crps_model_samples, crps_prob_samples, init_years)
            _by_trend = crpss_by_year(crps_model_samples, crps_trend_samples, init_years)
            _by_trend2 = crpss_by_year(crps_model_samples, crps_trend2_samples, init_years)
            for _rp, _rt, _rt2 in zip(_by_prob, _by_trend, _by_trend2):
                assert _rp["year"] == _rt["year"] == _rt2["year"]
                year_rows.append({
                    "variable": var, "lead_week": lead_week,
                    "year": _rp["year"], "n_samples": _rp["n_samples"],
                    "crpss_vs_prob": _rp["crpss_vs_prob"],
                    "crpss_vs_trend": _rt["crpss_vs_prob"],
                    "crpss_vs_trend2": _rt2["crpss_vs_prob"],
                })

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
                **provenance,
                "variable": var, "lead_week": lead_week,
                "crps_model": crps_model,
                "crps_clim_det": crps_detclim,
                "crps_clim_prob": crps_probclim,
                "crps_clim_trend": crps_climtrend,
                "crpss_vs_det": skill_det,
                "crpss_vs_prob": skill_prob,
                "crpss_vs_prob_ci_lo": blk["ci_lo"],
                "crpss_vs_prob_ci_hi": blk["ci_hi"],
                "crpss_vs_prob_boot_se": blk["boot_se"],
                "crpss_vs_prob_ci_lo_yr": yr["ci_lo"],
                "crpss_vs_prob_ci_hi_yr": yr["ci_hi"],
                "crpss_vs_trend": skill_trend,
                "crpss_vs_trend_ci_lo": blk_tr["ci_lo"],
                "crpss_vs_trend_ci_hi": blk_tr["ci_hi"],
                "crpss_vs_trend_boot_se": blk_tr["boot_se"],
                "crpss_vs_trend_ci_lo_yr": yr_tr["ci_lo"],
                "crpss_vs_trend_ci_hi_yr": yr_tr["ci_hi"],
                "crps_clim_trend2": crps_climtrend2,
                "crpss_vs_trend2": skill_trend2,
                "crpss_vs_trend2_ci_lo": blk_tr2["ci_lo"],
                "crpss_vs_trend2_ci_hi": blk_tr2["ci_hi"],
                "boot_block_len": blk["block_len"],
                "boot_n": blk["n_boot"],
                "acc_mean": acc_box,
                "rmse_mean": rmse_box,
                "spread_error_ratio": ser,
            })

    table = pd.DataFrame(rows)
    csv_path = results_dir / "metrics.csv"
    table.to_csv(csv_path, index=False)

    # Per-calendar-year CRPSS vs probabilistic climatology (Fix 2 / C2 sensitivity).
    year_table = pd.DataFrame(year_rows)
    year_table.to_csv(results_dir / "crpss_by_year.csv", index=False)

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

    print("\n=== Honest eval -- India box, test split "
          "(t2m: Kelvin anomaly; precip: log1p(mm/day) transformed anomaly) ===\n")
    _ci_cols = ["crps_model", "crpss_vs_prob", "crpss_vs_trend", "crpss_vs_trend_ci_lo",
                "crpss_vs_trend_ci_hi", "crpss_vs_trend2", "spread_error_ratio"]
    for var in out_vars:
        sub = table[table["variable"] == var].set_index("lead_week")
        print(f"--- {var} ---  (crpss_vs_prob = plain WOY clim; crpss_vs_trend = trend-detrended clim; 95% block CI)")
        print(sub[_ci_cols].to_string(float_format=lambda v: f"{v:.5f}"))
        print()

    gate_leads = list(cfg.eval.gate.lead_week)
    gate_thr = float(cfg.eval.gate.get("threshold", 0.0))
    g = table[table["lead_week"].isin(gate_leads)]
    gate_pass = bool((g["crpss_vs_prob"] > gate_thr).all()) if len(g) else False
    # CI-aware gate: does the 95% block CI exclude the threshold for EVERY gate cell?
    ci_excludes = bool((g["crpss_vs_prob_ci_lo"] > gate_thr).all()) if len(g) else False
    # C1 test: does the gate survive the TREND null (detrended reference)?
    trend_pass = bool((g["crpss_vs_trend"] > gate_thr).all()) if len(g) else False
    trend_ci_excludes = bool((g["crpss_vs_trend_ci_lo"] > gate_thr).all()) if len(g) else False
    print(
        f"Decision gate (lead weeks {gate_leads}, metric={cfg.eval.gate.metric} "
        f"vs {cfg.eval.gate.reference}, threshold {gate_thr}): "
        f"{'PASS' if gate_pass else 'FAIL'}  "
        f"[95% block CI excludes {gate_thr} at all gate cells: {'YES' if ci_excludes else 'NO'}]"
    )
    print(
        f"Trend-null gate (vs DETRENDED clim, crpss_vs_trend > {gate_thr}): "
        f"{'PASS' if trend_pass else 'FAIL'}  "
        f"[95% block CI excludes {gate_thr} at all gate cells: {'YES' if trend_ci_excludes else 'NO'}]"
    )
    for _, r in g.iterrows():
        print(
            f"    {r['variable']:<24} wk{int(r['lead_week'])}: "
            f"vs_prob={r['crpss_vs_prob']:.4f} CI[{r['crpss_vs_prob_ci_lo']:.4f},{r['crpss_vs_prob_ci_hi']:.4f}]  "
            f"vs_trend={r['crpss_vs_trend']:.4f} CI[{r['crpss_vs_trend_ci_lo']:.4f},{r['crpss_vs_trend_ci_hi']:.4f}]"
        )

    # Non-dynamical asymptote: at the longest lead there is little/no dynamical
    # memory, so CRPSS there is ~ the pure distributional (trend + calibrated
    # spread) advantage. crpss_vs_trend at wk3-4 collapsing toward this asymptote
    # would mean the gate is trend, not S2S skill (C1).
    max_lead = max(lead_weeks)
    wk6 = table[table["lead_week"] == max_lead]
    print(f"\nNon-dynamical asymptote (longest lead = wk{max_lead}):")
    for _, r in wk6.iterrows():
        print(
            f"    {r['variable']:<24} crpss_vs_prob={r['crpss_vs_prob']:.4f}  "
            f"crpss_vs_trend={r['crpss_vs_trend']:.4f}  "
            f"crpss_vs_trend2={r['crpss_vs_trend2']:.4f}"
        )
    print(f"\nSaved: {csv_path}, {results_dir / 'crpss_by_year.csv'} + PNGs in {results_dir}")


if __name__ == "__main__":
    main()
