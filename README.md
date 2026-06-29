# S2S India Baseline (G1 × P2)

A subseasonal-to-seasonal (S2S) weather forecasting **baseline** for the Indian
region. This is Phase A: its job is **infrastructure + a number
to beat**, not a good model. If it works, every later phase has a foundation and a
yardstick.

## Locked decisions

| Decision | Value | Why |
|---|---|---|
| Backbone (axis 1) | **G1** — patch ViT/Swin | Simplest, most reproducible; baseline only |
| Probabilistic (axis 2) | **P2** — cheap ensemble (multi-seed + IC perturbation) | Cheapest route to a scorable spread |
| Targets | **2m temperature + total precipitation** anomalies | Two channels; precip handled separately (skewed) |
| Lead times | **Weekly means, weeks 1–6** | Direct prediction, not autoregression |
| Input domain | **GLOBAL** (32×64 @ 5.625°) | S2S skill over India comes from *remote* drivers (MJO, ENSO, IOD, BSISO) — never crop the input |
| Eval domain | **India box** (lat 5–40°N, lon 65–100°E) | Score where we care; input stays global |
| Resolution | **5.625°** | Fastest iteration; correctness now, fidelity later |
| Forecast strategy | **Direct weekly-mean anomaly prediction** | Avoids autoregressive error accumulation |

## Success criterion (the decision gate)

Beat **climatology** *and* **persistence** at **week 3–4** over the India box,
measured on **CRPS**, with a fully reproducible run. Hitting this is "done" for
Phase A.

## Train/val/test split

**Phase A (now):** a simple **chronological** split — train on the bulk of the
record, validate on a few years, test on the *most recent* years. Default in
`configs/data/era5_india.yaml`: train 1979–2012, val 2013–2017, test 2018–2023,
with a 4-week embargo between blocks. The exact years are a **knob, not a fixed
decision** — change them freely. Testing on the latest years is deliberate: it
mimics operational use (forecast the future from the past) and forces the model
to handle the warming trend honestly.

**Final evaluation (Phase B onward):** switch to **leave-one-year-out / k-fold
cross-validation over years**. S2S has tiny *effective* sample sizes — weather is
autocorrelated over weeks, so a year holds only a handful of independent week 3–6
forecasts — and a single held-out block gives noisy skill estimates. Year-wise CV
is the defensible choice for thesis-grade numbers; it costs ~k× compute, which the
GPU budget absorbs by then. (Tracked as a Phase-B task, not done in Phase A.)

## The cardinal rule (read before touching the data)

Every statistic — climatology and normalization mean/std — is computed on
**training years only** and then *applied* to validation/test. A climatology that
accidentally includes test years inflates skill invisibly and never throws an
error. This is the #1 way S2S results turn out to be fiction. Verify it by hand.

## Stack

- **PyTorch + Lightning** — model + training loop
- **xarray + Zarr + Dask** — gridded data (named dims, lazy chunked reads)
- **Hydra** — config-driven experiments (one experiment = one config + one commit + one run)
- **Weights & Biases** — experiment tracking
- **Git + DVC** — code + data versioning

## Layout

```
configs/        Hydra configs (data / model / train / eval)
src/s2s/data/   Stage 1 (download) + Stage 2 (climatology, splits, datamodule)
src/s2s/models/ G1 patch-ViT
src/s2s/eval/   Metrics (CRPS/ACC/RMSE/reliability) + baselines (climatology, persistence)
scripts/        Numbered entry points to run each stage
tests/          Data-integrity tests (highest-ROI tests in ML)
data/           raw/ (Zarr pulls) + processed/ (anomalies, splits) — gitignored
```

## Run order (once implemented)

```bash
python scripts/00_pull_data.py        # Stage 1: ERA5 @ 5.625° from WeatherBench2
python scripts/01_build_dataset.py    # Stage 2: climatology, anomalies, splits
python scripts/02_run_baselines.py    # Climatology + persistence over India box
python -m s2s.train                   # Train G1 + P2 ensemble (Hydra entrypoint)
```

## Status

Phase A scaffold. All `src/` modules are stubs (signatures + docstrings + danger-zone
notes). Implement in task order: data (#2–3) → baselines + eval (#4–5) → model (#6) →
training (#7) → ensemble (#8) → decision gate (#9).
