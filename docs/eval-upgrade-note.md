# Eval-upgrade note — honest Phase-A baseline on the probabilistic bar

**Date:** 2026-06-26
**Commit:** `9feae8b` (eval upgrade `6ff54f0` → GPU `fb4ff6d` → lat/lon fix + results `9feae8b`)
**Run:** `seed_0/epoch=21-val_loss=0.4185.ckpt`, P2 IC-perturbation ensemble (10 members,
`ic_perturbation_std=0.05`), TEST split 2018–2023, India box, n=256 weekly samples, H100.

This records the **first honest numbers** produced by the upgraded evaluation
(phase-b-plan §6.1) — the metrics we will judge every Phase-B model change against,
captured *before* any model change.

## What the upgrade added

- **CRPSS vs probabilistic climatology** — the honest bar. Reference = a
  week-of-year-windowed (±3 weeks, circular) pool of TRAIN weekly anomalies
  (`climatology_woy_ensemble`), train-only so leakage-safe. This replaces Phase A's
  lenient **deterministic** zero-anomaly climatology (CRPS ≡ MAE) as the gate.
- **Calibration:** Talagrand rank histogram + spread-error ratio (spread ÷ RMSE of
  the ensemble mean; ≈1 is calibrated, <1 under-dispersed).
- **Deterministic ACC and RMSE** of the ensemble mean.
- **Reliability diagrams** for P(t2m anom>0), P(precip anom>0), and India-context
  absolute **weekly-mean** events P(T2m>40 °C) and P(precip>50 mm/day) — the latter
  reconstructed to physical units via the train climatology (weekly-mean
  exceedances, NOT daily extremes).

## Results (India box, test split)

### 2 m temperature

| lead | crps_model | crps_clim_det | crps_clim_prob | CRPSS_vs_det | CRPSS_vs_prob | acc_mean | rmse_mean | spread/err |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.75208 | 0.97498 | 0.78422 | +0.229 | +0.041 | 0.631 | 1.056 | 0.019 |
| 2 | 0.82406 | 0.97351 | 0.78259 | +0.154 | −0.053 | 0.490 | 1.171 | 0.012 |
| 3 | 0.83971 | 0.97401 | 0.78257 | +0.138 | **−0.073** | 0.446 | 1.199 | 0.012 |
| 4 | 0.84806 | 0.97329 | 0.78150 | +0.129 | **−0.085** | 0.416 | 1.215 | 0.011 |
| 5 | 0.85943 | 0.97315 | 0.78160 | +0.117 | −0.100 | 0.384 | 1.234 | 0.011 |
| 6 | 0.86799 | 0.97531 | 0.78341 | +0.110 | −0.108 | 0.366 | 1.247 | 0.010 |

### Total precipitation (24 hr accumulation, m/day)

| lead | crps_model | crps_clim_det | crps_clim_prob | CRPSS_vs_det | CRPSS_vs_prob | acc_mean | rmse_mean | spread/err |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.00181 | 0.00185 | 0.00166 | +0.026 | −0.091 | 0.320 | 0.00292 | 0.009 |
| 2 | 0.00187 | 0.00186 | 0.00166 | −0.007 | −0.128 | 0.173 | 0.00304 | 0.008 |
| 3 | 0.00187 | 0.00186 | 0.00166 | −0.004 | **−0.125** | 0.137 | 0.00307 | 0.008 |
| 4 | 0.00189 | 0.00186 | 0.00166 | −0.012 | **−0.133** | 0.137 | 0.00307 | 0.007 |
| 5 | 0.00191 | 0.00186 | 0.00166 | −0.026 | −0.149 | 0.116 | 0.00308 | 0.007 |
| 6 | 0.00191 | 0.00187 | 0.00167 | −0.025 | −0.147 | 0.074 | 0.00310 | 0.006 |

**Decision gate** (weeks 3–4, CRPSS vs probabilistic climatology, threshold 0): **FAIL.**

## Interpretation

1. **The deterministic-vs-probabilistic gap is the whole story.** Against the
   deterministic climatology, temperature keeps its Phase-A skill (CRPSS_det +0.13…+0.14
   at weeks 3–4). Against the *probabilistic* climatology, temperature is **negative from
   week 2 onward** (−0.07…−0.09 at weeks 3–4). The temperature "skill" lives entirely in
   the ensemble **mean** (ACC 0.45/0.42 at weeks 3–4, real signal) — it does **not** beat
   a calibrated climatological *distribution*.

2. **Under-dispersion is now measured, not just suspected.** `spread_error_ratio ≈ 0.01–0.02`
   across all leads — the ensemble spread is ~50–100× too small (calibrated ≈ 1). The rank
   histograms (`results/eval/rank_hist_*.png`) are correspondingly extreme-U-shaped: truth
   lands outside the ensemble almost every time. This is exactly the limitation Phase A
   flagged (report §7.1), now quantified by the honest harness.

3. **Precipitation is below climatology on the honest bar at every lead** (CRPSS_prob
   −0.09…−0.15), worse than its deterministic-bar near-zero. Consistent with Phase A:
   little real precip signal at 5.625°, plus the same dispersion problem.

## Phase-B implications (unchanged direction, now with a measured target)

- The under-dispersion (spread/err ≈ 0.01) is the **single biggest lever**: the Phase-B
  **learned functional perturbations** (build step 4) must lift spread/err toward 1 and
  push CRPSS_vs_prob > 0 at weeks 3–4. This is the core contribution and now has a concrete
  before-number.
- Temperature's positive ACC at weeks 3–4 confirms there *is* mean-state signal to make
  probabilistically useful — the model isn't starting from nothing.
- Precipitation needs the dedicated treatment (build step 6) on top of calibration.

## Known follow-ups (not blocking)

- The week-of-year CRPS scoring loop is per-sample × per-lead numpy — the runtime
  bottleneck after data assembly. Vectorize the WoY pooling for Phase-B iteration speed.
- Numbers here are a single IC-perturbation checkpoint (a smoke test of the eval path),
  not an ensemble of independently-trained members — the absolute CRPSS will shift once
  the real ensemble exists; the *honest harness* is what's now locked in.
