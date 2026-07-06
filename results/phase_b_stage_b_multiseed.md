# Phase-B Stage-B Multi-seed Consolidation

**Branch:** phase-b-consolidate  
**Date:** 2026-07-04  
**Config:** model=mosaic, loss=fair_crps, train_members=8, eval_members=16 (internal-noise ensemble)  
**Scoring:** unbiased (fair) CRPS for all seeds — equal footing vs climatology_prob

---

## 3-Seed Results Table

| seed | best epoch | best val (fair-CRPS) | t2m crpss_vs_prob wk3 | t2m crpss_vs_prob wk4 | prec crpss_vs_prob wk3 | prec crpss_vs_prob wk4 | t2m SER wk3/4 | gate PASS? |
|------|-----------|---------------------|----------------------|----------------------|----------------------|----------------------|---------------|-----------|
| 0 | 13 | 0.2728 | 0.204 | 0.215 | 0.194 | 0.185 | 0.879 / 0.922 | YES |
| 1 | 16 | 0.2734 | 0.210 | 0.201 | 0.182 | 0.189 | 0.921 / 0.887 | YES |
| 2 | 36 | 0.2824 | 0.083 | 0.131 | 0.183 | 0.179 | 0.424 / 0.598 | YES |

Gate threshold: crpss_vs_prob > 0.0 at lead weeks 3–4.

---

## Key Findings

**Gate:** PASS for all 3 seeds. The Stage-B fair-CRPS result is not a seed fluke — seeds 0 and 1 are highly consistent (t2m wk3/4 ~0.20–0.21, SER ~0.88–0.92).

**Seed-2 outlier:** Seed-2 passes the gate but is weaker on t2m (crpss_vs_prob 0.083/0.131 vs ~0.20 for seeds 0/1) and severely underdispersed on t2m (SER 0.42/0.60). Precipitation metrics are consistent across all seeds. This suggests seed-2 landed in a less calibrated basin for the temperature variable specifically. Best epoch also arrived much later (ep36 vs ep13/16), indicating a slower convergence trajectory.

**Spread-error ratio:** Seeds 0 and 1 are near-calibrated (SER 0.88–0.92 ≈ target 1.0), a dramatic improvement over the MSE baseline (~0.004). Seed-2 precip SER is also good (0.86–0.87); t2m SER is the outlier.

**Comparison to MSE deterministic Mosaic (biased scoring, for context):**
- MSE seed-0: t2m crpss_vs_prob wk3/4 = −0.123/−0.135, SER ~0.004 → gate FAIL
- fair-CRPS seed-0: t2m crpss_vs_prob wk3/4 = 0.204/0.215, SER ~0.88 → gate PASS

---

## Eval Artifacts

| seed | results dir |
|------|-------------|
| 0 | `results/eval_mosaic_crps_s0_fair/` |
| 1 | `results/eval_mosaic_crps_s1/` |
| 2 | `results/eval_mosaic_crps_s2/` |

Checkpoints in `/Datastorage/scdlds_bharat/s2s/checkpoints/seed_{N}/` (not committed).
