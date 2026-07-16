# Phase-C lever f: 1.5° resolution — first calibrated PASS (seed 0)

**Date:** 2026-07-12
**Config:** model=mosaic_15deg, data=era5_india_15deg (240×121, with poles), fair-CRPS,
train_members=8, batch_size=4, **accumulate_grad_batches=8** (effective gradient batch 32),
30 epochs. Checkpoint: `epoch=12-val_loss=0.2886` (best-val at ep12, calibration recovered).

## Result (India box, test split, 95% block-bootstrap CI)

| wk3 / wk4 crpss_vs_prob | 1.5° (this run) | 5.625° baseline (mosaic_fix45) |
|---|---|---|
| t2m    | 0.165 [0.119, 0.212] / 0.152 [0.116, 0.188] | 0.224 [0.167, 0.279] / 0.208 [0.155, 0.260] |
| precip | 0.107 [0.085, 0.131] / 0.037 [0.010, 0.063] | 0.089 [0.069, 0.110] / 0.083 [0.066, 0.100] |
| t2m SER | 0.694 / 0.664 | ~0.90 |

Both the decision gate (crpss_vs_prob>0) and the trend-null gate PASS at wk3/4, CIs exclude 0.

## Reading

- **Calibration collapse fixed (ADR-0008 confirmed).** Gradient accumulation moved best-val
  from ep1→ep12 and SER from ~0.006 to ~0.65–0.90 — the small-batch dynamics were the cause.
- **Resolution ≈ parity, not a win.** t2m CRPSS is *lower* at 1.5° (CIs overlap the baseline —
  comparable, not clearly worse); precip wk3 is comparable/slightly higher; **precip wk4 is a
  clear regression** (1.5° 0.037 vs baseline 0.083, CIs barely non-overlapping). Net: 1.5° is
  roughly at parity-to-slightly-below the 5.625° model, at ~14× compute. Consistent with the
  a/b/f pattern that the ceiling is intrinsic S2S predictability, not resolution/representation.
- **Two confounds before calling it a true parity/loss:** (1) the 1.5° model is still
  **under-dispersed (SER ~0.69 vs 0.90)** — residual under-dispersion penalizes CRPS, so some of
  the t2m gap is un-realised calibration, not a resolution ceiling; (2) single seed, and the
  1.5° config (dim, lr, nside, regularization) is inherited from 5.625°, not tuned.

## Next levers for the 1.5° model (the deliverable)

1. Improve calibration (SER 0.69→~0.9): more eval members / longer or better-regularised
   training / noise-mechanism check — likely lifts CRPSS since under-dispersion caps it.
2. Multi-seed (1/2) confirmation on the fixed pipeline.
3. `mosaic_15deg` config tuning for the finer grid.
Pending the Fable code review first (do not burn more compute until review is resolved).

## Note
The earlier `last.ckpt` freeze did NOT recur — best=ep12 saved correctly — confirming it was
disk exhaustion during the prior long run, not a checkpoint-config bug.
