# ADR-0008: 1.5° training calibration — gradient accumulation to match the batch regime

**Status:** Accepted (Phase C, lever f)
**Date:** 2026-07-12
**Refs:** ADR-0007 (lever f), `results/eval_15deg_s0`, `results/eval_fix45_full_s0`

## Context

The first full 1.5° run (30 epochs, seed 0) FAILED both gates decisively — but the result is
**confounded, not a clean resolution verdict**:

- t2m **SER ≈ 0.006–0.009** (target ~1.0): the ensemble is ~100× under-dispersed — the exact
  collapse fair-CRPS exists to prevent, and which the 5.625° model avoided (SER ~0.9 at its best
  epoch ~14).
- Best-val is **epoch 1**, then monotonic overfit. The best-val checkpoint is therefore
  pre-calibration — the noise→spread pathway never developed.
- `last.ckpt` was byte-identical to the epoch-1 file (frozen). The checkpoint config is standard
  and checkpointed the 5.625° baseline fine, so this is almost certainly **disk exhaustion**
  during the long run (processed_15deg is ~14× larger), not a code regression — verify with `df`.

## Diagnosis

The standout difference from the 5.625° runs that calibrated: **batch size**. Every 5.625°
fair-CRPS run used `batch_size=32`; the 1.5° run was forced to `batch_size=4` (memory: 14× grid
points → the FFN OOMs at bs≥8 with 8 members). That is **8× fewer distinct samples per gradient
step** → noisy small-batch updates that overfit before ensemble spread develops. This is a
training-dynamics failure, independent of resolution.

## Decision

**Gradient accumulation.** Keep `batch_size=4` (the memory ceiling) but accumulate 8 micro-batches
before each optimizer step (`accumulate_grad_batches=8`), so the gradient is averaged over
`4 × 8 = 32` distinct samples — matching the 5.625° regime — at **no extra memory** and ~the same
wall-clock (same number of forward passes; fewer optimizer steps). Mosaic uses RMSNorm (no
batch-statistic coupling), so accumulation is gradient-equivalent to a true batch of 32.

Wired as `cfg.train.accumulate_grad_batches` (default 1 = unchanged) in `train.py`.

## Plan

1. Verify Datastorage disk before retraining (the frozen-checkpoint tell).
2. Short confirmation run (~6 epochs, `accumulate_grad_batches=8`): check that val no longer
   collapses at epoch 1 and t2m SER recovers toward ~0.9. Cheap go/no-go.
3. If SER recovers → full 30-epoch retrain + honest eval vs `mosaic_fix45`. Only a **calibrated**
   1.5° model gives a fair resolution read.
4. If it still overfits early → add regularization / lr / epoch tuning (drop_rate, weight_decay);
   gradient checkpointing or multi-GPU DDP held in reserve if we ever want effective batch > 32.

## Consequences

- 1.5° is a hard project requirement, so the model must be made to train correctly here; this is
  the first, cheapest, most-targeted intervention (matches the known batch difference).
- No memory increase, ~same wall-clock. Backward-compatible (default 1) for all 5.625° runs.


## Outcome (2026-07-12) — CONFIRMED

Retrain with `accumulate_grad_batches=8` (effective batch 32): best-val moved to **epoch 12**
(was epoch 1), t2m **SER 0.66–0.69** (was ~0.006), both gates **PASS** (CIs exclude 0). t2m
crpss_vs_prob wk3/4 = 0.165/0.152, precip = 0.107/0.037. The batch-dynamics diagnosis is
confirmed. See `results/phase_c_lever_f_note.md`. Residual under-dispersion (SER ~0.69 vs the
5.625° ~0.90) remains — a follow-up calibration lever, not a collapse.
