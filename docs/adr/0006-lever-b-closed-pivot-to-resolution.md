# ADR-0006: Lever (b) closed — SST boundary forcing is null; pivot to resolution scaling (f)

**Status:** Accepted (Phase C)
**Date:** 2026-07-08
**Foundation:** `integration/all-fixes`
**Refs:** ADR-0005 (indices design), `results/eval_leverb_idx_s0`, `results/eval_fix45_full_s0`

## Context

On the fixed foundation (M1–M5, C1–C3), the SST-indices increment (Niño3.4 + DMI, seed 0) was
evaluated with block-bootstrap 95% CIs against the fixed seed-0 baseline (`mosaic_fix45`):

| wk3 / wk4 crpss_vs_prob | Niño3.4+DMI | baseline (fix45) |
|-------------------------|-------------|------------------|
| t2m    | 0.208 [0.156, 0.257] / 0.196 [0.151, 0.238] | 0.224 [0.167, 0.279] / 0.208 [0.155, 0.260] |
| precip | 0.082 [0.058, 0.105] / 0.075 [0.051, 0.099] | 0.089 [0.069, 0.110] / 0.083 [0.066, 0.100] |

The indices point estimate is marginally **lower** at all four gate cells and sits well inside
the baseline CI (and vice versa) — no statistically distinguishable gain. Precip, the variable
ENSO/IOD should most help, shows nothing. The model already extracts the usable SST signal from
the SST field it is fed; making ENSO/IOD explicit is redundant.

## Decision

**Close lever (b).** Both SST forms — raw longer-history (retired, pre-fix) and now explicit
indices (CI-confirmed null) — fail to move skill. `data.sst_indices` defaults to `[]` (machinery
retained, off).

Combined with lever (a) (mesh/interpolation, closed — M1 confirmed the 0.3 figure was an
untrained-gain artifact), **two independent input-representation levers have returned null.**
The signal is that the ceiling is not input representation but **spatial resolution / intrinsic
S2S predictability at 5.625°**. The model is a valid forecaster (both gates pass, CIs exclude
zero, t2m SER ~0.9–1.0); to *raise* the bar we must change fidelity.

**Pivot to lever (f): scale resolution 5.625° → 1.5°** (WeatherBench2 240×121 equiangular
with-poles, the WB2 standard eval grid; ~14× the grid points). Detailed design — target grid,
backbone at high resolution (the Mosaic block-sparse-attention question vs patch-ViT with the
C3 stochastic-member mechanism), pole handling, data rebuild, and compute staging — lands as
ADR-0007 before major work.

## Consequences

- Phase-C input-representation levers (a, b) are exhausted and documented as such — a clean,
  CI-backed negative, not a gap.
- The indices code stays as a tested, config-gated capability (off), so the null is reproducible.
- (f) is a substantial undertaking (new data pull + processing, ~14× compute, pole handling,
  and a real backbone/attention decision) — scoped and staged in ADR-0007, dev-subset first.
