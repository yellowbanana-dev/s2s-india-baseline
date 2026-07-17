# ADR-0007: Lever (f) — scale resolution to 1.5° with Mosaic block-sparse attention

**Status:** Accepted (Phase C)
**Date:** 2026-07-08
**Foundation:** `integration/all-fixes`
**Refs:** ADR-0002 (Mosaic vendoring; ops.py deliberately excluded), ADR-0006 (pivot to f)

## Context

Levers (a) mesh/interpolation and (b) SST boundary forcing both returned CI-confirmed nulls —
two input-representation bets that did not move skill. The remaining hypothesis is that the
ceiling is **spatial resolution / intrinsic predictability at 5.625°**, not input. This ADR
scales the pipeline to **1.5°** (WeatherBench2 `240 x 121` equiangular *with poles* — WB2's own
standard eval grid; ~14× the 32×64 grid points), where the India box becomes ~14× finer and
precip in particular has room to improve.

## Decisions

- **Resolution:** 5.625° → **1.5°** (240 lon × 121 lat, with poles). Confirm the exact WB2 store
  on-cluster (`gsutil ls`) before pulling. Separate processed path (`processed_15deg`) so the
  5.625° canonical is never clobbered.
- **Backbone:** **Mosaic + block-sparse attention.** At nside≈64 (49,152 HEALPix pixels for the
  29,040-point grid) dense global attention is O(N²)-infeasible, so the Stage-A "global = one
  block" setting can't hold. We re-enable the block-sparse path (local block + compressed +
  top-k selection) that ADR-0002 removed with the Triton `ops.py`. This keeps Mosaic — the
  calibrated fair-CRPS PASS vehicle — rather than switching backbones.
- **Staging:** **dev-subset dry run first** — validate the full 1.5° pipeline (data build,
  memory, throughput, one train+eval) on a few `dev_years` before the full 1979–2023 train.

## Increment plan

- **f0 (this commit) — resolution plumbing** (no sparse kernel; testable without Triton):
  `MosaicBackbone` grid derived from the datamodule coords, not the `(32,64)` hardcode;
  `verify_pull` made resolution/pole-aware; `configs/data/era5_india_15deg.yaml` added.
- **f1 — block-sparse attention:** obtain the sparse kernel (see Open question), wire the sparse
  path in `primitives.py` (replace the `ImportError` stub), add `configs/model/mosaic_15deg.yaml`
  (nside=64, bottleneck_nside, `sparse_every>0`, `sparse_block_size/count`, dims), attribution +
  `triton` dep in `environment.yml` if Triton-vendored. Add sparse-path tests (GPU-gated).
- **f2 — dev-subset dry run:** build 1.5° dev data (subset years + splits), train+eval a few
  epochs; validate correctness, GPU memory, and wall-clock. Fail-fast on plumbing/OOM cheaply.
- **f3 — full-record train + eval:** full 1.5° train, honest eval (C1 trend-aware, C2 bootstrap
  CIs) over the India box; compare to the 5.625° fixed baseline (`mosaic_fix45`) on a COMMON
  grid (see "Pre-registration" below) — native-grid CRPSS is not comparable across resolutions.

## Open question (blocks f1)

The upstream Mosaic `ops.py` (Triton block-sparse kernel) was never vendored, and an attempt to
fetch it upstream timed out. Two ways to source the sparse path:
1. **Vendor upstream `ops.py`** (Triton) — faithful to the paper/kernel, but adds a Triton/CUDA
   build dependency and a large validation surface. Needs the upstream file.
2. **PyTorch reference implementation** of the same block-sparse strategy — portable (no Triton
   build dep), more reproducible for a thesis, slower; needs the algorithm spec to match faithfully.
Resolved with the human before f1 (which path, and whether the upstream `ops.py` can be supplied).

## Pre-registration — cross-resolution comparison protocol (review 2026-07-14, MAJ-3)

Native-grid CRPSS is NOT comparable across resolutions: CRPS magnitude scales with grid (the
1.5° target carries more small-scale variance and a heavier precip double-penalty, and the
probabilistic-climatology reference's own CRPS also changes with resolution). Registered BEFORE
the f3 comparison is interpreted, the headline 1.5°-vs-5.625° readout is made on a COMMON grid:

- Both models are scored with `eval.common_grid={resolution_deg: 5.625}`. The 1.5° forecast
  members, truth, and the train climatology pool are area-conservatively coarsened to the
  64×32 5.625° grid over the India box before CRPS; the 5.625° baseline coarsens to itself
  (identity), so its numbers are unchanged. This equalises both the CRPS scale and the reference.
- Native-grid CRPSS is reported as a SECONDARY diagnostic only and must not be compared directly
  across the two resolutions.
- Conclusion rule (supersedes the Consequences wording below): resolution is credited as the
  ceiling only if the COMMON-GRID 1.5° CRPSS exceeds the 5.625° baseline with paired-bootstrap-
  separated CIs. A native-grid-only lift does not qualify.

## Consequences

- ~14× grid points → smaller batch and likely gradient checkpointing; memory validated in f2.
- Poles enter the global input (India eval is far from poles; cos-lat weight → 0 at ±90 is fine).
- If Triton-vendored: CC-BY-NC attribution on `ops.py`; `triton` runtime dep.
- If 1.5° lifts CRPSS beyond the baseline CI **on the common grid** (see Pre-registration),
  resolution was the ceiling — the Phase-C result. If not, intrinsic S2S predictability at these
  scales is the limit, and the thesis pivots to consolidation/benchmarking (levers d/e).
