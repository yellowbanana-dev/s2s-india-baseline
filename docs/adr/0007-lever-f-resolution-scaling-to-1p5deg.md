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

## Attention path as actually run (review 2026-07-14 — MAJ-1, MIN-4)

The block-sparse path is enabled with the **fine top-k selection branch deferred** to the
un-vendored Triton kernel: in `primitives.py` the selection output reuses the compressed result
(`o_slc = o_cmp`), so the 3-way strategy gate reduces to `w0·local + (w1+w2)·compressed`. The
model as trained is therefore a **local + mean-pooled-compressed Mosaic**, not the full
local+compressed+selection architecture. Consequences to state wherever f3 results appear:

- **Label honestly.** At 5.625° the calibrated baseline ran *dense global attention every layer*
  (`block_attn_size = npix`). The 1.5° model gives each token a fine receptive field only over
  its local block; all longer-range context arrives as block-mean-pooled summaries. This is a
  strictly weaker attention structure — report f3 as "local+compressed 1.5° Mosaic," not "the
  same architecture at higher resolution." (Note: the epoch-12 PASS run holds this attention
  fixed vs the failed run, so the collapse was batch-dynamics, not attention — but the *skill
  ceiling* of the approximation is still open and is what the MAJ-2 sparse-ablation control tests.)
- **Gate degeneracy / no drop-in.** Because `o_slc` *is* `o_cmp`, only `w1+w2` is identifiable;
  the cmp/slc logit split is arbitrary. A checkpoint trained this way is **not** a drop-in for a
  future selection kernel — gates are not transferable, so expect a retrain, not an upgrade.
  `gqa_ratio` is now asserted to be 1 at construction (this PyTorch reference supports MHA only).
- **The degenerate slot BIASES the gate 2:1 toward compressed at init (found 2026-07-20).**
  Beyond being unidentifiable, `o_slc = o_cmp` means the 3-way softmax puts ~1/3 on each slot at
  initialisation while TWO slots hold the same tensor — so attention starts as
  `1/3 * local + 2/3 * compressed`, and the compressed branch is mean-pooled, i.e. deliberately
  low-variance. The dense baseline has no gate and no such shrinkage. Every sparse-path model
  therefore begins training biased toward a smoothed global summary; that is a structural
  property of the placeholder, not a learned choice. Evidence: the 5.625 deg sparse control
  (`mosaic_5p625_sparse`) collapsed in two independent runs (best val 0.3805 @ ep4 and 0.3818 @
  ep2 vs the dense baseline's 0.2886 @ ep14) with spread_error_ratio ~0.001, and its checkpoint
  shows a learned weakening of the noise pathway (noise_bias/w13 RMS ratio 0.129 vs the dense
  0.442; NoiseGenerator `to_noise` norm 0.309 vs 0.766). NOTE the gate cannot suppress noise
  DIRECTLY — attention and the noise-carrying FFN are separate residual branches
  (`x = x + attn(norm1(x)); x = x + ffn(norm2(x), z)`) — so any effect is indirect, via `x3` or
  SiLU saturation in `w2(SiLU(x1+noise) * x3)`.

- **RoPE mean-pool bias (MIN-4).** The compressed logit equals the block-average of the fine
  logits (`q·mean(k)=mean(q·k)`), a defensible summary — but averaging *RoPE-rotated* keys
  interferes destructively for high-wavenumber components, so the global summary is biased toward
  low-wavenumber (large-scale) structure. Acceptable for a reference implementation; documented.

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

## f3 OUTCOME (2026-07-20) — resolution scaling did NOT beat the 5.625° baseline

Executed under the pre-registered protocol above. Both checkpoints scored with
`eval.common_grid={resolution_deg: 5.625}`; the 5.625° baseline's identity control reproduced
its native values to <5e-5 (t2m wk3 0.22392 vs 0.2239), confirming the comparison is sound.
Paired moving-block bootstrap on the CRPSS difference (`scripts/04_compare_runs.py`),
A = 5.625° `mosaic_fix45`, B = 1.5°:

| variable | lead | CRPSS 5.625° | CRPSS 1.5° | Δ (B−A) | 95% CI | verdict |
|---|---|---|---|---|---|---|
| t2m    | wk3 | 0.2239 | 0.1984 | −0.0255 | [−0.0444, −0.0083] | **1.5° significantly WORSE** |
| t2m    | wk4 | 0.2083 | 0.1928 | −0.0156 | [−0.0355, +0.0082] | indistinguishable |
| precip | wk3 | 0.0891 | 0.0903 | +0.0012 | [−0.0225, +0.0232] | indistinguishable |
| precip | wk4 | 0.0832 | 0.0469 | −0.0364 | [−0.0542, −0.0176] | **1.5° significantly WORSE** |

0/4 gate cells better, 2/4 significantly worse. **Per the conclusion rule above, resolution is
NOT credited as the ceiling.** Both models individually clear the CRPSS>0 gate against
probabilistic climatology; the 1.5° model simply does not improve on the existing baseline.

Note the native-grid comparison was misleading in the *opposite* direction to naive intuition:
the 1.5° model scores HIGHER on the common grid (t2m wk3 0.1675 native → 0.1984 common) because
its native target is intrinsically harder. Native-grid CRPSS understated it; either way the two
were never comparable, which is what MAJ-3 exists to prevent.

### Attribution is NOT established — do not write this up as "intrinsic predictability"

The Consequences bullet below says a null implies intrinsic S2S predictability is the limit.
**That inference is not supported by this evidence**, because two variables changed alongside
resolution:

1. **Attention structure (MAJ-1).** The 1.5° model runs local+compressed attention (each token
   sees 1/96 of the domain at full resolution, the rest at 64:1 compression); the 5.625°
   baseline runs *dense global attention in every layer*. These are different architectures.
2. **Ensemble calibration.** The 1.5° model is systematically under-dispersed — t2m
   spread-error ratio 0.77–0.84 across all leads vs 0.96–1.03 for the baseline — which inflates
   CRPS independently of resolution. precip wk4 is the extreme case (SER 0.639, CRPSS collapsing
   to 0.0469 between wk3=0.0903 and wk5=0.0771); that single-lead collapse looks anomalous
   rather than physical and warrants separate investigation.

The discriminating experiment is the MAJ-2 control: the 5.625° config run with the SAME
local+compressed approximation (`configs/model/mosaic_5p625_sparse.yaml`). If that control loses
comparable skill versus the dense baseline, the f3 null is attributable to attention capacity,
not resolution, and ADR-0007's headline conclusion must be rewritten accordingly.

### MAJ-2 control OUTCOME (2026-07-20) — the control FAILED; attribution remains UNRESOLVED

`configs/model/mosaic_5p625_sparse.yaml` holds resolution fixed at 5.625° and changes only the
attention path, to separate attention capacity from resolution. **It did not produce a valid
comparison.** Two independent runs, same seed/schedule as `mosaic_fix45`:

| run | geometry | best val | vs dense 0.2886 @ ep14 | SER | paired Δ vs dense |
|---|---|---|---|---|---|
| v1 | local 32, compress 64 → 48 summaries | 0.3805 @ ep4 | far worse | ~0.001 | −0.30…−0.36, 12/12 worse |
| v2 | local 32, compress 4 → 768 summaries (matches 1.5°) | 0.3818 @ ep2 | far worse | ~0.0004–0.0013 | −0.31…−0.37, 12/12 worse |

v1 used the wrong invariant (compressed-token COUNT, not compression ratio, sets global context:
48 vs the 1.5° model's 768). v2 corrected the geometry to match the 1.5° model on all three
invariants — local fraction 1/96, 768 compressed tokens, 1:8 compress:local — and produced
**numbers within noise of v1**. Geometry is therefore ruled out as the explanation.

**Why the control is invalid.** A valid control must reproduce the 1.5° model's behaviour at
fixed resolution. It does not: the deficit is ~10× the 1.5° model's (−0.30 vs −0.03) and the
failure is qualitatively different — ensemble fully collapsed (SER ~0.001) versus merely
under-dispersed (SER 0.77–0.84). The 1.5° model uses the SAME sparse path without collapsing,
so the sparse path per se is not the cause; the 5.625° instantiation falls into a distinct
broken optimisation regime. The control measures that regime, not the approximation under test.

**Supporting diagnostic** (`scripts/_diag_noise_path.py`, weights only, no GPU): the sparse
checkpoint shows a *learned* weakening of the ensemble-noise pathway — noise_bias/w13 RMS ratio
0.129 vs the dense baseline's 0.442 (29%), NoiseGenerator `to_noise` norm 0.309 vs 0.766 (40%).
A 3.4× weight reduction cannot by itself explain a ~1000× spread collapse, so the mechanism is
only partly identified. Note the strategy gate cannot suppress noise directly: attention and the
noise-carrying FFN are separate residual branches. See the 2:1 compressed-branch init bias
documented under "Attention path as actually run" for a structural contributor.

**Consequence for the thesis.** The f3 null is **NOT attributable to resolution**, and the
control designed to test the attention confound could not answer the question. This must be
written up as an open confound, not resolved in either direction. Diagnosing the collapse
further would not rescue the control: any fix would make it differ from the 1.5° model by
whatever was changed, reintroducing the confound it exists to remove. Deliberately stopped here
rather than buying further GPU runs for an unanswerable comparison.

### Residual limitation of the common-grid protocol (precip)

Coarsening equalises the GRID but not the ANOMALY DEFINITION: anomalies and the `log1p`
transform are computed at each store's native resolution *before* coarsening, and `log1p` does
not commute with area-averaging. The two runs' climatology references differ by 0.8% for t2m
(0.78005 vs 0.77382) but **11% for precip** (0.25281 vs 0.22730). The t2m comparison is sound;
the precip comparison carries this caveat. A fully clean protocol would coarsen in physical
space and recompute anomalies — deferred, and flagged here so it is not discovered downstream.

## Consequences

- ~14× grid points → smaller batch and likely gradient checkpointing; memory validated in f2.
- Poles enter the global input (India eval is far from poles; cos-lat weight → 0 at ±90 is fine).
- If Triton-vendored: CC-BY-NC attribution on `ops.py`; `triton` runtime dep.
- If 1.5° lifts CRPSS beyond the baseline CI **on the common grid** (see Pre-registration),
  resolution was the ceiling — the Phase-C result. If not, intrinsic S2S predictability at these
  scales is the limit, and the thesis pivots to consolidation/benchmarking (levers d/e).
