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
  CIs) over the India box; compare to the 5.625° fixed baseline (`mosaic_fix45`).

## Open question (blocks f1)

The upstream Mosaic `ops.py` (Triton block-sparse kernel) was never vendored, and an attempt to
fetch it upstream timed out. Two ways to source the sparse path:
1. **Vendor upstream `ops.py`** (Triton) — faithful to the paper/kernel, but adds a Triton/CUDA
   build dependency and a large validation surface. Needs the upstream file.
2. **PyTorch reference implementation** of the same block-sparse strategy — portable (no Triton
   build dep), more reproducible for a thesis, slower; needs the algorithm spec to match faithfully.
Resolved with the human before f1 (which path, and whether the upstream `ops.py` can be supplied).

## Consequences

- ~14× grid points → smaller batch and likely gradient checkpointing; memory validated in f2.
- Poles enter the global input (India eval is far from poles; cos-lat weight → 0 at ±90 is fine).
- If Triton-vendored: CC-BY-NC attribution on `ops.py`; `triton` runtime dep.
- If 1.5° lifts CRPSS beyond the baseline CI, resolution was the ceiling — the Phase-C result.
  If not, intrinsic S2S predictability at these scales is the limit, and the thesis pivots to
  consolidation/benchmarking (levers d/e).
