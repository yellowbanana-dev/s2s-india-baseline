# ADR-0004: Probabilistic (fair-CRPS) training for the Mosaic ensemble

**Status:** Proposed (Phase-B Stage B)
**Date:** 2026-07-02
**Supersedes the root-cause claim in:** ADR-0003 and the early ablation note
**Related:** ADR-0001 (Mosaic corner), ADR-0002 (vendoring), ADR-0003 (Stage-A fixes)

## Context: the diagnostic chain that led here

The Mosaic backbone lost to patch-ViT on deterministic mean skill, and a sequence of
hypotheses was raised and *falsified* before the real cause was found. The wrong turns
are recorded deliberately — the elimination is the scientific value.

1. **"Mosaic overfits (too big)."** 14.5 M vs 4.9 M. Falsified: a parameter-matched
   3.7 M slim variant, added dropout, and lower LR all still peaked at validation
   epoch 0.
2. **"Block-local receptive field misses teleconnections."** Stage A gave the encoder
   global attention (one block), plus RoPE and the native season embedding. This
   improved mean ACC (+~0.06 at wk4) but did **not** move the epoch-0 collapse.
3. **"The cSwiGLU noise FFN corrupts a deterministic objective."** Falsified: setting
   noise_dim=0 produced an identical epoch-0 curve.
4. **Optimization mismatch (CONFIRMED).** Mosaic zero-inits its residual branches
   (to_o, ffn.w2 at std=0.01), so epoch 0 is a near-identity map already at the
   climatology-ish MSE floor (val ~0.428). At lr=3e-4 the first optimizer step
   overshoots and destroys that initialization; every run with lr <= 3e-5 breaks the
   epoch-0 curse. At lr=3e-5 Mosaic trains normally and reaches patch-ViT val-MSE
   parity (0.4253 vs 0.4254).

**Residual finding after the fix:** at MSE parity, Mosaic still trails patch-ViT by
~4-5 ACC points at gate weeks. But this is on the *deterministic mean* axis, where a
global MSE-trained ViT is expected to win and where **both** models FAIL the honest
gate with spread-to-error ~= 0.004-0.01 (≈250x too little spread). Optimizing which
under-dispersed MSE model has marginally better mean ACC is optimizing a rounding
error relative to the actual problem.

## Decision

Train Mosaic **probabilistically** with a **fair (unbiased) CRPS** objective over an
ensemble drawn from its own noise mechanism, and evaluate on calibrated-spread metrics
— Mosaic's design axis and the thesis's actual contribution.

- **Loss:** latitude-weighted fair CRPS,
  `CRPS_fair = (1/M) Σ_i |x_i-y| - 1/(2 M (M-1)) Σ_{i,j} |x_i-x_j|`.
  The M(M-1) (Ferro 2014) normalisation is the *unbiased* spread estimator. The biased
  1/(2 M^2) form (kept in `eval.metrics` for scoring at large M) penalises spread and,
  minimised by SGD, would re-collapse the ensemble — the exact failure we are fixing.
  Verified numerically: the fair estimator recovers the analytic calibrated CRPS
  (~0.564) while the biased one inflates it (~0.62); calibrated ensembles score below
  collapsed ones only under the fair estimator.
- **Members:** M=8 per training step (fresh functional-perturbation vectors from
  `NoiseGenerator`, injected in every cSwiGLU FFN); M=16 at eval, drawn from the same
  internal mechanism (NOT the external P2 IC-perturbation wrapper).
- **Optimizer:** the confirmed lr=3e-5 per-model default, warmup=2, cosine, wd=0.1.

## Safeguard

Before trusting Stage-B India-box numbers, `scripts/healpix_recon_check.py` measures the
reconstruction error of the lon/lat <-> HEALPix round-trip, globally and over the India
box. A large India-box error would mean the mesh mapping distorts our eval region (a
candidate explanation for the residual mean-ACC gap) and would compromise the regional
CRPS — to be ruled out, not assumed.

**Correction (Fix 7/M1).** The interpolators (`CrossAttentionInterpolate`) are LEARNED,
not a fixed geometric operation, so the round-trip is weight-dependent. The earlier
figure quoted from an *untrained* run (~0.3 relative RMSE) was an artifact of random
initialisation gain, not interpolation loss, and is **retracted**. The diagnostic now
(a) removes the best least-squares gain/bias per region before computing RMSE, so the
residual reflects spatial distortion rather than an arbitrary scale, and (b) accepts
`eval.checkpoint=<path>` to load the trained interpolator weights and measure the
*trained* round-trip — the only version that bears on a trained model's regional skill.
The safeguard should be read off the gain/bias-corrected, trained-weights number.

## Consequences

- **Success criterion (honest):** spread-to-error ratio moving from ~0.004 toward 1,
  and CRPSS vs probabilistic climatology crossing 0 at weeks 3-4.
- If Stage B lifts calibration substantially, the Mosaic bet is vindicated on its design
  axis regardless of the deterministic mean gap. If it does not, that is itself the
  decisive evidence on whether to keep Mosaic for the heavier lifting ahead.
- patch-ViT retains the deterministic MSE path unchanged and remains the mean-skill
  reference; the fair-CRPS path is available to it too (via a uniform member interface),
  though it has no stochastic mechanism to disperse.
