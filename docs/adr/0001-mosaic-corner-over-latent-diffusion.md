# ADR 0001: Mosaic-corner IC perturbation over latent-diffusion ensembling (P2)

## Status

Accepted (Phase A).

## Context

The project's locked decision (README, axis 2) is **P2 — a cheap ensemble**:
multi-seed training plus initial-condition (IC) perturbation, rather than a
learned generative ensemble. The open question this ADR resolves is *how* to
perturb the initial condition cheaply.

Two candidate approaches were considered for producing the per-member input
perturbation that feeds `cfg.train.ensemble.ic_perturbation_std`:

1. **Mosaic-corner perturbation** -- add i.i.d. Gaussian noise (`ic_perturbation_std`)
   to spatial tiles ("corners") of the global input field, with tile placement
   varied per ensemble member. Cheap: a few lines of tensor noising, no extra
   model, no extra training run, fully deterministic given a seed.
2. **Latent-diffusion ensembling** -- train a diffusion model over a learned
   latent representation of the input state, and sample IC perturbations (or
   full stochastic forecasts) from it. This is the standard approach in recent
   generative weather models (e.g. GenCast-style architectures) and produces
   physically-structured, spatially-correlated perturbations rather than
   tile-wise noise.

## Decision

Use **mosaic-corner perturbation** for Phase A's P2 ensemble.

## Rationale

Phase A's stated job (README: "Locked decisions") is **infrastructure + a
number to beat, not a good model**. Every choice in this phase is evaluated
against that job, not against asymptotic forecast quality:

- **Cost.** A diffusion ensemble requires training and validating a second
  model (the diffusion prior) before the *baseline* model can even produce a
  scorable ensemble. That is a second research project nested inside the
  baseline. Mosaic-corner perturbation costs nothing beyond the already-planned
  multi-seed training loop.
- **Reproducibility.** One config + one commit + one run (README, "Stack")
  is much easier to guarantee for tile-noise (fully determined by
  `cfg.train.ensemble.seeds` and `ic_perturbation_std`) than for a sampling
  procedure from a separately-trained generative model with its own seeds,
  checkpoints, and training dynamics.
- **Debuggability.** If the P2 ensemble's CRPS/reliability looks wrong, a
  tile-noise perturbation has exactly two knobs (`std`, tile layout) to
  audit by hand. A latent-diffusion ensemble adds an entire extra failure
  surface (latent encoder quality, diffusion sampling steps, prior
  calibration) that is very hard to separate from "is the baseline pipeline
  correct" -- the question Phase A exists to answer.
- **Decision gate is about the pipeline, not ensemble realism.** The Phase A
  gate (configs/eval/default.yaml: beat climatology *and* persistence at
  week 3-4 CRPS) only needs *some* honest spread, not a meteorologically
  faithful one. Tile noise is sufficient to produce a non-degenerate
  ensemble for that gate.

## Consequences

- The P2 ensemble's spread will plausibly be **less physically structured**
  than a diffusion-based ensemble's -- it perturbs input tiles independently
  rather than respecting the input field's spatial covariance. This may show
  up as overconfident or miscalibrated `reliability()` diagrams later; that is
  an expected, accepted limitation of a Phase A baseline, not a hidden bug.
- This decision is **revisited in Phase B**, alongside the switch to
  leave-one-year-out CV (README, "Train/val/test split" section). If the
  baseline clears its decision gate but later phases need a sharper,
  better-calibrated ensemble, latent-diffusion ensembling is the natural next
  candidate -- at that point the cost is justified because the underlying
  pipeline (data, splits, metrics, baselines) is already verified correct by
  Phase A.
- No code in this repo currently implements either option's training step
  (`src/s2s/train.py` is still a stub); this ADR fixes the design choice that
  `train.py`'s ensemble construction must follow when implemented.
