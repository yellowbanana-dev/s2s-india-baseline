# ADR-0009: Two-way strategy gate — removing the MAJ-1 duplicate-slot init bias

**Status:** Proposed (Phase C, lever f follow-up) — PRE-REGISTERED, no results yet
**Date:** 2026-07-21
**Refs:** ADR-0007 (MAJ-1 gate degeneracy; spread-calibration attribution OUTCOME; Step A)

## Context

The f3 deficit is attributable to ensemble **under-dispersion**, not resolution: spread-calibrated,
the gate cells move from 2/4 significantly worse to **4/4 indistinguishable** (ADR-0007 OUTCOME).
Step A then localised the defect — the 1.5 deg model's noise pathway is **structurally intact**
(sparse/dense RMS 0.7666, versus the collapsed 5.625 deg control's ~0.29 relative), so the spread
loss is NOT a dead NoiseGenerator; it happens **downstream**, in the gating/SiLU path of
`w2(SiLU(x1 + noise_bias(z)) * x3)`.

The MAJ-1 placeholder is the obvious structural suspect. Because `o_slc = o_cmp`, the 3-way softmax
starts the model at **1/3 local + 2/3 compressed**, and the compressed branch is mean-pooled, i.e.
deliberately low-variance. Gate bias -> smoothed low-variance features -> suppressed effective
noise -> under-dispersion is a coherent chain consistent with every measurement so far. It is
**suggestive, not proven**: nothing yet links gate weights to spread directly.

## Decision

Add `gate_slots` (default **3** = unchanged, so every existing checkpoint still loads) and run the
1.5 deg model with **`gate_slots=2`**: the duplicate selection slot is removed, so the gate is a
genuine local-vs-compressed choice starting **unbiased at 1/2 - 1/2**. Config:
`configs/model/mosaic_15deg_gate2.yaml`, verified to differ from `mosaic_15deg.yaml` in
`gate_slots` **only** (single-variable experiment).

This changes the gate layer's shape, so it is a **NEW experiment requiring a retrain**, not a
checkpoint upgrade — consistent with ADR-0007's finding that these gates are not transferable.

## Pre-registration (before any result exists)

**Hypothesis.** The 2:1 init bias toward the low-variance compressed branch is the mechanism
producing the 1.5 deg model's under-dispersion. Removing it should raise trained spread.

**Primary readout — trained t2m spread-error ratio at gate leads (wk3/wk4), currently 0.811/0.772
(dense baseline: 0.978/0.997):**
- **CONFIRMS** the mechanism if SER >= **0.90** at both gate leads.
- **DISCONFIRMS** it if SER <= **0.85** at both. The gate bias is then NOT the cause, the
  under-dispersion originates elsewhere, and the attention -> calibration chain asserted in
  ADR-0007 is materially weakened — which must be written up as such.
- Between 0.85 and 0.90: partial / inconclusive; no headline claim either way.

**Secondary.** The gap between calibrated and as-forecast CRPSS should SHRINK — the model should
realise at training time what post-hoc rescaling supplied. Currently |crpss_cal - crpss| at the
gate cells is 0.015-0.042 for the 1.5 deg model versus 0.001-0.006 for the baseline; it should move
toward the baseline's range. The as-forecast common-grid deficit vs `mosaic_fix45` should narrow.

**VOID condition (check first).** The run must train stably — best-val at a sensible epoch and a
CRPSS > 0 gate pass. If it collapses like the 5.625 deg sparse control (SER ~0.001, best-val at
epoch <= 4), the experiment is **VOID**: that is the known broken optimisation regime documented
in ADR-0007, and says nothing about the gate hypothesis.

**Not claimed.** Even a confirming result would not show 1.5 deg BEATS 5.625 deg — only that the
under-dispersion is fixable at training time, letting the resolution question finally be judged on
a calibrated model.

## Consequences

- `gate_slots` defaults to 3: all existing configs, checkpoints and reproductions are unaffected.
- One 1.5 deg training run (~1 day) + eval. Evaluate with `eval.spread_calibration.enabled=true`
  so the calibrated-vs-as-forecast gap (the secondary readout) is measurable in the same run.
- If confirmed, the honest Phase-C narrative becomes: resolution is not the ceiling, and the
  observed deficit traced to a fixable implementation artifact of the deferred-selection
  placeholder — not to intrinsic predictability and not to resolution.
