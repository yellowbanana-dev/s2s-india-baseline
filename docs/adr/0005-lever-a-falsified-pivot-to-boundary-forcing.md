# ADR-0005: Lever (a) falsified — HEALPix round-trip is not the India-box bottleneck; pivot to boundary forcing (lever b)

**Status:** Accepted (Phase C)
**Date:** 2026-07-06
**Refs:** ADR-0002 (vendoring, nside=16), ADR-0004 (Stage-B safeguard `healpix_recon_check.py`),
`scripts/healpix_recon_decomp.py`, `results/phase_b_stage_b_multiseed.md`

## Context

Phase C opened on lever (a): the ADR-0004 safeguard `healpix_recon_check.py` reported
~0.307 India-box relative RMSE for the lon/lat ↔ HEALPix round-trip, flagged as
"unrealized headroom." That check measures **untrained** `CrossAttentionInterpolate`
modules, so the number confounds mesh geometry with random projection weights. We built
`scripts/healpix_recon_decomp.py` to decompose it into three references: a weight-free
inverse-distance floor (`geom_idw`, pure geometry on the same haversine k-NN graph), the
untrained attention (`attn_untrained`, reproduces the check), and the trained attention
(`attn_trained`, Stage-B seed-0 checkpoint `epoch=13-val_loss=0.2728`).

## Evidence (seed-0 Stage-B, commit cf99451, real datamodule grid)

India-box relative RMSE (smooth planetary-wave fields; white noise in parens):

| reference | India-box rel RMSE | reading |
|-----------|--------------------|---------|
| `geom_idw` (nside=16) | 0.007–0.029 (noise 0.131) | mesh round-trips near-losslessly |
| `geom_idw` (nside=32) | 0.002–0.009 (noise 0.049) | finer mesh barely needed |
| `attn_untrained` | 0.29–0.36 | reproduces the original 0.307 → harness validated |
| `attn_trained` | 0.27–0.33 | training barely moved it |

- The geometric floor is near-zero → the nside=16 HEALPix mesh is **not** the bottleneck.
- `attn_untrained` reproduces 0.307 → the decomposition harness matches the original check.
- `attn_trained ≈ attn_untrained` → the forecasting objective did not drive the
  interpolator-pair identity error down.

## The confound (recorded honestly)

`attn_trained` composes `interp_to_hp → interp_to_ll` back-to-back, which **never happens
in the real forward pass**: the full U-Net sits between them, and `interp_to_ll` is trained
on *decoder* features, not on `interp_to_hp`'s raw output. So `attn_trained ≈ attn_untrained`
does **not** prove the deployed model loses ~0.30 of the India signal; it shows only that the
forecasting objective never optimized this particular composition toward identity (expected —
the U-Net can absorb the representation). The script's original decision-rule comment
overstated `attn_trained` as "the loss the deployed model incurs"; corrected in-source.

## Decision

**Abandon lever (a).** The mesh geometry is demonstrably adequate (IDW floor ~1–3% for
smooth fields, and nside=32 offers little), and the round-trip metric is not a valid measure
of deployed information loss. Do **not** scale nside or re-engineer the interpolation on the
basis of the 0.307.

**Pivot Phase C to lever (b):** slow boundary forcing — SST anomalies as an evolving boundary
condition — plus explicit season and lead-time conditioning, the true S2S signal source.

## Consequences

- The Mosaic ↔ patch-ViT mean-ACC gap remains unexplained but is **not** attributable to the
  mesh. If a future interventional test is wanted, the clean falsifiable design is an IDW
  warm-start / skip on the interpolators followed by a retrain — recorded here, not pursued.
- `scripts/healpix_recon_decomp.py`, `src/s2s/eval/healpix_recon.py`, and
  `tests/test_healpix_recon.py` stay in the tree as the reusable geometric-floor diagnostic
  and its 6 guardrail tests.
- Lever (b)'s concrete design (SST representation, season/lead conditioning, staging, and
  the leakage guard for any SST-derived indices) lands as **ADR-0006** before major code.
- One cheap diagnostic (no training run) closed lever (a) — the intended build-fast/fail-fast.
