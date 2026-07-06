# ADR-0006: Lever (b) increment 1 — SST longer-history boundary forcing + season conditioning

**Status:** Accepted (Phase C)
**Date:** 2026-07-06
**Refs:** ADR-0003 (season embedding caveat), ADR-0005 (pivot to lever b),
`src/s2s/data/assemble.py` (pack_windows = channel authority), `configs/data/era5_india.yaml`

## Context

ADR-0005 pivoted Phase C to lever (b): slow boundary forcing + explicit season/lead
conditioning. Current input state:

- **SST** enters only as init-time weekly-mean anomaly inside the shared 2-week history
  (`history_weeks=2`), one channel per history week. No forward boundary, no tendency.
- **Season** is a single `doy_cos` channel; Mosaic additionally recovers a day fraction via
  `arccos(doy_cos)` (range [0, 0.5] → spring/fall ambiguous) with year=0 (ADR-0003 caveat).
- **Leads** 1–6 are joint output channels; no explicit lead-time conditioning.

Bar (from the session): CRPSS_vs_prob at wk3–4 must clear the Stage-B PASS (t2m ≈ 0.20 /
precip ≈ 0.19) **and** t2m SER stay near 1.0; iterate on seed 0 vs the fixed 2018–2023 test
split, then confirm on 3 seeds.

## Decision

Increment 1 = **season fix + SST longer raw history**, keeping the joint output head.
Lead-time conditioning (b3) is deferred to a later increment/ADR.

### Season conditioning
- Add a `doy_sin` input channel alongside `doy_cos`.
- In `mosaic_backbone.forward`, recover the true day fraction via
  `atan2(doy_sin, doy_cos) / 2π (mod 1)` instead of `arccos` — removes the spring/fall
  ambiguity and feeds Mosaic's `time_embedding` the correct angle.
- **Year is deferred.** Test years (2018–23) lie outside train years (1979–2012); a raw year
  signal would force extrapolation and risks fitting the warming trend unsafely. The ADR-0003
  year=0 gap is acknowledged, not closed here.

### SST longer history (boundary forcing)
- Add SST weekly-mean anomaly channels at **extra lags `[4, 8, 12]` weeks** before init,
  beyond the 2-week shared history (config `data.sst_history_lags_weeks`).
- **Spaced, not contiguous, on purpose.** Consecutive weekly SST anomalies are near-collinear
  (SST varies slowly); dyadic-ish lags capture SST *tendency/evolution* over ~1–3 months with
  only 3 extra channels, limiting the overfitting exposure that raw longer history invites at
  5.625° with the current sample size. The lag list is a config knob.
- **Leakage-safe by construction.** The extra channels are sliced from the same SST field that
  is standardized with TRAIN-only mean/std before any windowing; no new normalizer stats. The
  valid-init lower bound is extended to the deepest SST lag, and the daily-window leakage
  assertions are extended to cover it.

### Channel-layout authority
`pack_windows` remains the single source of truth. New layout:
`[history_weeks × n_in_vars] + [SST extra-lag channels] + [doy_cos, doy_sin]`.
`in_out_channels()` counts it; both `assemble_arrays` (test/W-MON) and
`daily_init_weekly_windows` (train/val) produce identical layout.

## Consequences

- `in_channels` grows by `len(sst_history_lags_weeks) + 1` (the +1 = new doy_sin; doy_cos
  already counted). Model build is generic (`in_out_channels` → `lit`), so patch-ViT and
  Mosaic pick up the new width with no architecture change; only `mosaic_backbone`'s season
  recovery is edited.
- Touch-points: `configs/data/era5_india.yaml`, `assemble.py`, `windows.py`,
  `mosaic_backbone.py`; tests `test_assemble`, `test_mosaic_backbone`, `test_phase_b_sampling`
  updated, plus new guardrails (doy_sin/atan2 recovery; SST-lag layout parity between the two
  builders; extended leakage bound). Suite stays green.
- **NOTE (executor):** increment 1 changes the input tensor width, so
  `scripts/01_build_dataset.py` must be **re-run** to rebuild `daily_anom.zarr`? No — the zarr
  holds raw daily anomalies per variable; only the *assembly* changed. Re-running the build is
  unnecessary, but a fresh training run IS required (old checkpoints have the old input width).
- If increment 1 clears the seed-0 gate and confirms on 3 seeds, it is the first lever-(b) win.
  Lead-time conditioning (b3) would then be ADR-0007 if pursued.
