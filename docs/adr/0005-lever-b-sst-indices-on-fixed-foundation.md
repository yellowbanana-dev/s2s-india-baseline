# ADR-0005: Phase-C lever b — SST teleconnection indices, rebuilt on the fixed foundation

**Status:** Accepted (Phase C)
**Date:** 2026-07-08
**Foundation:** `integration/all-fixes` (M1–M5, C1–C3)
**Refs:** ADR-0004 + `src/s2s/eval/recon.py` (M1, HEALPix diagnostic), `src/s2s/data/assemble.py`

## Context

Phase-C lever (a) — "fix the HEALPix interpolation to realise India-box headroom" — is
**closed**. The ~0.307 round-trip figure was an artifact of measuring *untrained*
`CrossAttentionInterpolate` weights (random-init gain), not interpolation geometry. This was
reached independently twice: an inverse-distance geometric-floor decomposition, and the M1 fix
(`src/s2s/eval/recon.py`: gain/bias `affine_correct` + a trained-weights mode), which retracts
the 0.3 figure in the ADR-0004 safeguard. The mesh is not the bottleneck; no mesh/interpolation
rework is pursued.

The remaining boundary-forcing lever is **SST**. An earlier increment fed raw SST *longer
history* (extra week-lags); those numbers were produced on a pre-fix pipeline and are **retired**
(they predate the M2 precip-units fix, the M4 train/test lead-alignment fix, and the C1
trend-aware climatology reference), so no finding from them is carried as established here. The
season fix (doy_sin + atan2) that increment also introduced is now the foundation's M5 — kept,
not re-authored.

## Decision

Pursue lever (b) as **SST teleconnection indices** — the physically-motivated, low-dimensional
India-monsoon drivers — rather than the raw SST field:

- `nino34` (ENSO): cos-lat-weighted SST-anomaly mean over Niño 3.4 (5S–5N, 170W–120W).
- `dmi` (IOD Dipole Mode Index): West box (10S–10N, 50–70E) minus East box (10S–0, 90–110E).
- Fed as globally-broadcast channels placed after the history stack and before the seasonal
  pair; `pack_windows` remains the single channel-layout authority. Config:
  `data.sst_indices: [nino34, dmi]`, `data.sst_index_lags_weeks: [0]` (init-time; extendable,
  e.g. `[0, 8]`, for index tendency). Default `in_channels = 12 + 2 + 2 = 16`.

## Leakage & caveats

- Indices are computed from the pipeline's **train-standardized** SST anomaly field, so they
  inherit the train-only normalisation guard. The valid-init lower bound and the daily-window
  leakage assertion extend to the deepest index lag.
- Being box means of per-cell standardized anomalies, these are a **monotone proxy** for the
  conventional raw-anomaly Niño3.4/DMI, not the operational indices verbatim — enough to test
  whether an explicit ENSO/IOD signal helps, documented so numbers aren't over-read.

## Plan & gate

1. Re-establish a clean Stage-B Mosaic fair-CRPS **baseline** on the fixed foundation (M2/M4
   data, C1 trend-aware `crpss_vs_prob`, C2 bootstrap CIs). All prior baseline numbers are
   retired; this becomes the reference.
2. Train + eval the indices config (seed-0), compare against the fresh baseline.
3. Gate (unchanged in spirit): beat `crpss_vs_prob` at wk3/4 — with the C2 bootstrap CI as the
   significance check — while keeping t2m SER near 1.0; then confirm on 3 seeds. Watch **precip**
   especially (ENSO/IOD are precip drivers; and precip skill itself changes under the M2 fix).
   If indices don't lift CRPSS on the fixed foundation, boundary forcing is exhausted at 5.625°
   and the next fork is precip-specific treatment or lead conditioning (ADR-0006).
