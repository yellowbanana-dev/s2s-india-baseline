# ADR-0007: SST longer-history is a null lever; keep the season fix, pivot to SST indices

**Status:** Accepted (Phase C, lever b increment 2)
**Date:** 2026-07-07
**Refs:** ADR-0006 (increment 1 design), `src/s2s/data/assemble.py`, four-way seed-0 eval

## Context

Increment 1 (ADR-0006) bundled two changes — raw SST longer-history channels and a season
conditioning fix (doy_sin + atan2). Seed-0 honest eval over four configs isolated their effects
(India box, test split, wk3/wk4):

| config | t2m CRPSS | precip CRPSS | t2m SER |
|--------|-----------|--------------|---------|
| Stage-B baseline | 0.204 / 0.215 | 0.194 / 0.185 | 0.879 / 0.922 |
| SST hist [4,8,12] | 0.230 / 0.199 | 0.169 / 0.183 | 0.958 / 0.998 |
| SST hist [4,8]    | 0.219 / 0.212 | 0.182 / 0.178 | 1.019 / 0.990 |
| season-only ([])  | 0.214 / 0.205 | 0.188 / 0.182 | 0.909 / 0.953 |

Best-val (fair-CRPS) was flat across all runs (0.2728 / 0.2734 / 0.2731), and the model
checkpoints on best-val, so post-peak overfitting did not degrade the evaluated weights.

## Findings

- **Raw SST longer-history is a null skill lever.** It does not lift t2m CRPSS beyond seed
  noise and it *degrades* precip CRPSS (more lags → worse), at both depths. The SER move
  toward 1.0 is a mechanical spread inflation (more collinear input variance → more ensemble
  dispersion), not new skill — and it costs precip.
- **The season fix alone is nearly a no-op on skill** (season-only sits between baseline and
  the SST runs on every metric), but it is a genuine *correctness* improvement: doy_sin +
  atan2 removes the spring/fall ambiguity of the ADR-0003 arccos path, at zero skill cost and
  a small SER gain. Worth keeping.

## Decision

1. **Disable raw SST-history by default** (`data.sst_history_lags_weeks: []`). The machinery
   stays in the code, configurable, but off — it is not a skill lever.
2. **Keep the season fix** (doy_sin + atan2), now the increment-1 keeper.
3. **Pivot to SST teleconnection indices** (increment 2): inject the known India-monsoon S2S
   drivers directly as low-dimensional, globally-broadcast channels instead of a raw collinear
   SST field:
   - `nino34` — ENSO, area-weighted SST-anomaly mean over Niño 3.4 (5S–5N, 170W–120W).
   - `dmi` — IOD Dipole Mode Index = West box (10S–10N, 50–70E) minus East box (10S–0, 90–110E).
   - Config: `data.sst_indices: [nino34, dmi]`, `data.sst_index_lags_weeks: [0]` (init-time;
     extendable, e.g. `[0, 8]`, to add index tendency). Channels sit after the history/raw-SST
     block and before the seasonal pair; `pack_windows` remains the layout authority.

## Leakage & caveats

- Indices are computed from the pipeline's **train-standardized** SST anomaly field (the same
  field already normalized train-only), so they inherit the leakage guard. The valid-index
  lower bound extends to the deepest index lag; daily-window leakage assertions cover it.
- Because the source is per-cell standardized anomalies, these are a **monotone proxy** for the
  conventional raw-anomaly Niño3.4/DMI, not the operational indices verbatim — sufficient to
  test whether an explicit ENSO/IOD signal helps; documented so numbers aren't over-read.

## Consequences

- Default `in_channels` becomes `12 (history) + 2 (nino34, dmi) + 2 (seasonal) = 16`.
- Touch-points: `assemble.py` (index helpers + wiring), `windows.py` (daily builder),
  `configs/data/era5_india.yaml`; new guardrail test (index channels + ENSO tracking).
- **Gate (unchanged):** seed-0 vs the Stage-B baseline — beat CRPSS_vs_prob at wk3/4 while
  keeping t2m SER near 1.0 — then confirm on 3 seeds. If indices don't lift CRPSS either, the
  boundary-forcing lever is exhausted at 5.625° and the next fork is precip-specific treatment
  or lead-time conditioning (ADR-0008).
