# ADR-0010: Real fine top-k selection branch (per query-block), in portable PyTorch

**Status:** Proposed (Phase C, lever f) — PRE-REGISTERED, no results yet
**Date:** 2026-07-21
**Refs:** ADR-0002 (ops.py never vendored), ADR-0007 (MAJ-1 placeholder, f3 OUTCOME),
ADR-0009 (two-way gate — the alternative fix)

## Correction of an earlier claim

ADR-0007 records that the fine top-k selection branch is "memory-infeasible in pure PyTorch at
this scale." **That is wrong, and this ADR corrects it.** It is true only for **per-TOKEN**
selection — which is what the vendored `attn_topk` computes, and which needs a
`(b, seq, h, k*bs, d)` gather (~2e11 elements at nside=64). NSA selects **per QUERY-BLOCK**:
all queries in a block share one selected key-block set, so the branch is processed one query
block at a time. At nside=64 (seq=49152, query block 512, k=16, sparse block 64):

| branch | score-pairs | verdict |
|---|---|---|
| dense global | 2.42e9 | infeasible |
| **selection (per query-block)** | **5.03e7** | **48x cheaper than dense** |
| compressed (already running) | 3.77e7 | comparable |

Peak transient is ~0.27 GB per query block (and SDPA never materialises the scores). So the
branch is affordable **without Triton**, and the "obtain ops.py" open question in ADR-0007 is
no longer blocking.

## Decision

Implement `selection_attention()` in `primitives.py` and gate it behind `selection` (default
**False**, so the legacy placeholder path stays byte-identical and every existing checkpoint —
including the gate2 run training now — remains loadable). Config:
`configs/model/mosaic_15deg_sel.yaml`, verified to differ from `mosaic_15deg.yaml` in
`selection` **only**.

**Relationship to ADR-0009.** Both fix the same MAJ-1 defect (`o_slc = o_cmp`, which starts the
gate 2:1 biased toward the low-variance mean-pooled branch), by opposite routes:
- ADR-0009 `gate_slots=2` **deletes** the fake slot (cheap; tests whether the bias mattered).
- ADR-0010 `selection=true` **makes it real** (adds capability; restores the actual Mosaic
  attention). Requires `gate_slots=3`; combining the two is rejected at construction.

They are complementary, not redundant: gate2 isolates the *bias*, selection tests the *capability*.

## Pre-registration (before any result exists)

**Hypothesis.** The local+compressed approximation caps skill because every token's only
full-resolution view is its own 512-pixel block; restoring fine attention to the top-k most
relevant distant blocks should recover skill and/or spread.

**Primary readout** — common-grid CRPSS vs the 5.625 deg `mosaic_fix45` baseline
(`04_compare_runs.py`), with the current 1.5 deg model as the reference point
(t2m wk3/4 -0.0255/-0.0155, precip wk3/4 +0.0012/-0.0364):
- **CONFIRMS** capability-limitation if the as-forecast deficit **narrows at >= 2 of 4 gate
  cells** with the paired CI separated from the current model's delta.
- **DISCONFIRMS** if the deficit is unchanged or worse at all four cells — the local+compressed
  approximation was then NOT the binding constraint, and ADR-0007's attention confound is
  correspondingly weakened.

**Secondary** — trained t2m SER at wk3/4 (currently 0.811/0.772). If selection also lifts SER
toward ~0.9 it supports the ADR-0009 chain (gate bias -> low-variance features ->
under-dispersion) via the "make the slot real" route.

**VOID condition.** As in ADR-0009: if the run collapses into the known broken sparse regime
(best-val at epoch <= 4, SER ~0.001), the experiment says nothing and is discarded.

**Not claimed.** This is not the upstream Triton kernel and is not bit-compatible with it;
fidelity to `ops.py` remains unverifiable (the file was never obtained). It is a faithful
implementation of the *documented* NSA selection strategy, not of that specific kernel.

## Consequences

- `selection` defaults False: no existing config, checkpoint or reproduction changes.
- Cost: one 1.5 deg training run + eval. Expect a wall-clock increase from the 96-iteration
  query-block loop (compute is ~48x below dense global, but it is a Python loop, not a kernel).
- Gate shape is unchanged (still `3 * q_heads`), so unlike ADR-0009 this is not a shape change —
  but the trained gate means something different, so it is still a new experiment, not a
  checkpoint upgrade.
