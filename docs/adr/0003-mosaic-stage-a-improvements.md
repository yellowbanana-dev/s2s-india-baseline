# ADR-0003 — Mosaic Stage-A Improvements: Global Attention, RoPE, Native Season Embedding

**Date:** 2026-06-29  
**Status:** Accepted  
**Refs:** ADR-0001 (IC perturbation), ADR-0002 (vendoring), `results/phase_b_ablation_note.md`

---

## Context

Phase-B ablation (seed 0, full record 1979–2012, daily-stride, MSE loss) showed Mosaic losing
to patch-ViT on mean t2m ACC/RMSE at all lead weeks:

| Model | wk3 ACC | wk4 ACC | Best val epoch |
|-------|---------|---------|---------------|
| Mosaic 14.5M | 0.369 | 0.326 | 0 (severe overfitting) |
| Mosaic slim+dropout | 0.408 | 0.399 | 0 |
| **patch-ViT 4.9M** | **0.437** | **0.413** | **3** |

Three root-cause deficits were identified (architectural, not data):

1. **Local receptive field at encoder/decoder.** `block_attn_size=512` splits 3072 encoder tokens
   into 6 independent blocks — each block sees only ~1/6 of the globe. The bottleneck (nside=8,
   768 tokens, 1 block) is the only stage with global context. S2S anomaly forecasting requires
   teleconnection signals (e.g. MJO, ENSO precursors) that span hemispheres; a 6-block encoder
   cannot capture them.

2. **RoPE off.** The original config relied on XYZ static coordinates appended by
   `initialize_static_vars` for positional encoding. RoPE provides explicit relative-position
   bias inside every attention head and should improve the model's ability to learn spatial
   structure on the HEALPix grid.

3. **Zeros fed as time embedding.** `mosaic_backbone.py` passed `day_year_time=zeros` on the
   grounds that doy_cos is already in C_in[-1]. However, the Transformer's `time_embedding` path
   (sin/cos of day and year, broadcast over all tokens) is a separate conditioning signal that
   the model can use to modulate its forecast by season. Feeding zeros disables this pathway.

---

## Decision

Apply three Stage-A fixes to the Mosaic config and adapter. Keep training deterministic (MSE,
num_noise_samples=1). Stage-B (CRPS/noise, multi-seed ensemble, SER improvement) follows if
Stage-A shows improved validation trajectory.

### Fix 1 — Global encoder attention

Set `block_attn_size: 3072` in `configs/model/mosaic.yaml`. For nside=16 (seq_len=3072), this
produces exactly 1 block → single global softmax at every encoder and decoder layer. The
implementation in `primitives.py:block_attention()` handles this correctly: the einops reshape
`(nb bs)` with `bs=seq_len` gives `nb=1` and a standard full-sequence attention.

**Config knob retained** — `block_attn_size` is still a config parameter; the old 6-block local
attention is recovered by setting `block_attn_size: 512`.

Memory note: at 3072 tokens, dense self-attention is O(3072²) ≈ 9.4M element score matrix per
head per layer — well within H100 NVL SRAM limits.

### Fix 2 — RoPE

Set `rope: true` in `configs/model/mosaic.yaml`. The vendored `RoPE.initialize_rope()` reads
HEALPix (lon, lat) coordinates (via `utils.get_healpix_grid(nside)`) and registers `cos_freqs`
and `sin_freqs` buffers for each stage. Applied inside `MosaicAttention.forward()` to Q and K
before the attention call. No code changes required in `mosaic_backbone.py`.

### Fix 3 — Native season embedding

In `mosaic_backbone.py:forward()`, derive `day_normalized` from the `doy_cos` input channel
instead of passing zeros:

```
doy_cos = x[:, -1, 0, 0]               # broadcast across spatial dims; scalar per sample
day_normalized = arccos(doy_cos) / 2π  # ∈ [0, 0.5]
day_year_time[:, 0, 0] = day_normalized
```

The Transformer's `time_embedding` then computes:
- `cos(2π · day_normalized) = doy_cos`  (exact recovery)
- `sin(2π · day_normalized) = |doy_sin|` (magnitude; sign lost because arccos maps to [0,π])

Year component is left at 0 (batch contains no year information without threading from the
datamodule, which is deferred to Stage-B). This still gives a real, non-zero seasonal signal vs
the previous all-zeros baseline.

---

## Consequences

**Positive:**
- Encoder sees the full globe at every layer → can learn teleconnections
- RoPE provides per-head explicit positional bias on the HEALPix grid
- Time embedding is non-zero → model can condition on season (partial; year=0)
- `block_attn_size` remains a config knob → global-vs-local is ablatable

**Neutral / trade-offs:**
- Increased FLOPs per encoder layer (O(3072²) vs O(512²) × 6 ≈ 6× higher). Wall-clock per step
  roughly doubles on H100; acceptable for research.
- arccos-derived `day_normalized` ∈ [0, 0.5] cannot distinguish spring from fall (symmetric
  cosine). Fix: thread actual timestamps through the datamodule (deferred to Stage-B or beyond).

**Stage-B next steps:**
- If Stage-A best val epoch > 0 (indicating the fix resolved the epoch-0 overfitting), proceed
  to probabilistic training (CRPS loss, `num_noise_samples > 1`, multi-seed ensemble).
- Add year normalization by threading timestamps from the datamodule.

---

## Addendum (2026-07-01): Stage-A epoch-0 collapse was NOT caused by these fixes — it was an LR mismatch

**Status: Correction to prior context section**

After Stage-A training completed (best epoch=0, val=0.4278), a further diagnostic sweep
(branch `mosaic-optim-sweep`, SHA ff5573c) established that the epoch-0 collapse was caused by
an **LR/zero-init-residual mismatch**, not by the architectural deficits listed above.

**Root cause:** Mosaic zero-inits residual projections (`to_o` and `ffn.w2`, std=0.01). This
makes the model a near-identity at initialization (val~0.428 at epoch 0). lr=3e-4 overshoots
this in a single update, permanently destroying the good initial state. lr≤3e-5 lets the model
learn past the init floor.

**Evidence:**
- Noise-FFN diagnostic (`noise_dim=0`, plain SwiGLU): same epoch-0 collapse → noise not the cause
- Optimizer sweep R3 (lr=3e-5, warmup=2, wd=0.1): best ep=3, val=0.4252 — epoch-0 curse broken
- Full 50-epoch run at lr=3e-5: best ep=3, val=0.4253 — model trains normally

**Implication for this ADR:** The three Stage-A fixes (global attention, RoPE, season embedding)
are still valid improvements. They were correctly applied. However, their true effect was masked
by the LR mismatch — Stage-A's "epoch-0 best" was entirely due to lr=3e-4, not the architecture.

With corrected lr=3e-5, added to `configs/model/mosaic.yaml` as a per-model override, Mosaic
trains to ep3 best val=0.4253, matching patch-ViT (0.4254). Downstream t2m ACC at wk3/4 is
0.394/0.391 vs patch-ViT 0.437/0.413 — Mosaic improves over the lr=3e-4 baseline but still
trails patch-ViT. See `results/phase_b_ablation_note.md` for full eval tables.
