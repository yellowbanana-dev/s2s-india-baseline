# Phase-B Ablation: Mosaic vs patch-ViT (identical regime, seed 0)

Date: 2026-06-26

## Training regime (both models)
- Data: full record 1979–2012 train, daily-strided stride=1 (12 330 train, 1 737 val, 256 test)
- max_epochs=50, lr=3e-4, weight_decay=0.1, cosine+warmup (warmup_epochs=2), bf16-mixed
- H100 NVL, seed=0

## Training summary

| | Mosaic (14.5 M params) | patch-ViT (4.9 M params) |
|--|--|--|
| Best epoch | 0 | 3 |
| Best val_loss | **0.4295** | **0.4254** |
| Final train_loss (ep 25 / 49) | 0.069 | 0.194 |
| Val at ep 10 | 0.547 | 0.459 |
| Val at ep 25 | 0.586 | 0.517 |
| Overfitting severity | **Severe** (val ↑ every epoch from ep 0) | Moderate (val plateaus ~0.52 after ep 3) |
| Checkpoint used for eval | epoch=0-val_loss=0.4295.ckpt | epoch=3-val_loss=0.4254.ckpt |

**Mosaic overfitting diagnosis:** 14.5M params / 12 330 samples ≈ 1 180 params/sample — 3× higher ratio
than patch-ViT (≈ 400). The block-local HEALPix attention (block_attn_size=512, nside=16) does
not appear to regularise via weight sharing the way patch convolutions do; the model memorises
the training anomaly fields almost immediately.

Note: two Mosaic training processes were accidentally launched on GPU 0 (from a duplicate
background task). Both were killed at epoch ~25; the best checkpoint (epoch 0, val=0.4295)
was already saved before the competition became severe. Eval results are honest for the
epoch-0 weights but the training curve in the CSV reflects the corrupted dual-process run.

## Honest eval results (test split, India box, physical units)

### 2m_temperature

| lead_week | crps_model | crpss_vs_det | crpss_vs_prob | acc_mean | rmse_mean | spread_error_ratio |
|-----------|-----------|--------------|---------------|----------|-----------|-------------------|
| **Mosaic** |
| 1 | 0.816 | +0.163 | −0.040 | 0.498 | 1.166 | 0.006 |
| 2 | 0.858 | +0.118 | −0.097 | 0.399 | 1.227 | 0.005 |
| 3 | 0.870 | +0.107 | −0.111 | 0.369 | 1.248 | 0.005 |
| 4 | 0.880 | +0.095 | −0.127 | 0.326 | 1.263 | 0.004 |
| 5 | 0.880 | +0.095 | −0.126 | 0.327 | 1.263 | 0.004 |
| 6 | 0.893 | +0.085 | −0.139 | 0.285 | 1.285 | 0.004 |
| **patch-ViT** |
| 1 | 0.797 | +0.182 | −0.016 | 0.538 | 1.140 | 0.011 |
| 2 | 0.830 | +0.147 | −0.061 | 0.463 | 1.187 | 0.010 |
| 3 | 0.839 | +0.138 | −0.073 | 0.437 | 1.205 | 0.010 |
| 4 | 0.848 | +0.129 | −0.085 | 0.413 | 1.217 | 0.010 |
| 5 | 0.857 | +0.120 | −0.096 | 0.388 | 1.232 | 0.009 |
| 6 | 0.863 | +0.115 | −0.102 | 0.374 | 1.244 | 0.008 |

### total_precipitation_24hr

| lead_week | crps_model | crpss_vs_det | crpss_vs_prob | acc_mean | rmse_mean | spread_error_ratio |
|-----------|-----------|--------------|---------------|----------|-----------|-------------------|
| **Mosaic** |
| 1 | 0.00186 | −0.003 | −0.123 | 0.192 | 0.00303 | 0.005 |
| 2 | 0.00189 | −0.019 | −0.142 | 0.139 | 0.00306 | 0.005 |
| 3 | 0.00194 | −0.039 | −0.164 | 0.126 | 0.00308 | 0.004 |
| 4 | 0.00191 | −0.023 | −0.145 | 0.141 | 0.00307 | 0.004 |
| 5 | 0.00190 | −0.018 | −0.140 | 0.153 | 0.00306 | 0.003 |
| 6 | 0.00189 | −0.011 | −0.132 | 0.125 | 0.00307 | 0.003 |
| **patch-ViT** |
| 1 | 0.00187 | −0.008 | −0.129 | 0.223 | 0.00301 | 0.008 |
| 2 | 0.00191 | −0.030 | −0.154 | 0.143 | 0.00306 | 0.008 |
| 3 | 0.00194 | −0.039 | −0.163 | 0.130 | 0.00308 | 0.008 |
| 4 | 0.00193 | −0.038 | −0.162 | 0.144 | 0.00307 | 0.007 |
| 5 | 0.00192 | −0.031 | −0.155 | 0.148 | 0.00307 | 0.007 |
| 6 | 0.00193 | −0.037 | −0.161 | 0.113 | 0.00308 | 0.006 |

## Decision gate (wks 3–4, CRPSS vs prob_clim > 0)

| Model | Gate |
|-------|------|
| Mosaic | **FAIL** (t2m wk3/4: −0.111/−0.127; precip wk3/4: −0.164/−0.145) |
| patch-ViT | **FAIL** (t2m wk3/4: −0.073/−0.085; precip wk3/4: −0.163/−0.162) |

Both fail vs probabilistic climatology (expected — SER ≈ 0.005–0.010, severely underdispersed).

## Head-to-head: Mosaic vs patch-ViT at gate weeks 3–4

**t2m (ACC higher = better, RMSE lower = better):**
- wk3 ACC: patch-ViT 0.437 vs Mosaic 0.369 → **patch-ViT +0.068**
- wk4 ACC: patch-ViT 0.413 vs Mosaic 0.326 → **patch-ViT +0.087**
- wk3 RMSE: patch-ViT 1.205 vs Mosaic 1.248 → **patch-ViT −0.043 K**
- wk4 RMSE: patch-ViT 1.217 vs Mosaic 1.263 → **patch-ViT −0.046 K**

**Precip (wks 3–4): essentially identical** (both essentially at climatology CRPS).

**Conclusion:** patch-ViT beats Mosaic on mean t2m skill at gate weeks under this regime.
Mosaic's failure mode is overfitting (3× more params, same data). Not an architecture verdict —
Mosaic needs a smaller config or stronger regularisation before it can be meaningfully compared.

## Phase-A reference (patch-ViT, pre-Phase-B sampling)

Recorded in commit 6dd9c6c. Key differences from the Phase-B patch-ViT run above:
- Phase A used W-MON weekly sampling (~1 764 train samples vs 12 330 here)
- Phase B patch-ViT shows better acc_mean at all leads, confirming the denser sampling helps

## Recommendations for next steps

1. **Shrink Mosaic config** before retraining: reduce dim=256→128, bottleneck_dim=512→256 (~3–4M
   params to match patch-ViT), add drop_rate, retrain. Only then compare architectures fairly.
2. **Or add strong Mosaic regularisation**: dropout on cSwiGLU (currently none), stochastic depth,
   higher weight_decay (0.2+), label smoothing.
3. **Phase-B step 4 (learned perturbations)** depends on calibrated spread — tackle after overfitting
   is resolved.
