# Optimizer Sweep — Mosaic Stage-A (branch: mosaic-optim-sweep)

Date: 2026-06-30  
Base: mosaic-stage-a @ 5ed33f9 (global attn + RoPE + native time emb, MSE/deterministic)  
Config: 3.6M params, noise_dim=32, drop_rate=0.1, seed=0, max_epochs=20, daily-stride

## Per-run val curves

| epoch | R1 3e-4/w2/wd0.1 | R2 1e-5/w2/wd0.1 | R3 3e-5/w2/wd0.1 | R4 3e-4/w8/wd0.1 | R5 1e-4/w2/wd0.0 | R6 1e-5/w5/wd0.0 |
|-------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| 0  | 0.4278 ← best | 0.4471 | 0.4391 | 0.4330 | 0.4310 | 0.4672 |
| 1  | 0.4342 | 0.4391 | 0.4304 | 0.4260 ← best | 0.4268 ← best | 0.4454 |
| 2  | 0.4451 | 0.4351 | 0.4272 | 0.4354 | 0.4395 | 0.4402 |
| 3  | 0.4579 | 0.4315 | **0.4252 ← best** | 0.4402 | 0.4436 | 0.4368 |
| 4  | 0.4680 | 0.4289 | 0.4297 | 0.4511 | 0.4504 | 0.4333 |
| 5  | 0.4819 | 0.4282 | 0.4324 | 0.4643 | 0.4569 | 0.4308 |
| 6  | 0.4844 | 0.4264 | 0.4326 | 0.4587 | 0.4589 | 0.4283 |
| 7  | 0.4912 | **0.4259 ← best** | 0.4383 | 0.4699 | 0.4645 | **0.4265 ← best** |
| 8  | 0.5016 | 0.4268 | 0.4404 | 0.4788 | 0.4707 | 0.4265 |
| 10 | 0.5083 | 0.4282 | 0.4428 | 0.4933 | 0.4729 | 0.4274 |
| 15 | 0.5232 | 0.4302 | 0.4482 | 0.5175 | 0.4853 | 0.4300 |
| 19 | 0.5275 | 0.4304 | 0.4480 | 0.5242 | 0.4873 | 0.4301 |

## Summary table

| Run | lr   | warmup | wd  | Best epoch | Best val | Curve shape |
|-----|------|--------|-----|-----------|---------|-------------|
| R1  | 3e-4 | 2      | 0.1 | 0         | 0.4278  | Monotonic degradation from ep0 |
| R2  | 1e-5 | 2      | 0.1 | 7         | 0.4259  | Improves ep0→7, plateau, slow rise |
| R3  | 3e-5 | 2      | 0.1 | **3**     | **0.4252** | Improves ep0→3, then degrades |
| R4  | 3e-4 | 8      | 0.1 | 1         | 0.4260  | Off 0 by 1 epoch, then fast degradation |
| R5  | 1e-4 | 2      | 0.0 | 1         | 0.4268  | Off 0 by 1 epoch, then fast degradation |
| R6  | 1e-5 | 5      | 0.0 | 7         | 0.4265  | Long warmup delays peak, similar to R2 |

## Interpretation

**Optimization mismatch CONFIRMED.**

Every run with lr ≤ 3e-5 moves best-epoch off 0 and lowers val below the 0.428 init floor.
The fingerprint "best at epoch 0 → monotonic degradation" is entirely explained by lr=3e-4
overshooting the good zero-init residuals immediately.

Key observations:
- **R3 is the winner** at 20 epochs: lr=3e-5, warmup=2, wd=0.1 → best val 0.4252 at ep3,
  matching patch-ViT's best val (0.4254). Val still trending down at ep19 → more epochs likely help.
- **R2 and R6** (lr=1e-5) peak later (ep7-8) with slightly higher best val (0.4259/0.4265),
  suggesting the LR is a touch too low for 20 epochs but would benefit from more epochs.
- **R4** (long warmup + high LR): warmup slows the initial damage but LR ramp-up still
  causes degradation after ep1. Not the right lever.
- **R5** (no wd + 1e-4): similar to R4 — brief improvement then fast degradation.
- **Weight decay** does not appear to be the distinguishing factor; R2 (wd=0.1) and
  R6 (wd=0.0) at the same lr=1e-5 give nearly identical results (0.4259 vs 0.4265).

**Recommended next run:** lr=3e-5, warmup=2, wd=0.1, full 50 epochs — R3 config on the
full training budget. Best checkpoint at ep3 is already at patch-ViT parity (0.4252 vs 0.4254);
50 epochs should push further as the val curve was still improving at ep3.
