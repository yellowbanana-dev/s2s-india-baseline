# ADR 0002: Vendor Mosaic backbone in dense-attention mode at 5.625 deg

## Status

Accepted (Phase B, step 3).

## Context

Phase B step 3 (phase-b-plan.md §6.3) replaces the G1 PatchViT with a more capable
backbone that:
1. Operates on a native-resolution grid without pre-chunking into coarse patches.
2. Includes a learned functional-perturbation mechanism (cSwiGLU + NoiseGenerator)
   needed by Phase B step 4 (task 6: IC-perturbation → learned ensemble).
3. Keeps the full pipeline (data, splits, metrics, baselines) unchanged.

ADR-0001 already chose Mosaic-corner IC perturbation over latent diffusion for the
Phase A ensemble. The Mosaic paper's codebase (Zhdanov, Lucic, Welling, van de Meent —
ICML 2026, https://github.com/maxxxzdn/mosaic, CC-BY-NC-4.0) implements exactly the
noise-injection FFN (cSwiGLU + NoiseGenerator) that Phase B step 4 needs.

Two implementation paths were considered:

1. **Clean-room rewrite** — write a native-resolution attention backbone from scratch,
   replicating only the pieces we need (cross-attention interpolation between lon-lat and
   HEALPix, cSwiGLU, NoiseGenerator).

2. **Vendor Mosaic** — copy the upstream source into `src/s2s/models/mosaic/`, apply
   minimal local modifications (optional flash_attn, SDPA fallback, remove ops.py
   dependency), and build a thin adapter that presents the same interface PatchViT uses.

## Decision

**Vendor Mosaic** (option 2).

## Rationale

- **Faithful to ADR-0001's Mosaic corner.** The Mosaic codebase is the direct
  expression of the "Mosaic architecture" referenced in the research plan. Vendoring it
  reuses the exact cSwiGLU and NoiseGenerator implementations verbatim, which matters
  for Phase B step 4: learned functional perturbation must match the paper's design so
  the ensemble diagnostic comparisons stay reproducible against the published baseline.

- **Less surface area to validate.** A clean-room rewrite requires separately
  validating cross-attention interpolation between irregular grids (lon-lat ↔ HEALPix),
  HEALPix up/downsample layers, and RoPE for spherical positions — all against an
  external reference. Vendoring eliminates that validation: the upstream tests and paper
  results serve as the reference.

- **ADR-0001 insight: dense mode at coarse resolution.** ADR-0001 notes "full
  attention at coarse res; block-sparse only when scaling to fine grids." At 5.625 deg
  (32 × 64 = 2048 grid points → nside=16 HEALPix with 3072 pixels), the block-sparse
  Triton kernel (ops.py) is unnecessary overhead and a hard CUDA-build dependency.
  Setting `sparse_every ≤ 0` makes all MosaicBlocks run in `block_attn_only=True` mode,
  which uses only local block attention (implemented with flash_attn / SDPA) — no
  ops.py import required.

## Scope of local modifications

The following changes are made to the vendored files (marked in-source with
`# LOCAL MODIFICATION:`):

| File | Change |
|------|--------|
| `primitives.py` | flash_attn import is made optional; SDPA fallback added to `block_attention()`. `from ops import mosaic_sparse_attn` removed; guard added to raise `ImportError` if the sparse path is ever reached (it never is when `sparse_every ≤ 0`). |
| `mosaic.py` | Import paths changed from bare `from utils import ...` to `from s2s.models.mosaic.utils import ...` (package-relative). Same for `primitives`. |
| `utils.py` | Import path change only (standalone, no functional change). |
| `base.py` | Dataset import replaced with a forward-declaration stub (we use `Transformer` directly, not `WeatherModel`). |

`ops.py` (the Triton block-sparse CUDA kernel) is **not vendored** — it is only
needed for the `sparse_every > 0` code path that we permanently disable.

## HEALPix resolution choice: nside=16

Our lon-lat grid has 32 × 64 = 2048 points.

| nside | npix | ratio to our grid |
|-------|------|-------------------|
| 8     | 768  | 0.38× (too coarse) |
| 16    | 3072 | 1.50× (**chosen**) |
| 32    | 12288 | 6.00× (overkill at 5.625 deg) |

nside=16 gives ~1.5× oversampling relative to our grid, comparable to what WeatherBench2
models use for regridding to HEALPix before processing. It is set as `cfg.model.nside`
and can be changed without code changes.

The single CG-stage U-Net downsamples to nside=8 (768 tokens) for the bottleneck,
where full attention (block_attn_size = 768) is affordable (768² ≈ 590K ops per head
per batch item; flash_attn / SDPA eliminates the O(N²) memory).

## Adapter axis mapping (`mosaic_backbone.py`)

Our data format:  `(B, C_in=13, lat=32, lon=64)`
Mosaic input:     `(b, n=1, t=1, lon=64, lat=32, c=C_in)` — lon-first, variables last

Steps in `MosaicBackbone.forward`:
1. Permute lat↔lon:   `(B, 13, lat=32, lon=64)` → `(B, 13, lon=64, lat=32)`
2. Reshape to Mosaic: `→ (B, 1, 1, 64, 32, 13)`   (n=1, t=1, c=13)
3. Run `Transformer.forward(x, day_year_time=zeros(B,1,2), num_noise_samples=1)`
4. Output:            `(B, 1, lon=64, lat=32, n_lead*C_out=12)`
5. Squeeze n:         `(B, 64, 32, 12)`
6. Permute lon↔lat:   `(B, 32, 64, 12)`
7. Reshape leads:     `(B, 32, 64, n_lead=6, C_out=2)`
8. Permute to target: `(B, 6, 2, 32, 64)`

The `postprocess[-1]` linear in `Transformer` is replaced at construction time:
`nn.Linear(cfg.model.dim, len(variables)=13)` → `nn.Linear(cfg.model.dim, n_lead*C_out=12)`

`num_noise_samples=1` during the deterministic backbone stage. The NoiseGenerator /
cSwiGLU stays wired so that Phase B step 4 can increase `num_noise_samples` without
any architectural change.

`day_year_time` is passed as zeros — our doy_cos channel (last of C_in=13) already
encodes the seasonal cycle as a normalised cosine. Mosaic's built-in time embedding
(sin/cos of day/year) would be redundant; passing zeros disables it cleanly without
removing the code path.

`static_variables = []` — no static surface fields (orography, land-sea mask) in our
current ERA5 anomaly dataset. The `space_dim=3` XYZ position encoding from
`initialize_static_vars` is always appended, providing grid-point coordinates to the
model.

## Approximate parameter count

| Component | Params |
|-----------|--------|
| CrossAttentionInterpolate × 2 | ~394 k |
| Preprocess Linear layers | ~71 k |
| Postprocess (with replaced head) | ~69 k |
| Encoder stage (nside=16, dim=256, depth=2) | ~1.12 M |
| HEALPix Downsample (256→512) | ~530 k |
| Bottleneck (nside=8, dim=512, depth=2) | ~4.32 M |
| HEALPix Upsample (512→256) | ~536 k |
| Decoder stage (nside=16, dim=256, depth=2) | ~1.12 M |
| NoiseGenerator | ~1 k |
| **Total** | **~8–9 M** |

All within the "tens of M" budget on H100.

## Consequences

- **License**: CC-BY-NC-4.0. This project is non-commercial research; attribution
  required. Each vendored file carries a header crediting the original authors.
- **New runtime deps**: `healpy`, `einops`, `scikit-learn` — added to `environment.yml`.
- **flash_attn**: optional. When absent, `block_attention()` falls back to
  `torch.nn.functional.scaled_dot_product_attention` (PyTorch ≥ 2.0 SDPA, which uses
  FlashAttention kernels when available). Tests verify the model runs without flash_attn.
- **PatchViT is not deleted**: both `model=patch_vit` and `model=mosaic` remain
  selectable via Hydra. The ablation between them is the subject of a later evaluation
  step, not this ADR.
- **No retraining in this step**: this ADR only lands the architecture and proves one
  training step succeeds. Full Mosaic-backbone retraining is the next step.
