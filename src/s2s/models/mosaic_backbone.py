"""MosaicBackbone — thin adapter wiring the vendored Transformer to our pipeline.

Presents the same interface as PatchViT (see patch_vit.py):
    __init__(in_channels, out_channels, lead, cfg, latitude, longitude)
    forward(x: (B, C_in, lat, lon)) -> (B, lead, C_out, lat, lon)

See ADR-0002 for the full axis-mapping rationale.

Axis mapping summary
--------------------
Our format:   (B, C_in=13, lat=32, lon=64)           — lat-first
Mosaic input: (b, n=1, t=1, lon=64, lat=32, c=C_in)  — lon-first, vars last

Steps in forward():
  1. (B, C_in, lat, lon) → permute lat↔lon → (B, C_in, lon, lat)
  2. → (B, 1, 1, lon, lat, C_in)                 [n=1, t=1]
  3. Transformer.forward(x, day_year_time=0, num_noise_samples=1)
  4. Output: (B, 1, lon, lat, n_lead*C_out)
  5. Squeeze n → (B, lon, lat, n_lead*C_out)
  6. Permute lon↔lat → (B, lat, lon, n_lead*C_out)
  7. Reshape leads → (B, lat, lon, n_lead, C_out)
  8. Permute to target → (B, n_lead, C_out, lat, lon)

The `postprocess[-1]` linear in Transformer is replaced at construction time so
that output channels = n_lead * C_out (=12) instead of len(variables) (=13).

`num_noise_samples`=1 → deterministic (B,lead,C,lat,lon); >1 → ensemble
                  (B,M,lead,C,lat,lon) via Mosaic's NoiseGenerator (Phase-B Stage B).
`day_year_time` — day_normalized derived from the (doy_cos, doy_sin) input channels
(C_in[-2], C_in[-1]) via atan2 for the true year phase (Fix 5b/M5);
                  year left at 0 (not in batch). See forward() for the mapping.
`static_variables=[]` — no static fields yet; space_dim=3 XYZ coords are always
                        appended by initialize_static_vars.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

from s2s.models.mosaic.mosaic import (
    Transformer, ModelConfig, StageConfig, BottleneckConfig,
)

# Grid (lat, lon) is derived from the datamodule's coordinates, not hardcoded, so the
# adapter is resolution-agnostic (5.625 deg 32x64 ... 1.5 deg 121x240; lever f / ADR-0007).


class MosaicBackbone(nn.Module):
    """Mosaic Transformer adapted to the s2s-india-baseline pipeline interface."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lead: int,
        cfg,
        latitude: np.ndarray,
        longitude: np.ndarray,
        seed: int = 42,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lead = lead
        self._lat = int(len(latitude))
        self._lon = int(len(longitude))

        mc = cfg   # OmegaConf DictConfig from configs/model/mosaic.yaml

        nside_stage = int(mc.nside)
        nside_bn    = int(mc.bottleneck_nside)
        dim         = int(mc.dim)
        bn_dim      = int(mc.bottleneck_dim)
        num_heads   = int(mc.num_heads)
        bn_heads    = int(mc.bottleneck_num_heads)

        stage_cfg = StageConfig(
            nside=nside_stage,
            dim=dim,
            num_heads=num_heads,
            block_attn_size=int(mc.block_attn_size),
            sparse_block_size=int(mc.sparse_block_size),
            sparse_block_count=int(mc.sparse_block_count),
            encoder_depth=int(mc.encoder_depth),
            decoder_depth=int(mc.decoder_depth),
            mlp_ratio=float(mc.mlp_ratio),
            gqa_ratio=int(mc.gqa_ratio),
        )

        bn_cfg = BottleneckConfig(
            nside=nside_bn,
            dim=bn_dim,
            num_heads=bn_heads,
            block_attn_size=int(mc.bottleneck_block_attn_size),
            sparse_block_size=int(mc.sparse_block_size),
            sparse_block_count=int(mc.sparse_block_count),
            depth=int(mc.bottleneck_depth),
            mlp_ratio=float(mc.mlp_ratio),
            gqa_ratio=int(mc.gqa_ratio),
        )

        model_cfg = ModelConfig(
            dim=dim,
            num_heads=num_heads,
            k_neighbors=int(mc.k_neighbors),
            qk_norm=bool(mc.qk_norm),
            rope=bool(mc.rope),
            rope_theta=int(mc.rope_theta),
            sparse_every=int(mc.sparse_every),
            variables=[f"ch_{i}" for i in range(in_channels)],
            static_variables=[],
            qkv_compress_ratio=int(mc.qkv_compress_ratio),
            cg_stage_cfgs=[stage_cfg],
            bottleneck_cfg=bn_cfg,
            num_history_steps=1,
            noise_dim=int(mc.noise_dim),
            drop_rate=float(getattr(mc, "drop_rate", 0.0)),
            ortho_init=bool(getattr(mc, "ortho_init", False)),
            rmsnorm_elementwise_affine=True,
            no_compression=bool(getattr(mc, "no_compression", False)),
            gate_slots=int(getattr(mc, "gate_slots", 3)),
            selection=bool(getattr(mc, "selection", False)),
        )

        # seed threads cfg.seed (Fix 8/M3): the noise RNG must differ across
        # 'independent' training seeds, not be hardcoded to 42.
        self.transformer = Transformer(model_cfg, seed=int(seed))

        # Replace postprocess head: Transformer outputs len(variables)=C_in channels,
        # but we need n_lead * C_out channels. Swap the final Linear in-place.
        out_dim = lead * out_channels
        self.transformer.postprocess[-1] = nn.Linear(dim, out_dim, bias=False)
        nn.init.normal_(
            self.transformer.postprocess[-1].weight,
            mean=0.0,
            std=1.0 / math.sqrt(dim),
        )

        # Initialise grid mappings (CPU; buffers will be moved to device by .to()).
        lon_t = torch.tensor(longitude, dtype=torch.float32)
        lat_t = torch.tensor(latitude,  dtype=torch.float32)
        self.transformer.initialize_interpolation(lon_t, lat_t)
        # No static surface fields; space_dim=3 XYZ is appended by initialize_static_vars.
        empty_static = torch.zeros(len(longitude), len(latitude), 0)
        self.transformer.initialize_static_vars(empty_static, lon_t, lat_t)

    def forward(self, x: torch.Tensor, num_noise_samples: int = 1) -> torch.Tensor:
        """Forward pass.

        num_noise_samples == 1 (default): deterministic, returns
            (B, lead, C_out, lat, lon)  -- back-compatible with the MSE path,
            PatchViT, and the existing eval/tests.
        num_noise_samples == M > 1: probabilistic ensemble (Phase-B Stage B).
            Mosaic's NoiseGenerator draws a fresh functional-perturbation vector
            per (sample, member) and injects it in every cSwiGLU FFN, yielding M
            distinct members. Returns (B, M, lead, C_out, lat, lon).
        """
        b, c, lat, lon = x.shape
        if c != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {c}")
        if (lat, lon) != (self._lat, self._lon):
            raise ValueError(f"expected grid {(self._lat, self._lon)}, got {(lat, lon)}")
        M = int(num_noise_samples)

        # Extract day_normalized BEFORE permuting x. The last TWO input channels are
        # doy_cos (C_in[-2]) and doy_sin (C_in[-1]) (Fix 5b/M5), broadcast uniformly
        # over space. atan2(sin, cos) recovers the TRUE phase in (-π, π]; /2π (+1 mod 1)
        # gives day_normalized ∈ [0, 1). Previously only doy_cos was available and
        # arccos folded the year in half ([0, 0.5]), so Jan/Dec (and spring/autumn)
        # were indistinguishable to the seasonal embedding. Year stays 0.
        doy_cos_vals = x[:, -2, 0, 0]  # (B,), spatial broadcast → [0,0] is representative
        doy_sin_vals = x[:, -1, 0, 0]  # (B,)
        day_normalized = torch.atan2(doy_sin_vals, doy_cos_vals) / (2.0 * math.pi)  # (B,) in (-0.5, 0.5]
        day_normalized = day_normalized % 1.0  # -> [0, 1)
        day_year_time = torch.zeros(b, 1, 2, device=x.device, dtype=x.dtype)
        day_year_time[:, 0, 0] = day_normalized

        # Step 1-2: (B, C_in, lat, lon) → (B, 1, 1, lon, lat, C_in)
        x = x.permute(0, 2, 3, 1)          # (B, lat, lon, C_in)
        x = x.permute(0, 2, 1, 3)          # (B, lon, lat, C_in)
        x = x.unsqueeze(1).unsqueeze(2)    # (B, 1, 1, lon, lat, C_in)

        # Step 3: Mosaic forward (n=1, t=1). Output: (B, M, lon, lat, lead*C_out).
        out = self.transformer(x, day_year_time, num_noise_samples=M)

        # Steps 4-8: reshape to (B, M, lead, C_out, lat, lon).
        out = out.permute(0, 1, 3, 2, 4)                       # (B, M, lat, lon, lead*C_out)
        out = out.reshape(b, M, self._lat, self._lon, self.lead, self.out_channels)
        out = out.permute(0, 1, 4, 5, 2, 3).contiguous()       # (B, M, lead, C_out, lat, lon)
        if M == 1:
            return out[:, 0]                                   # (B, lead, C_out, lat, lon)
        return out
