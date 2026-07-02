"""G1 - small patch ViT (task #6).

Global input (32x64 @ 5.625 deg) -> patch embed -> transformer -> decode to
multi-channel weekly-mean anomaly fields for weeks 1-6.

Kept tiny on purpose: the baseline's job is a correct pipeline, not capacity.
Dropout (cfg.model.drop_rate) is reused by the MC-dropout arm of the P2 ensemble.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

# Locked for the whole project (configs/data/era5_india.yaml: resolution_deg=5.625).
_GRID = (32, 64)


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Block(nn.Module):
    """Standard pre-norm transformer block (MHSA + MLP)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PatchViT(nn.Module):
    """G1 backbone: patch embed -> +pos_embed -> transformer blocks -> head.

    Input  (B, C_in, lat, lon)        global normalized anomalies (+ history).
    Output (B, lead, C_out, lat, lon) weekly-mean anomaly fields, weeks 1..lead.

    Kept tiny on purpose: the baseline's job is a correct pipeline, not capacity.
    Dropout (cfg.model.drop_rate) is reused by the MC-dropout arm of the P2 ensemble.
    """

    def __init__(self, in_channels: int, out_channels: int, lead: int, cfg):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lead = lead

        patch_size = int(cfg.patch_size)
        embed_dim = int(cfg.embed_dim)
        depth = int(cfg.depth)
        num_heads = int(cfg.num_heads)
        mlp_ratio = float(cfg.mlp_ratio)
        drop_rate = float(cfg.drop_rate)

        if _GRID[0] % patch_size != 0 or _GRID[1] % patch_size != 0:
            raise ValueError(f"patch_size {patch_size} must divide grid {_GRID}")

        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        n_h, n_w = _GRID[0] // patch_size, _GRID[1] // patch_size
        num_patches = n_h * n_w
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.drop = nn.Dropout(drop_rate)
        self.blocks = nn.ModuleList(
            _Block(embed_dim, num_heads, mlp_ratio, drop_rate) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, lead * out_channels * patch_size * patch_size)

    def forward(self, x: torch.Tensor, num_noise_samples: int = 1) -> torch.Tensor:
        """Deterministic backbone. Accepts num_noise_samples for a uniform interface
        with MosaicBackbone; PatchViT has no stochastic mechanism, so M>1 tiles the
        single deterministic prediction into M identical members (zero spread, which
        the calibration metrics will correctly report as under-dispersed)."""
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {c}")
        if (h, w) != _GRID:
            raise ValueError(f"expected grid {_GRID}, got {(h, w)}")

        tokens = self.patch_embed(x)  # (B, embed_dim, n_h, n_w)
        tokens = rearrange(tokens, "b e nh nw -> b (nh nw) e")
        tokens = self.drop(tokens + self.pos_embed)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        out = self.head(tokens)  # (B, N, lead * out_channels * p * p)
        n_h, n_w = h // self.patch_size, w // self.patch_size
        out = rearrange(
            out,
            "b (nh nw) (lead c ph pw) -> b lead c (nh ph) (nw pw)",
            nh=n_h, nw=n_w, lead=self.lead, c=self.out_channels,
            ph=self.patch_size, pw=self.patch_size,
        )
        if int(num_noise_samples) > 1:
            # (B, lead, C, lat, lon) -> (B, M, lead, C, lat, lon), identical members.
            out = out.unsqueeze(1).expand(-1, int(num_noise_samples), -1, -1, -1, -1).contiguous()
        return out
