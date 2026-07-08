"""G1 - small patch ViT (task #6).

Global input (32x64 @ 5.625 deg) -> patch embed -> transformer -> decode to
multi-channel weekly-mean anomaly fields for weeks 1-6.

Kept tiny on purpose: the baseline's job is a correct pipeline, not capacity.
Dropout (cfg.model.drop_rate) is reused by the MC-dropout arm of the P2 ensemble.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat

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

    def __init__(self, in_channels: int, out_channels: int, lead: int, cfg, seed: int = 42):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lead = lead
        self.seed = int(seed)

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

        # Stochastic member mechanism (Fix 6/C3): FiLM-condition the patch tokens on
        # a noise vector drawn per (sample, member). Comparable cheapness to Mosaic's
        # cSwiGLU noise (one Linear). Gated on noise_dim>0 so the deterministic
        # patch-ViT (default configs) is byte-for-byte unchanged. Zero-init => members
        # start identical and spread emerges under fair-CRPS training.
        self.noise_dim = int(getattr(cfg, 'noise_dim', 0))
        self._noise_gen = None
        if self.noise_dim > 0:
            self.to_film = nn.Linear(self.noise_dim, 2 * embed_dim)
            nn.init.zeros_(self.to_film.weight)
            nn.init.zeros_(self.to_film.bias)

    def _draw_noise(self, n: int, device, dtype) -> torch.Tensor:
        """n noise vectors from a seeded generator (reproducible members)."""
        if self._noise_gen is None:
            self._noise_gen = torch.Generator(device=device)
            self._noise_gen.manual_seed(self.seed)
        return torch.randn((n, self.noise_dim), generator=self._noise_gen,
                           device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, num_noise_samples: int = 1) -> torch.Tensor:
        """num_noise_samples == 1 (default): deterministic (B, lead, C, lat, lon).
        num_noise_samples == M > 1:
          - noise_dim > 0 (Fix 6/C3): FiLM-inject a per-(sample, member) noise vector
            at the patch embedding, yielding M DISTINCT members -> (B, M, lead, C, lat, lon).
          - noise_dim == 0 (legacy): tile the single deterministic prediction into M
            identical members (zero spread; calibration metrics report under-dispersion)."""
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {c}")
        if (h, w) != _GRID:
            raise ValueError(f"expected grid {_GRID}, got {(h, w)}")
        M = int(num_noise_samples)
        n_h, n_w = h // self.patch_size, w // self.patch_size

        tokens = self.patch_embed(x)  # (B, embed_dim, n_h, n_w)
        tokens = rearrange(tokens, "b e nh nw -> b (nh nw) e")
        tokens = self.drop(tokens + self.pos_embed)

        if self.noise_dim > 0 and M > 1:
            # Stochastic ensemble: replicate tokens per member and FiLM-condition on noise.
            tokens = repeat(tokens, "b n e -> (b m) n e", m=M)
            z = self._draw_noise(b * M, x.device, x.dtype)          # (B*M, noise_dim)
            gamma, beta = self.to_film(z).chunk(2, dim=-1)          # each (B*M, embed_dim)
            tokens = (1.0 + gamma).unsqueeze(1) * tokens + beta.unsqueeze(1)
            for block in self.blocks:
                tokens = block(tokens)
            tokens = self.norm(tokens)
            out = self.head(tokens)                                 # (B*M, N, lead*C*p*p)
            return rearrange(
                out,
                "(b m) (nh nw) (lead c ph pw) -> b m lead c (nh ph) (nw pw)",
                b=b, m=M, nh=n_h, nw=n_w, lead=self.lead, c=self.out_channels,
                ph=self.patch_size, pw=self.patch_size,
            )

        # Deterministic path (unchanged).
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        out = self.head(tokens)  # (B, N, lead * out_channels * p * p)
        out = rearrange(
            out,
            "b (nh nw) (lead c ph pw) -> b lead c (nh ph) (nw pw)",
            nh=n_h, nw=n_w, lead=self.lead, c=self.out_channels,
            ph=self.patch_size, pw=self.patch_size,
        )
        if M > 1:
            out = out.unsqueeze(1).expand(-1, M, -1, -1, -1, -1).contiguous()
        return out
