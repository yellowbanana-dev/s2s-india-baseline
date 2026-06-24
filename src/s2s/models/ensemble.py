"""P2 cheap ensemble (task #8): IC-perturbation members around trained checkpoint(s).

Per ADR 0001 (docs/adr/0001-mosaic-corner-over-latent-diffusion.md), perturbation
is i.i.d. Gaussian noise added to a randomly placed rectangular tile ("mosaic
corner") of the input field, varied per member by RNG seed -- cheap, fully
deterministic given a seed, no extra model to train or validate.

When only one trained checkpoint is available (e.g. a Phase-A smoke test), the
ensemble is produced entirely by IC perturbation around that single model; once
cfg.train.ensemble.seeds each have their own trained checkpoint, pass one
S2SLitModule per seed and this class round-robins models across members.
"""
from __future__ import annotations

import torch

from s2s.models.lit import S2SLitModule


def _mosaic_corner_noise(x: torch.Tensor, std: float, generator: torch.Generator) -> torch.Tensor:
    """Add i.i.d. Gaussian noise to one randomly placed tile, half the grid per side."""
    noisy = x.clone()
    _, c, h, w = x.shape
    th, tw = max(1, h // 2), max(1, w // 2)
    top = int(torch.randint(0, h - th + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, w - tw + 1, (1,), generator=generator).item())
    noise = torch.randn(x.shape[0], c, th, tw, generator=generator) * std
    noisy[:, :, top : top + th, left : left + tw] += noise.to(x.device)
    return noisy


class P2Ensemble:
    """Wraps one or more trained S2SLitModules into an n_members-wide forecast.

    forecast(x) -> (n_members, B, lead, out_channels, lat, lon) stacked predictions.
    """

    def __init__(self, models: list[S2SLitModule], cfg, seeds=None):
        if not models:
            raise ValueError("P2Ensemble needs at least one trained model")
        for m in models:
            m.eval()
        self.models = models
        self.cfg = cfg
        self.std = float(cfg.train.ensemble.ic_perturbation_std)
        self.seeds = list(seeds) if seeds is not None else list(cfg.train.ensemble.seeds)

    @torch.no_grad()
    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        n_models = len(self.models)
        members = []
        for i, seed in enumerate(self.seeds):
            model = self.models[i % n_models]
            generator = torch.Generator().manual_seed(int(seed))
            x_member = _mosaic_corner_noise(x, self.std, generator)
            members.append(model(x_member))
        return torch.stack(members, dim=0)
