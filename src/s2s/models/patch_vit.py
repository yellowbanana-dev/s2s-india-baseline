"""G1 - small patch ViT (task #6).

Global input (32x64 @ 5.625 deg) -> patch embed -> transformer -> decode to
multi-channel weekly-mean anomaly fields for weeks 1-6.

Kept tiny on purpose: the baseline's job is a correct pipeline, not capacity.
Dropout (cfg.model.drop_rate) is reused by the MC-dropout arm of the P2 ensemble.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class PatchViT(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, lead: int, cfg):
        super().__init__()
        # TODO: patch_embed -> +pos_embed -> transformer blocks -> head
        # Output shape: (B, lead, out_channels, lat, lon)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
