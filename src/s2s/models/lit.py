"""Lightning wrapper for G1 (task #7): training loop + latitude-weighted loss.

Loss is latitude-weighted MSE (cos(lat), matching the weighting used by
s2s.eval.metrics) so high-latitude grid cells -- which cover less actual area
on an equirectangular grid -- don't dominate the gradient just because there
are more of them.
"""
from __future__ import annotations

import lightning as L
import numpy as np
import torch

from s2s.models.patch_vit import PatchViT


def _lat_weights(latitude: np.ndarray) -> torch.Tensor:
    w = np.cos(np.deg2rad(latitude))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


class S2SLitModule(L.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, lead: int, latitude, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = PatchViT(in_channels, out_channels, lead, cfg.model)
        self.register_buffer("lat_weight", _lat_weights(np.asarray(latitude)).view(1, 1, 1, -1, 1))

    def forward(self, x):
        return self.model(x)

    def _weighted_mse(self, pred, target):
        return ((pred - target) ** 2 * self.lat_weight).mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._weighted_mse(self(x), y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self._weighted_mse(self(x), y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=x.shape[0])
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=float(self.cfg.train.lr), weight_decay=float(self.cfg.train.weight_decay)
        )
