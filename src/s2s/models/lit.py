"""Lightning wrapper for G1 (task #7): training loop + latitude-weighted loss.

Loss is latitude-weighted MSE (cos(lat), matching the weighting used by
s2s.eval.metrics) so high-latitude grid cells -- which cover less actual area
on an equirectangular grid -- don't dominate the gradient just because there
are more of them.
"""
from __future__ import annotations

import math

import lightning as L
import numpy as np
import torch

from s2s.models.patch_vit import PatchViT


def _lat_weights(latitude: np.ndarray) -> torch.Tensor:
    w = np.cos(np.deg2rad(latitude))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _build_backbone(in_channels, out_channels, lead, cfg, latitude, longitude):
    name = str(getattr(cfg.model, "name", "patch_vit"))
    if name == "mosaic":
        from s2s.models.mosaic_backbone import MosaicBackbone
        return MosaicBackbone(in_channels, out_channels, lead, cfg.model, latitude, longitude)
    return PatchViT(in_channels, out_channels, lead, cfg.model)


class S2SLitModule(L.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, lead: int, latitude, cfg,
                 longitude=None):
        super().__init__()
        self.cfg = cfg
        self.model = _build_backbone(
            in_channels, out_channels, lead, cfg,
            np.asarray(latitude),
            np.asarray(longitude) if longitude is not None else (np.arange(64) * 5.625),
        )
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
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.train.lr),
            weight_decay=float(self.cfg.train.weight_decay),
        )

        warmup = int(self.cfg.train.warmup_epochs)
        max_ep = int(self.cfg.train.max_epochs)
        min_lr = float(self.cfg.train.min_lr)
        base_lr = float(self.cfg.train.lr)

        def _lr_lambda(epoch: int) -> float:
            # Linear warmup: ramp from 1/warmup to 1.0 over `warmup` epochs.
            if epoch < warmup:
                return (epoch + 1) / warmup
            # Cosine decay from base_lr to min_lr over remaining epochs.
            progress = (epoch - warmup) / max(1, max_ep - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr + cosine * (base_lr - min_lr)) / base_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
