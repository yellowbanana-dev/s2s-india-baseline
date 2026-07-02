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


def fair_crps(members: torch.Tensor, target: torch.Tensor, lat_weight: torch.Tensor) -> torch.Tensor:
    """Latitude-weighted FAIR (unbiased) CRPS for a finite ensemble. Pure function
    so it is unit-testable without constructing a model.

    members: (B, M, lead, C, lat, lon)   target: (B, lead, C, lat, lon)
    lat_weight: broadcastable to (B, lead, C, lat, lon), e.g. (1,1,1,lat,1)

        CRPS_fair = (1/M) Σ_i |x_i - y| - 1/(2 M (M-1)) Σ_{i,j} |x_i - x_j|

    The M(M-1) normalisation (Ferro 2014) is the UNBIASED estimator of the spread
    term E|X-X'|; the biased 1/(2 M^2) form penalises spread and, minimised by SGD,
    collapses the members (the under-dispersion we are fighting). The pairwise sum
    uses the sorted identity  Σ_{i,j}|x_i-x_j| = 2 Σ_k (2k-(M-1)) x_(k)  (ascending),
    which is O(M log M) and avoids the M×M spatial broadcast.
    """
    M = members.shape[1]
    if M < 2:
        raise ValueError("fair CRPS needs at least 2 ensemble members")
    term1 = (members - target.unsqueeze(1)).abs().mean(dim=1)     # (B, lead, C, lat, lon)
    xs, _ = torch.sort(members, dim=1)
    k = torch.arange(M, device=members.device, dtype=members.dtype)
    coeff = (2.0 * k - (M - 1)).view(1, M, 1, 1, 1, 1)
    pairwise_sum = 2.0 * (coeff * xs).sum(dim=1)                  # Σ_{i,j}|x_i-x_j|
    term2 = pairwise_sum / (2.0 * M * (M - 1))
    crps = term1 - term2
    return (crps * lat_weight).mean()


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

        # Training objective: "latitude_weighted_mse" (deterministic, default) or
        # "fair_crps" (Phase-B Stage B probabilistic training). train_members is the
        # ensemble size drawn per step for the CRPS estimator.
        self.loss_name = str(getattr(cfg.train, "loss", "latitude_weighted_mse"))
        self.train_members = int(getattr(cfg.train, "train_members", 8))

    def forward(self, x):
        return self.model(x)

    def _weighted_mse(self, pred, target):
        return ((pred - target) ** 2 * self.lat_weight).mean()

    def _fair_crps(self, members: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return fair_crps(members, target, self.lat_weight)

    def _step(self, batch):
        x, y = batch
        if self.loss_name == "fair_crps":
            if self.train_members < 2:
                raise ValueError(
                    "fair_crps requires train_members >= 2 (got "
                    f"{self.train_members}); a 1-member forward also squeezes the "
                    "member axis, so guard here rather than fail cryptically."
                )
            members = self.model(x, num_noise_samples=self.train_members)
            return self._fair_crps(members, y)
        return self._weighted_mse(self(x), y)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch[0].shape[0])
        return loss

    def _effective_lr(self) -> float:
        # cfg.model.lr overrides cfg.train.lr so per-model defaults (e.g. Mosaic
        # at 3e-5) don't silently revert when the global train default changes.
        model_lr = getattr(self.cfg.model, "lr", None)
        return float(model_lr if model_lr is not None else self.cfg.train.lr)

    def configure_optimizers(self):
        base_lr = self._effective_lr()
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=base_lr,
            weight_decay=float(self.cfg.train.weight_decay),
        )

        warmup = int(self.cfg.train.warmup_epochs)
        max_ep = int(self.cfg.train.max_epochs)
        min_lr = float(self.cfg.train.min_lr)

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
