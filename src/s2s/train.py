"""Hydra entrypoint: train ONE G1 ensemble member (task #7).

Run:  python -m s2s.train  [overrides...]
One process trains one model with seed=cfg.seed. The P2 ensemble
(cfg.train.ensemble) is produced by invoking this entrypoint once per seed,
e.g. a Hydra multirun: `python -m s2s.train -m seed=0,1,2,3,4,5,6,7,8,9` --
not by looping internally. Keeps "one experiment = one config + one commit +
one W&B run" true for every individual member.
"""
from __future__ import annotations

import os
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig

from s2s.data.datamodule import S2SDataModule
from s2s.models.lit import S2SLitModule


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    L.seed_everything(int(cfg.seed), workers=True)

    dm = S2SDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    print(f"in_channels={dm.in_channels}  out_channels={dm.out_channels}  grid={dm.grid}")
    print(
        f"samples: train={len(dm.train_dataset)} "
        f"val={len(dm.val_dataset)} test={len(dm.test_dataset)}"
    )

    lit = S2SLitModule(
        in_channels=dm.in_channels,
        out_channels=dm.out_channels,
        lead=len(cfg.data.lead_weeks),
        latitude=dm.latitude,
        longitude=dm.lon,
        cfg=cfg,
    )

    fast_dev_run = bool(cfg.get("fast_dev_run", False))

    logger = False
    callbacks = []
    if not fast_dev_run:
        ckpt_dir = Path(cfg.paths.checkpoints) / f"seed_{cfg.seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # CSVLogger gives a plain per-epoch metrics.csv alongside W&B -- handy for
        # reading the loss curve back without touching the W&B binary log format.
        logger = [
            WandbLogger(
                project=cfg.project,
                save_dir=os.environ.get("WANDB_DIR", str(ckpt_dir)),
                offline=os.environ.get("WANDB_MODE") == "offline",
            ),
            CSVLogger(save_dir=str(ckpt_dir), name="csv"),
        ]
        callbacks.append(
            ModelCheckpoint(dirpath=str(ckpt_dir), filename="{epoch}-{val_loss:.4f}", monitor="val_loss", save_last=True)
        )

    trainer = L.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        precision=cfg.train.precision,
        gradient_clip_val=float(cfg.train.gradient_clip_val),
        fast_dev_run=fast_dev_run,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=not fast_dev_run,
        accelerator="auto",
        devices=1,
    )
    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()
