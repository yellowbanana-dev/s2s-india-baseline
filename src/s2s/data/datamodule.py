"""Stage 2c - Lightning DataModule / Dataset (task #7).

Reads the already-built data/processed/daily_anom.zarr (Stage 2, scripts/01_build_dataset.py),
standardizes every variable using TRAIN-only mean/std (cardinal rule), and builds
per-sample tensors:

  train / val  -- daily-strided rolling 7-day windows (daily_init_weekly_windows).
                  Stride controlled by cfg.data.train_stride_days (default 1 => ~7x denser).
                  Denser windows DO NOT change normalizer stats -- stats are fit on raw
                  daily train anomalies before any windowing.
  test         -- W-MON weekly bins (assemble_arrays), kept for eval comparability.

Everything is loaded eagerly into numpy -- dev-subset sizes are small;
a full-record run would need a lazy rewrite.
"""
from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from s2s.data.assemble import assemble_arrays, in_out_channels
from s2s.data.assemble import target_vars as _target_vars
from s2s.data.climatology import fit_normalizer
from s2s.data.windows import daily_init_weekly_windows, daily_to_weekly_mean

_SPLITS = ("train", "val", "test")


class S2SDataset(Dataset):
    """Wraps pre-assembled (inputs, targets) numpy arrays as torch tensors."""

    def __init__(self, arrays: dict):
        self.inputs = torch.from_numpy(arrays["inputs"])
        self.targets = torch.from_numpy(arrays["targets"])
        # Keep the per-sample init-week timestamps (assemble_arrays / window builders
        # both return "time") so eval can hard-check its own time reconstruction
        # against the actual assembled axis instead of a re-derivation. None-safe.
        self.time = arrays.get("time")

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def _standardize(ds: xr.Dataset, normalizer: dict) -> xr.Dataset:
    """Apply TRAIN-only mean/std to every variable in `ds` (val/test included)."""
    out = ds.copy()
    for var, stats in normalizer.items():
        if var in out.data_vars:
            std = stats["std"] if stats["std"] > 0 else 1.0
            out[var] = (out[var] - stats["mean"]) / std
    return out


class S2SDataModule(L.LightningDataModule):
    """Wires Stage 1 + 2 (already-built daily_anom.zarr) into train/val/test loaders."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.in_channels, self.out_channels = in_out_channels(cfg)
        self.target_vars = _target_vars(cfg)
        self.grid = None
        self.latitude = None
        self.lon = None
        self.normalizer = None  # TRAIN-only per-variable {mean, std}; physical <-> standardized
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """No download here -- Stage 2 (scripts/01_build_dataset.py) must already
        have written daily_anom.zarr. Training shouldn't silently trigger a
        network pull; fail loudly with the command to run instead."""
        anom_path = Path(self.cfg.data.paths.processed) / "daily_anom.zarr"
        if not anom_path.exists():
            raise FileNotFoundError(
                f"{anom_path} not found. Run scripts/01_build_dataset.py first "
                f"(e.g. `python scripts/01_build_dataset.py data.dev_years=[2010,2018]`)."
            )

    def setup(self, stage: str | None = None):
        if self.train_dataset is not None:
            return  # idempotent: Lightning may call setup() more than once

        anom_path = Path(self.cfg.data.paths.processed) / "daily_anom.zarr"
        daily_anom = xr.open_zarr(anom_path)
        split = daily_anom["split"].astype(str)

        # Normalizer fit on raw TRAIN daily anomalies -- denser windowing must not alter this.
        train_daily = daily_anom.sel(time=split == "train").drop_vars("split")
        normalizer = fit_normalizer(train_daily, self.cfg)
        self.normalizer = normalizer

        standardized = _standardize(daily_anom.drop_vars("split"), normalizer)
        standardized = standardized.assign_coords(split=("time", split.values))

        self.latitude = standardized.latitude.values
        self.lon = standardized.longitude.values
        self.grid = (standardized.sizes["latitude"], standardized.sizes["longitude"])

        stride = int(getattr(self.cfg.data, "train_stride_days", 1))

        arrays = {}
        for name in _SPLITS:
            sub = standardized.sel(time=standardized.split == name).drop_vars("split")
            if name in ("train", "val"):
                # Dense daily-strided windows for better coverage without leakage.
                arrays[name] = daily_init_weekly_windows(sub, self.cfg, stride_days=stride)
            else:
                # W-MON bins kept for test so eval metrics stay directly comparable.
                weekly = daily_to_weekly_mean(sub)
                arrays[name] = assemble_arrays(weekly, self.cfg)

        self.train_dataset = S2SDataset(arrays["train"])
        self.val_dataset = S2SDataset(arrays["val"])
        self.test_dataset = S2SDataset(arrays["test"])

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=shuffle,
            num_workers=int(self.cfg.train.num_workers),
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
