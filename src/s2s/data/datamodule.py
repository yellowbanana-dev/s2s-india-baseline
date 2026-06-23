"""Stage 2c - Lightning DataModule / Dataset (task #3).

Yields (input, target) tensors:
  input  : (C_in,  lat, lon)  global normalized anomalies + history weeks
  target : (lead, C_out, lat, lon)  weekly-mean anomalies, weeks 1-6

Shuffle samples but NEVER across split boundaries. Reads lazily from Zarr.
"""
from __future__ import annotations
import lightning as L
from torch.utils.data import Dataset


class S2SDataset(Dataset):
    """Maps an initialization time -> (global input anomalies, weekly-mean targets)."""

    def __init__(self, anomalies, clim, normalizer, cfg, split: str):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class S2SDataModule(L.LightningDataModule):
    """Wires Stage 1 + 2 into train/val/test loaders for the trainer."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        """Stage 1 pull + Stage 2 build, cached to data/processed (rank-0 only)."""
        raise NotImplementedError

    def setup(self, stage: str | None = None):
        raise NotImplementedError

    def train_dataloader(self): raise NotImplementedError
    def val_dataloader(self):   raise NotImplementedError
    def test_dataloader(self):  raise NotImplementedError
