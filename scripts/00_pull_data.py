"""Stage 1 entrypoint: pull ERA5 @ 5.625 deg and verify it (task #2).

Run from the repo root, in an environment with network access to Google Cloud
Storage (your HPC / laptop — NOT an air-gapped node):

    python scripts/00_pull_data.py

To work on a fast subset first, set dev_years in configs/data/era5_india.yaml
or override on the CLI:

    python scripts/00_pull_data.py data.dev_years=[2015,2018]
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from s2s.data.download import pull_era5, verify_pull


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    ds = pull_era5(cfg)        # lazy open over the network (no full download)
    verify_pull(ds, cfg)       # HUMAN-OWNED GATE: read the printed summary
    print(ds)
    # Once dev_years is set, optionally cache locally for offline Stage 2:
    # ds.to_zarr(f"{cfg.data.paths.raw}/era5_5625_subset.zarr", mode="w")


if __name__ == "__main__":
    main()
