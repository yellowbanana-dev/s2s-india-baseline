"""Hydra entrypoint: train G1 + P2 ensemble (task #7).

Run:  python -m s2s.train  [overrides...]
One experiment = one config + one commit + one W&B run.
"""
from __future__ import annotations
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO:
    #   1. seed everything (cfg.seed)
    #   2. build S2SDataModule(cfg)
    #   3. train cfg.train.ensemble.n_members members (different seeds + IC perturbation)
    #   4. log to W&B; checkpoint each member
    #   5. run eval harness over the India box; report vs baselines
    raise NotImplementedError


if __name__ == "__main__":
    main()
