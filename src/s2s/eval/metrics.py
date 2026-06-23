"""Evaluation metrics (task #5). Reused UNCHANGED for the whole thesis.

VERIFY EACH METRIC against a hand-computed tiny example before trusting it.
A CRPS/ACC off by a normalization makes everything look great and never errors.

All metrics are scored over the India box (cfg.data.eval_box) and
latitude-weighted where appropriate. Temp ~ Gaussian; precip is skewed — report
separately, never pool the two into one number.
"""
from __future__ import annotations
import numpy as np


def crps_ensemble(forecast_members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Continuous Ranked Probability Score from an ensemble. PRIMARY metric.

    forecast_members: (M, ...)   truth: (...)
    Returns per-gridpoint CRPS (lower is better).
    """
    raise NotImplementedError


def acc(forecast: np.ndarray, truth: np.ndarray) -> float:
    """Anomaly correlation coefficient over the India box (higher is better)."""
    raise NotImplementedError


def rmse(forecast: np.ndarray, truth: np.ndarray) -> float:
    """Latitude-weighted RMSE of the ensemble mean."""
    raise NotImplementedError


def reliability(forecast_members: np.ndarray, truth: np.ndarray, n_bins: int = 10):
    """Calibration diagram data: predicted prob vs observed frequency."""
    raise NotImplementedError
