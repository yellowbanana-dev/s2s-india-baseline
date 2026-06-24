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

    Empirical CRPS (Gneiting & Raftery 2007):
        CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent draws from the ensemble. For a single-member
    (deterministic) forecast the second term is 0 and CRPS reduces to |f - y|.
    """
    forecast_members = np.asarray(forecast_members, dtype=float)
    truth = np.asarray(truth, dtype=float)
    m = forecast_members.shape[0]

    term1 = np.mean(np.abs(forecast_members - truth[np.newaxis, ...]), axis=0)
    if m == 1:
        return term1

    diff = forecast_members[:, np.newaxis, ...] - forecast_members[np.newaxis, :, ...]
    term2 = np.mean(np.abs(diff), axis=(0, 1))
    return term1 - 0.5 * term2


def acc(forecast: np.ndarray, truth: np.ndarray) -> float:
    """Anomaly correlation coefficient over the India box (higher is better).

    Uncentered correlation (standard ACC convention: inputs are already
    anomalies, so no additional mean-subtraction is applied).
    """
    f = np.asarray(forecast, dtype=float).ravel()
    o = np.asarray(truth, dtype=float).ravel()
    mask = np.isfinite(f) & np.isfinite(o)
    f, o = f[mask], o[mask]
    if f.size == 0:
        return float("nan")
    denom = np.sqrt(np.sum(f**2) * np.sum(o**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(f * o) / denom)


def rmse(forecast: np.ndarray, truth: np.ndarray) -> float:
    """Latitude-weighted RMSE of the ensemble mean.

    Latitude weighting (if any) must be applied by the caller before this
    function sees the arrays -- e.g. multiply both forecast and truth by
    sqrt(cos(lat)) so that (f - o)**2 carries the weight correctly.
    """
    f = np.asarray(forecast, dtype=float).ravel()
    o = np.asarray(truth, dtype=float).ravel()
    mask = np.isfinite(f) & np.isfinite(o)
    f, o = f[mask], o[mask]
    if f.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((f - o) ** 2)))


def reliability(forecast_members: np.ndarray, truth: np.ndarray, n_bins: int = 10):
    """Calibration diagram data: nominal exceedance probability vs observed frequency.

    For each of `n_bins` evenly spaced ensemble-rank thresholds, compares the
    nominal probability P(truth <= threshold) implied by the ensemble rank to
    the actually observed frequency. A well-calibrated ensemble lies on y = x.

    Returns (nominal_probs, observed_freqs), each shape (n_bins,).
    """
    f = np.asarray(forecast_members, dtype=float)
    o = np.asarray(truth, dtype=float).ravel()
    m = f.shape[0]
    f_flat = f.reshape(m, -1)

    sorted_f = np.sort(f_flat, axis=0)
    nominal = (np.arange(n_bins) + 0.5) / n_bins
    observed = np.empty(n_bins)
    for i, p in enumerate(nominal):
        idx = min(int(p * m), m - 1)
        threshold = sorted_f[idx]
        observed[i] = np.mean(o <= threshold)
    return nominal, observed
