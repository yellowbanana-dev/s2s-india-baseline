"""Evaluation metrics (task #5). Reused UNCHANGED for the whole thesis.

VERIFY EACH METRIC against a hand-computed tiny example before trusting it.
A CRPS/ACC off by a normalization makes everything look great and never errors.

All metrics are scored over the India box (cfg.data.eval_box) and
latitude-weighted where appropriate. Temp ~ Gaussian; precip is skewed — report
separately, never pool the two into one number.
"""
from __future__ import annotations
import numpy as np


def crps_ensemble(forecast_members: np.ndarray, truth: np.ndarray,
                  fair: bool = False) -> np.ndarray:
    """Continuous Ranked Probability Score from an ensemble. PRIMARY metric.

    forecast_members: (M, ...)   truth: (...)
    Returns per-gridpoint CRPS (lower is better).

    Empirical CRPS (Gneiting & Raftery 2007):
        CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent draws from the ensemble. For a single-member
    (deterministic) forecast the second term is 0 and CRPS reduces to |f - y|.

    fair=False (default): biased spread estimator 1/(2 M^2) ΣΣ|x_i-x_j|.
    fair=True: UNBIASED estimator 1/(2 M(M-1)) ΣΣ (Ferro 2014) — needed when
    comparing ensembles of DIFFERENT sizes on equal footing (e.g. a 16-member
    model vs a ~240-member climatology), since the biased form's M-dependence
    otherwise disadvantages the smaller ensemble.
    """
    forecast_members = np.asarray(forecast_members, dtype=float)
    truth = np.asarray(truth, dtype=float)
    m = forecast_members.shape[0]

    term1 = np.mean(np.abs(forecast_members - truth[np.newaxis, ...]), axis=0)
    if m == 1:
        return term1

    diff = forecast_members[:, np.newaxis, ...] - forecast_members[np.newaxis, :, ...]
    spread_sum = np.abs(diff).sum(axis=(0, 1))
    denom = m * (m - 1) if fair else m * m
    return term1 - 0.5 * (spread_sum / denom)


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


def crpss(crps_model: float, crps_reference: float) -> float:
    """CRPS skill score (higher better; >0 == model beats the reference).

        CRPSS = 1 - CRPS_model / CRPS_reference

    Phase-B's honest bar uses a *probabilistic* climatology reference (a
    week-of-year-windowed pool of train weekly anomalies), unlike Phase A's
    deterministic zero-anomaly climatology (CRPS == MAE). Pure ratio of two
    already-reduced scalars.
    """
    if crps_reference == 0 or not np.isfinite(crps_reference):
        return float("nan")
    return float(1.0 - crps_model / crps_reference)


def rank_histogram(forecast_members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Talagrand rank histogram counts (ensemble calibration; flat == calibrated).

    forecast_members: (M, ...)   truth: (...)
    For each verification point the observation's rank among the M sorted members
    falls in one of M+1 bins; ties broken randomly (fixed seed) so a calibrated
    ensemble is flat. A U-shape => UNDER-dispersed (the Phase-A failure); a
    centre-heavy dome => over-dispersed. Returns length-(M+1) counts over all
    finite points.
    """
    f = np.asarray(forecast_members, dtype=float)
    o = np.asarray(truth, dtype=float)
    m = f.shape[0]
    f_flat = f.reshape(m, -1).T          # (P, M)
    o_flat = o.reshape(-1)               # (P,)
    finite = np.isfinite(o_flat) & np.isfinite(f_flat).all(axis=1)
    f_flat, o_flat = f_flat[finite], o_flat[finite]
    rng = np.random.default_rng(0)       # deterministic tie-breaking
    below = (f_flat < o_flat[:, None]).sum(axis=1)
    ties = (f_flat == o_flat[:, None]).sum(axis=1)
    rank = below + (rng.random(len(o_flat)) * (ties + 1)).astype(int)
    rank = np.clip(rank, 0, m)
    return np.bincount(rank, minlength=m + 1)[: m + 1]


def spread_error_ratio(forecast_members: np.ndarray, truth: np.ndarray) -> float:
    """Spread-skill ratio (calibrated ~ 1; <1 under-dispersed, >1 over-dispersed).

    ratio = sqrt(((M+1)/M) * mean ensemble variance) / RMSE(ensemble mean).
    A scalar summary of the rank histogram. Latitude weighting (if wanted) must be
    pre-applied by the caller, consistent with rmse() above.
    """
    f = np.asarray(forecast_members, dtype=float)
    o = np.asarray(truth, dtype=float)
    m = f.shape[0]
    ens_mean = f.mean(axis=0)
    ens_var = f.var(axis=0, ddof=1) if m > 1 else np.zeros_like(ens_mean)
    fm = ens_mean.ravel(); ov = o.ravel(); vv = ens_var.ravel()
    mask = np.isfinite(fm) & np.isfinite(ov) & np.isfinite(vv)
    fm, ov, vv = fm[mask], ov[mask], vv[mask]
    if fm.size == 0:
        return float("nan")
    rmse_mean = np.sqrt(np.mean((fm - ov) ** 2))
    if rmse_mean == 0:
        return float("nan")
    spread = np.sqrt(((m + 1) / m) * np.mean(vv))
    return float(spread / rmse_mean)


def inflate_ensemble_spread(forecast_members: np.ndarray, alpha: float) -> np.ndarray:
    """Rescale ensemble deviations about the ensemble mean by `alpha` (post-hoc diagnostic).

        members -> ens_mean + alpha * (members - ens_mean)

    The ensemble MEAN is untouched, so RMSE(mean) is unchanged and `spread_error_ratio`
    scales EXACTLY by `alpha` (spread is a pure function of the deviations). Operating
    pointwise over the member axis, it commutes with spatial subsetting: inflating then
    boxing == boxing then inflating.

    THIS DOES NOT PRODUCE A BETTER MODEL. It answers one attribution question: how much of
    a CRPS gap would remain if the ensemble were perfectly dispersed, holding the ensemble
    mean (and hence the deterministic skill) fixed? Any CRPS improvement it shows is an
    UPPER BOUND on the share of the gap attributable to under-dispersion.

    M < 2, or non-finite alpha, returns an unmodified copy.
    """
    f = np.asarray(forecast_members, dtype=float)
    # alpha == 1 short-circuits so the no-op is BIT-exact: mean + 1*(f - mean) otherwise
    # differs from f by float round-off (subtract-then-re-add the mean).
    if f.shape[0] < 2 or not np.isfinite(alpha) or float(alpha) == 1.0:
        return f.copy()
    mean = f.mean(axis=0, keepdims=True)
    return mean + float(alpha) * (f - mean)


def spread_inflation_factor(forecast_members: np.ndarray, truth: np.ndarray,
                            target_ser: float = 1.0) -> float:
    """alpha such that spread_error_ratio(inflate_ensemble_spread(m, alpha), truth) == target_ser.

    Since inflation scales spread linearly and leaves the ensemble mean (and RMSE) fixed,
    alpha = target_ser / current_ser exactly. Returns NaN when the current ratio is not
    finite/positive (e.g. M==1, or a degenerate zero-error field).
    """
    ser = spread_error_ratio(forecast_members, truth)
    if not np.isfinite(ser) or ser <= 0:
        return float("nan")
    return float(target_ser) / ser


def reliability_curve(prob_forecast: np.ndarray, event_truth: np.ndarray, n_bins: int = 10):
    """Reliability-diagram points for a binary event (forecast prob vs obs freq).

    prob_forecast : ensemble probability of the event per point (0..1), any shape.
    event_truth   : 0/1 observed event per point, same shape.
    Returns (bin_centers, observed_freq, bin_counts) over n_bins equal-width bins;
    perfectly reliable => observed_freq == bin_centers. NaN where a bin is empty.
    Used for P(anomaly>0), P(upper tercile), and the India-context absolute
    weekly-mean events P(T2m>40C), P(precip>50mm/day).
    """
    p = np.asarray(prob_forecast, dtype=float).reshape(-1)
    y = np.asarray(event_truth, dtype=float).reshape(-1)
    finite = np.isfinite(p) & np.isfinite(y)
    p, y = p[finite], y[finite]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    obs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = idx == b
        counts[b] = int(sel.sum())
        if counts[b]:
            obs[b] = float(y[sel].mean())
    return centers, obs, counts


def event_probability(forecast_members: np.ndarray, threshold: float, comparison: str = "gt") -> np.ndarray:
    """Ensemble probability the field exceeds (gt) or falls below (lt) a threshold.

    forecast_members: (M, ...) -> returns (...) fraction of members satisfying it.
    """
    f = np.asarray(forecast_members, dtype=float)
    if comparison == "gt":
        hit = f > threshold
    elif comparison == "lt":
        hit = f < threshold
    else:
        raise ValueError(f"unknown comparison: {comparison!r}")
    return hit.mean(axis=0)
