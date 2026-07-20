"""Paired bootstrap confidence intervals for CRPSS (Fix 2 / C2). No retraining.

CRPSS is an aggregate ratio of means:  CRPSS = 1 - mean(crps_model) / mean(crps_ref).
To put a 95% CI on it without refitting anything, we resample the *paired* per-sample
CRPS values (model_s, ref_s) — the model members already forecast every test init, and
the probabilistic-climatology reference already produces one CRPS per init — and
recompute the ratio each time.

Weekly test inits are serially correlated, so an i.i.d. bootstrap would understate the
CI. We use a moving-block bootstrap (default block length 8 weeks) as the primary
method, and a resample-by-calendar-year bootstrap as a coarser, autocorrelation-robust
sensitivity. Both are paired: the SAME resampled indices are applied to model and
reference, so the CI is on their difference, not on two independent draws.
"""
from __future__ import annotations

import numpy as np


def crpss_from_samples(model_s, ref_s) -> float:
    """CRPSS = 1 - mean(model) / mean(ref), NaN-safe. Matches metrics.crpss on the
    aggregate means (crpss(mean(model_s), mean(ref_s)))."""
    m = float(np.nanmean(model_s))
    r = float(np.nanmean(ref_s))
    if r == 0.0:
        return float("nan")
    return 1.0 - m / r


def moving_block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block resample of {0..n-1}: draw ceil(n/block_len) contiguous
    (wrap-around) blocks of length block_len, concatenate, trim to n."""
    if block_len < 1:
        raise ValueError("block_len must be >= 1")
    block_len = min(block_len, n)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + block_len) % n) for s in starts])
    return idx[:n]


def _percentile_ci(stats: np.ndarray, alpha: float):
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def block_bootstrap_crpss(
    model_s, ref_s, block_len: int = 8, n_boot: int = 5000,
    alpha: float = 0.05, seed: int = 0,
) -> dict:
    """Moving-block bootstrap CI for CRPSS over paired per-sample CRPS.

    Returns {point, ci_lo, ci_hi, boot_mean, boot_se, n, block_len, n_boot}.
    point is the estimate on the full sample; ci_* are percentile bounds.
    """
    model_s = np.asarray(model_s, dtype=float)
    ref_s = np.asarray(ref_s, dtype=float)
    if model_s.shape != ref_s.shape or model_s.ndim != 1:
        raise ValueError("model_s and ref_s must be 1-D arrays of equal length")
    n = model_s.size
    rng = np.random.default_rng(seed)
    point = crpss_from_samples(model_s, ref_s)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = moving_block_indices(n, block_len, rng)
        stats[b] = crpss_from_samples(model_s[idx], ref_s[idx])
    lo, hi = _percentile_ci(stats, alpha)
    return {
        "point": point, "ci_lo": lo, "ci_hi": hi,
        "boot_mean": float(np.nanmean(stats)), "boot_se": float(np.nanstd(stats)),
        "n": n, "block_len": min(block_len, n), "n_boot": n_boot,
    }


def year_bootstrap_crpss(
    model_s, ref_s, years, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0,
) -> dict:
    """Resample-by-calendar-year bootstrap CI (sensitivity to block choice).

    Draw len(unique_years) whole years with replacement, pool their sample indices,
    recompute CRPSS. Coarser than the block bootstrap (few years => wide CI), but
    fully robust to within-year autocorrelation.
    """
    model_s = np.asarray(model_s, dtype=float)
    ref_s = np.asarray(ref_s, dtype=float)
    years = np.asarray(years)
    uniq = np.unique(years)
    groups = [np.where(years == y)[0] for y in uniq]
    rng = np.random.default_rng(seed)
    point = crpss_from_samples(model_s, ref_s)
    stats = np.empty(n_boot, dtype=float)
    k = len(uniq)
    for b in range(n_boot):
        pick = rng.integers(0, k, size=k)
        idx = np.concatenate([groups[j] for j in pick])
        stats[b] = crpss_from_samples(model_s[idx], ref_s[idx])
    lo, hi = _percentile_ci(stats, alpha)
    return {
        "point": point, "ci_lo": lo, "ci_hi": hi,
        "boot_mean": float(np.nanmean(stats)), "boot_se": float(np.nanstd(stats)),
        "n_years": int(k), "n_boot": n_boot,
    }


def crpss_by_year(model_s, ref_s, years) -> list[dict]:
    """Point CRPSS within each calendar year separately (no resampling)."""
    model_s = np.asarray(model_s, dtype=float)
    ref_s = np.asarray(ref_s, dtype=float)
    years = np.asarray(years)
    rows = []
    for y in sorted(np.unique(years).tolist()):
        m = years == y
        rows.append({
            "year": int(y),
            "n_samples": int(m.sum()),
            "crpss_vs_prob": crpss_from_samples(model_s[m], ref_s[m]),
        })
    return rows


def paired_delta_crpss_bootstrap(
    model_a, ref_a, model_b, ref_b, block_len: int = 8, n_boot: int = 5000,
    alpha: float = 0.05, seed: int = 0,
) -> dict:
    """Moving-block bootstrap CI on the DIFFERENCE in CRPSS between two models (B - A)
    scored on the SAME test inits and the SAME grid (MAJ-3 / ADR-0007 f3).

    Comparing two independently-computed marginal CIs is the wrong test: overlapping
    marginal CIs do NOT imply the difference is insignificant. Because both models forecast
    the identical set of inits, we apply the SAME resampled block indices to all four
    per-sample arrays, so the sampling noise common to both cancels. That makes this test
    strictly more powerful than eyeballing marginal-CI overlap.

    Returns {delta, ci_lo, ci_hi, boot_se, p_two_sided, crpss_a, crpss_b, n, ...}.
    delta > 0 means model B is more skilful.
    """
    a_m, a_r, b_m, b_r = (np.asarray(x, dtype=float) for x in (model_a, ref_a, model_b, ref_b))
    n = a_m.size
    if any(x.shape != (n,) for x in (a_r, b_m, b_r)):
        raise ValueError("all four per-sample arrays must be 1-D of equal length")
    rng = np.random.default_rng(seed)
    crpss_a = crpss_from_samples(a_m, a_r)
    crpss_b = crpss_from_samples(b_m, b_r)
    point = crpss_b - crpss_a
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = moving_block_indices(n, block_len, rng)
        stats[i] = (crpss_from_samples(b_m[idx], b_r[idx])
                    - crpss_from_samples(a_m[idx], a_r[idx]))
    lo, hi = _percentile_ci(stats, alpha)
    p = 2.0 * min(float(np.mean(stats <= 0.0)), float(np.mean(stats >= 0.0)))
    return {
        "delta": point, "ci_lo": lo, "ci_hi": hi,
        "boot_se": float(np.nanstd(stats)), "p_two_sided": min(1.0, p),
        "crpss_a": crpss_a, "crpss_b": crpss_b,
        "n": n, "block_len": min(block_len, n), "n_boot": n_boot,
    }
