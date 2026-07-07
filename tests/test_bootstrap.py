"""Fix 2 (C2): paired bootstrap CIs for CRPSS. Synthetic, torch-free."""
import numpy as np
import pytest

from s2s.eval.bootstrap import (
    block_bootstrap_crpss,
    crpss_by_year,
    crpss_from_samples,
    moving_block_indices,
    year_bootstrap_crpss,
)
from s2s.eval.metrics import crpss as crpss_scalar


def test_crpss_from_samples_matches_scalar_crpss():
    rng = np.random.default_rng(0)
    model_s = np.abs(rng.normal(1.0, 0.2, size=200))
    ref_s = np.abs(rng.normal(1.3, 0.2, size=200))
    got = crpss_from_samples(model_s, ref_s)
    want = crpss_scalar(float(model_s.mean()), float(ref_s.mean()))
    assert abs(got - want) < 1e-12


def test_moving_block_indices_valid_and_full_length():
    rng = np.random.default_rng(1)
    for n, bl in [(256, 8), (100, 13), (5, 8)]:  # bl>n clamps to n
        idx = moving_block_indices(n, bl, rng)
        assert idx.shape == (n,)
        assert idx.min() >= 0 and idx.max() < n


def test_identical_model_ref_gives_zero_crpss_and_degenerate_ci():
    rng = np.random.default_rng(2)
    s = np.abs(rng.normal(1.0, 0.3, size=256))
    out = block_bootstrap_crpss(s, s, block_len=8, n_boot=1000, seed=3)
    assert out["point"] == 0.0
    assert abs(out["ci_lo"]) < 1e-9 and abs(out["ci_hi"]) < 1e-9


def test_strong_signal_ci_excludes_zero():
    """Model CRPS uniformly ~40% of reference -> CRPSS ~0.6, CI must exclude 0."""
    rng = np.random.default_rng(4)
    ref_s = np.abs(rng.normal(1.0, 0.1, size=256)) + 0.5
    model_s = 0.4 * ref_s
    out = block_bootstrap_crpss(model_s, ref_s, block_len=8, n_boot=2000, seed=5)
    assert out["ci_lo"] > 0.0
    assert out["ci_lo"] <= out["point"] <= out["ci_hi"]


def test_no_signal_ci_includes_zero():
    """model == ref in distribution (paired, tiny symmetric noise) -> CI spans 0."""
    rng = np.random.default_rng(6)
    base = np.abs(rng.normal(1.0, 0.2, size=256)) + 0.3
    noise = rng.normal(0.0, 0.05, size=256)
    model_s = base + noise
    ref_s = base - noise  # symmetric, so mean diff ~ 0
    out = block_bootstrap_crpss(model_s, ref_s, block_len=8, n_boot=3000, seed=7)
    assert out["ci_lo"] < 0.0 < out["ci_hi"]


def test_block_bootstrap_reproducible():
    rng = np.random.default_rng(8)
    model_s = np.abs(rng.normal(1.0, 0.2, size=128))
    ref_s = np.abs(rng.normal(1.2, 0.2, size=128))
    a = block_bootstrap_crpss(model_s, ref_s, block_len=8, n_boot=500, seed=11)
    b = block_bootstrap_crpss(model_s, ref_s, block_len=8, n_boot=500, seed=11)
    assert a == b


def test_year_bootstrap_runs_and_orders():
    rng = np.random.default_rng(9)
    n = 260
    years = np.repeat([2018, 2019, 2020, 2021, 2022], n // 5)
    ref_s = np.abs(rng.normal(1.0, 0.1, size=len(years))) + 0.5
    model_s = 0.5 * ref_s
    out = year_bootstrap_crpss(model_s, ref_s, years, n_boot=1000, seed=1)
    assert out["n_years"] == 5
    assert out["ci_lo"] <= out["point"] <= out["ci_hi"]


def test_crpss_by_year_partitions_samples():
    rng = np.random.default_rng(10)
    years = np.repeat([2018, 2019, 2020], 10)
    ref_s = np.abs(rng.normal(1.0, 0.1, size=30)) + 0.5
    model_s = 0.6 * ref_s
    rows = crpss_by_year(model_s, ref_s, years)
    assert [r["year"] for r in rows] == [2018, 2019, 2020]
    assert sum(r["n_samples"] for r in rows) == 30
    assert all(r["crpss_vs_prob"] > 0 for r in rows)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        block_bootstrap_crpss(np.ones(10), np.ones(9))
