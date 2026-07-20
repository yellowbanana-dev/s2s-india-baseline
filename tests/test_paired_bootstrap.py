"""Paired model-vs-model CRPSS bootstrap (MAJ-3 / ADR-0007 f3)."""
import numpy as np

from s2s.eval.bootstrap import paired_delta_crpss_bootstrap


def _mk(n=400, seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.gamma(2.0, 0.4, size=n) + 0.1
    return rng, ref


def test_identical_models_give_zero_delta_and_ci_spanning_zero():
    rng, ref = _mk()
    model = ref * 0.8
    r = paired_delta_crpss_bootstrap(model, ref, model, ref, n_boot=400, seed=1)
    assert abs(r["delta"]) < 1e-12
    assert r["ci_lo"] <= 0.0 <= r["ci_hi"]


def test_uniformly_better_model_is_detected():
    rng, ref = _mk(seed=2)
    a = ref * 0.90          # A: 10% better than reference
    b = ref * 0.70          # B: 30% better -> clearly more skilful
    r = paired_delta_crpss_bootstrap(a, ref, b, ref, n_boot=400, seed=3)
    assert r["delta"] > 0
    assert r["ci_lo"] > 0                      # CI excludes zero
    assert r["p_two_sided"] < 0.05
    assert r["crpss_b"] > r["crpss_a"]


def test_sign_flips_when_arguments_swap():
    rng, ref = _mk(seed=4)
    a, b = ref * 0.90, ref * 0.70
    fwd = paired_delta_crpss_bootstrap(a, ref, b, ref, n_boot=300, seed=5)
    rev = paired_delta_crpss_bootstrap(b, ref, a, ref, n_boot=300, seed=5)
    assert np.isclose(fwd["delta"], -rev["delta"], atol=1e-12)


def test_pairing_is_more_powerful_than_marginal_cis():
    """The whole point: a small but CONSISTENT per-sample edge is detectable paired even
    when the two models' marginal CRPSS CIs overlap heavily."""
    from s2s.eval.bootstrap import block_bootstrap_crpss
    rng = np.random.default_rng(6)
    ref = rng.gamma(2.0, 0.4, size=500) + 0.1
    u = rng.uniform(0.3, 1.5, size=500)            # per-sample spread -> WIDE marginal CIs
    a = ref * u
    b = a * 0.98                                   # consistent 2% edge, sample-by-sample
    ma = block_bootstrap_crpss(a, ref, n_boot=400, seed=7)
    mb = block_bootstrap_crpss(b, ref, n_boot=400, seed=7)
    assert not (mb["ci_lo"] > ma["ci_hi"])         # marginal CIs overlap -> inconclusive
    pr = paired_delta_crpss_bootstrap(a, ref, b, ref, n_boot=400, seed=7)
    assert pr["ci_lo"] > 0                         # paired test resolves it
