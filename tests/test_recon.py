"""Fix 7 (M1): HEALPix round-trip gain/bias correction. Torch-free."""
import numpy as np

from s2s.eval.recon import affine_correct, latweighted_rmse


def _grid(seed=0, H=16, W=24):
    lat = np.linspace(80, -80, H)
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(H, W))
    return lat, f


def test_affine_recovers_exact_gain_bias():
    """recon = a*f + b exactly => correction recovers (a,b) and residual ~0."""
    lat, f = _grid()
    a_true, b_true = 3.7, -1.2
    recon = a_true * f + b_true
    a, b, corr = affine_correct(recon, f, lat)
    assert abs(a - 1 / a_true) < 1e-9   # inverse map back onto truth
    assert np.allclose(corr, f, atol=1e-9)
    assert latweighted_rmse(corr, f, lat) < 1e-9


def test_correction_never_increases_rmse():
    lat, f = _grid(1)
    recon = 2.0 * f + 5.0 + 0.01 * np.random.default_rng(2).normal(size=f.shape)
    raw = latweighted_rmse(recon, f, lat)
    _, _, corr = affine_correct(recon, f, lat)
    corrected = latweighted_rmse(corr, f, lat)
    assert corrected <= raw + 1e-12
    assert corrected < raw  # a genuine gain mismatch is removed


def test_mask_restricts_fit_and_score():
    lat, f = _grid(3, H=16, W=24)
    lat_mask = (lat >= 5) & (lat <= 40)
    lon_mask = np.zeros(24, bool); lon_mask[8:16] = True
    # distort only OUTSIDE the box; box should reconstruct near-perfectly after fit
    recon = 1.5 * f.copy() + 0.3
    recon[~lat_mask, :] += 10.0
    _, _, corr = affine_correct(recon, f, lat, lat_mask, lon_mask)
    assert latweighted_rmse(corr, f, lat, lat_mask, lon_mask) < 1e-9


def test_latweighted_rmse_matches_manual():
    lat = np.array([60.0, 0.0])
    # constant error of 1 everywhere -> weighted RMSE is exactly 1 regardless of weights
    pred = np.ones((2, 2))
    truth = np.zeros((2, 2))
    assert abs(latweighted_rmse(pred, truth, lat) - 1.0) < 1e-12
    # err=2 at lat=60 (w=0.5, 2 cols), err=0 at lat=0 (w=1): num=0.5*(4+4)=4, den=3 -> sqrt(4/3)
    pred2 = np.array([[2.0, 2.0], [0.0, 0.0]])
    assert abs(latweighted_rmse(pred2, truth, lat) - np.sqrt(4.0 / 3.0)) < 1e-12


def test_singular_falls_back_to_identity():
    lat, f = _grid(4)
    const = np.zeros_like(f)  # recon has zero variance -> singular normal eqs
    a, b, corr = affine_correct(const, f, lat)
    assert (a, b) == (1.0, 0.0)


def test_interp_state_subset_rekeys_and_filters():
    from s2s.eval.recon import interp_state_subset
    sd = {
        "model.transformer.interp_to_hp.to_q.weight": 1,
        "model.transformer.interp_to_hp.to_kv.weight": 2,
        "model.transformer.interp_to_ll.to_o.weight": 3,
        "model.transformer.encoder_stages.0.attn.to_q.weight": 4,  # excluded
        "model.transformer.postprocess.0.weight": 5,               # excluded
    }
    out = interp_state_subset(sd)
    assert set(out) == {
        "interp_to_hp.to_q.weight",
        "interp_to_hp.to_kv.weight",
        "interp_to_ll.to_o.weight",
    }
    assert out["interp_to_hp.to_q.weight"] == 1
