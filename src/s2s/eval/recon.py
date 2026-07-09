"""Reconstruction-error helpers for the HEALPix round-trip diagnostic (Fix 7/M1).

Torch-free so the numerics are unit-testable without a model.

The lon/lat <-> HEALPix interpolators (CrossAttentionInterpolate) are LEARNED,
not a fixed geometric operation. On an untrained network their linear maps
(to_q/to_kv/to_o) are random, so the round-trip output differs from the input by
an essentially arbitrary affine transform (gain + bias). Raw RMSE then measures
that random gain, not interpolation geometry. `affine_correct` removes the best
global gain/bias by least squares first, so the residual RMSE reflects spatial
distortion (neighbour blurring) rather than a trivial scale mismatch.
"""
from __future__ import annotations

import numpy as np


def _weights_2d(lat_deg, shape, lat_mask=None, lon_mask=None):
    w = np.cos(np.deg2rad(np.asarray(lat_deg, dtype=np.float64))).reshape(-1, 1)
    w = np.broadcast_to(w, shape).copy()
    mask = np.ones(shape, dtype=bool)
    if lat_mask is not None:
        mask &= np.asarray(lat_mask, bool).reshape(-1, 1)
    if lon_mask is not None:
        mask &= np.asarray(lon_mask, bool).reshape(1, -1)
    return w, mask


def latweighted_rmse(pred, truth, lat_deg, lat_mask=None, lon_mask=None) -> float:
    """Latitude-weighted RMSE over (lat, lon), optionally restricted by masks."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    w, mask = _weights_2d(lat_deg, truth.shape, lat_mask, lon_mask)
    err2 = (pred - truth) ** 2
    num = float((err2 * w)[mask].sum())
    den = float(w[mask].sum())
    return float(np.sqrt(num / den))


def affine_correct(recon, truth, lat_deg, lat_mask=None, lon_mask=None):
    """Least-squares gain/bias: a, b minimising the lat-weighted
    Σ w (a*recon + b - truth)^2 over the (optionally masked) region.

    Returns (a, b, corrected_recon) where corrected_recon = a*recon + b (applied
    everywhere, using the region-fitted a, b). Isolates spatial distortion from the
    learned interpolator's arbitrary global scale/offset.
    """
    recon = np.asarray(recon, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    w, mask = _weights_2d(lat_deg, truth.shape, lat_mask, lon_mask)
    r = recon[mask]
    t = truth[mask]
    wf = w[mask]
    # Weighted normal equations for [a, b].
    Sww = wf.sum()
    Swr = (wf * r).sum()
    Swrr = (wf * r * r).sum()
    Swt = (wf * t).sum()
    Swrt = (wf * r * t).sum()
    A = np.array([[Swrr, Swr], [Swr, Sww]], dtype=np.float64)
    y = np.array([Swrt, Swt], dtype=np.float64)
    try:
        a, b = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        a, b = 1.0, 0.0
    return float(a), float(b), a * recon + b


def interp_state_subset(state_dict, submodules=("interp_to_hp", "interp_to_ll")) -> dict:
    """Pull the learned interpolator tensors out of a (Lightning) checkpoint
    state_dict, re-keyed relative to the interpolator submodule so they load into
    a bare Transformer via load_state_dict(strict=False).

    e.g. 'model.transformer.interp_to_hp.to_q.weight' -> 'interp_to_hp.to_q.weight'.
    Shapes of these modules depend only on dim/num_heads/k_neighbors, not the
    variable count, so a 1-channel diagnostic net accepts trained weights.
    """
    out = {}
    for k, v in state_dict.items():
        for sub in submodules:
            if sub in k:
                out[k[k.index(sub):]] = v
                break
    return out
