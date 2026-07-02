"""Phase-B Stage-B tests: fair-CRPS loss + member-producing forward.

Fair CRPS is the training objective that must reward calibrated spread rather than
punish it (the biased estimator collapses the ensemble). These checks pin the math
and the ensemble plumbing so a regression can't silently reintroduce under-dispersion.
"""
import numpy as np
import pytest
import torch

from s2s.models.lit import fair_crps


def _lw(H):
    w = np.cos(np.deg2rad(np.linspace(-87, 87, H)))
    return torch.tensor((w / w.mean()), dtype=torch.float32).view(1, 1, 1, H, 1)


def test_fair_crps_hand_value():
    # members {1,3}, truth 2: term1=1, fair spread E|X-X'|=2 => CRPS = 1 - 0.5*2 = 0.
    members = torch.tensor([1.0, 3.0]).view(1, 2, 1, 1, 1, 1)
    target = torch.tensor([2.0]).view(1, 1, 1, 1, 1)
    lw = torch.ones(1, 1, 1, 1, 1)
    assert abs(float(fair_crps(members, target, lw)) - 0.0) < 1e-6


def test_fair_crps_matches_bruteforce_pairwise():
    # The sorted-identity spread term must equal the explicit pairwise double sum.
    rng = torch.Generator().manual_seed(0)
    members = torch.randn(4, 8, 6, 2, 32, 64, generator=rng)
    target = torch.randn(4, 6, 2, 32, 64, generator=rng)
    lw = _lw(32)
    got = float(fair_crps(members, target, lw))

    M = members.shape[1]
    term1 = (members - target.unsqueeze(1)).abs().mean(dim=1)
    pw = (members.unsqueeze(1) - members.unsqueeze(2)).abs().sum(dim=(1, 2)) / (2 * M * (M - 1))
    brute = float(((term1 - pw) * lw).mean())
    assert abs(got - brute) < 1e-4


def test_fair_crps_rewards_calibration_over_collapse():
    # Over many truths drawn from N(0,1): a calibrated ensemble (draws from N(0,1))
    # must score LOWER expected fair CRPS than one collapsed to the mean. The biased
    # estimator would reverse/erase this; this guards against reintroducing it.
    rng = torch.Generator().manual_seed(0)
    M, T = 8, 3000
    lw = torch.ones(1, 1, 1, 1, 1)
    cal, coll = 0.0, 0.0
    ys = torch.randn(T, generator=rng)
    for t in range(T):
        y = ys[t].view(1, 1, 1, 1, 1)
        cal_mem = torch.randn(M, generator=rng).view(1, M, 1, 1, 1, 1)
        cal += float(fair_crps(cal_mem, y, lw))
        coll += float(fair_crps(torch.zeros(1, M, 1, 1, 1, 1), y, lw))
    assert cal / T < coll / T


def test_fair_crps_requires_two_members():
    with pytest.raises(ValueError):
        fair_crps(torch.zeros(1, 1, 1, 1, 1, 1), torch.zeros(1, 1, 1, 1, 1), torch.ones(1, 1, 1, 1, 1))
