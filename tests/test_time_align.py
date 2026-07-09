"""Fix 1 (m1): eval init-time reconstruction must match assemble_arrays exactly.

Pure synthetic data, no cluster / no torch. Guards against the previous silent
`init_times[-n_samples:]` fallback that masked the missing (history_weeks - 1)
leading-week drop.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from omegaconf import OmegaConf

from s2s.data.assemble import assemble_arrays
from s2s.eval.time_align import reconstruct_init_times, valid_init_index


def _make_cfg(history_weeks, lead_weeks):
    return OmegaConf.create(
        {
            "data": {
                "history_weeks": history_weeks,
                "lead_weeks": list(lead_weeks),
                "variables": {
                    "targets": {"surface": ["2m_temperature", "total_precipitation_24hr"]},
                    "predictors": {
                        "surface": ["sea_surface_temperature"],
                        "levels": {"geopotential": [500], "u_component_of_wind": [850, 200]},
                    },
                },
            }
        }
    )


def _make_weekly(n_weeks, lat=3, lon=5, seed=0):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2018-01-01", periods=n_weeks, freq="7D")
    shape = (n_weeks, lat, lon)
    names = [
        "2m_temperature", "total_precipitation_24hr", "sea_surface_temperature",
        "geopotential_500", "u_component_of_wind_850", "u_component_of_wind_200",
    ]
    dv = {n: (("time", "latitude", "longitude"), rng.normal(size=shape)) for n in names}
    return xr.Dataset(
        dv,
        coords={
            "time": time,
            "latitude": np.linspace(-60, 60, lat),
            "longitude": np.linspace(0, 270, lon),
        },
    )


# The real config plus a couple of shapes that exercise history_weeks > 1.
CASES = [
    (2, (1, 2, 3, 4, 5, 6), 30),   # production config
    (2, (1, 2, 3), 12),
    (1, (1, 2), 8),                # history_weeks-1 == 0: front drop is zero
    (4, (1, 2, 3, 4, 5, 6), 25),   # bigger history lookback
]


@pytest.mark.parametrize("history_weeks,lead_weeks,n_weeks", CASES)
def test_reconstruct_matches_assemble_exactly(history_weeks, lead_weeks, n_weeks):
    cfg = _make_cfg(history_weeks, lead_weeks)
    weekly = _make_weekly(n_weeks)
    assembled = assemble_arrays(weekly, cfg)
    assembled_time = assembled["time"]

    recon = reconstruct_init_times(weekly.time.values, history_weeks, max(lead_weeks))

    # Same count, same first/last, and full element-wise equality.
    assert len(recon) == len(assembled_time) == assembled["inputs"].shape[0]
    assert recon[0] == assembled_time[0]
    assert recon[-1] == assembled_time[-1]
    np.testing.assert_array_equal(recon, assembled_time)


@pytest.mark.parametrize("history_weeks,lead_weeks,n_weeks", CASES)
def test_naive_end_trim_is_wrong_when_history_gt_1(history_weeks, lead_weeks, n_weeks):
    """The OLD approach (trim only max_lead off the end) over-keeps the leading
    (history_weeks - 1) inits. Confirm the fix diverges from that bug exactly there."""
    weekly_time = _make_weekly(n_weeks).time.values
    max_lead = max(lead_weeks)
    naive = weekly_time[: len(weekly_time) - max_lead]        # buggy reconstruction
    recon = reconstruct_init_times(weekly_time, history_weeks, max_lead)

    assert len(naive) - len(recon) == history_weeks - 1
    if history_weeks > 1:
        # naive starts (history_weeks-1) weeks too early; recon starts at the right week.
        assert naive[0] != recon[0]
        np.testing.assert_array_equal(naive[history_weeks - 1:], recon)
    else:
        np.testing.assert_array_equal(naive, recon)


def test_valid_init_index_endpoints():
    idx = valid_init_index(n_time=30, history_weeks=2, max_lead=6)
    assert idx[0] == 1 and idx[-1] == 23 and len(idx) == 23


def test_raises_when_no_valid_week():
    with pytest.raises(ValueError):
        reconstruct_init_times(np.arange(5), history_weeks=2, max_lead=6)
