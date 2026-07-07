"""Stage 2c assemble tests (task #7 guardrails). Synthetic data, no network."""
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf

from s2s.data.assemble import assemble_arrays, in_out_channels


def _make_cfg(history_weeks=2, lead_weeks=(1, 2, 3)):
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


def _make_weekly(n_weeks=10, lat=4, lon=8, seed=0):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2010-01-04", periods=n_weeks, freq="7D")
    shape = (n_weeks, lat, lon)
    data_vars = {
        "2m_temperature": (("time", "latitude", "longitude"), rng.normal(size=shape)),
        "total_precipitation_24hr": (("time", "latitude", "longitude"), rng.normal(size=shape)),
        "sea_surface_temperature": (("time", "latitude", "longitude"), rng.normal(size=shape)),
        "geopotential_500": (("time", "latitude", "longitude"), rng.normal(size=shape)),
        "u_component_of_wind_850": (("time", "latitude", "longitude"), rng.normal(size=shape)),
        "u_component_of_wind_200": (("time", "latitude", "longitude"), rng.normal(size=shape)),
    }
    return xr.Dataset(
        data_vars, coords={"time": time, "latitude": np.linspace(-60, 60, lat), "longitude": np.linspace(0, 270, lon)}
    )


def test_in_out_channels_matches_config():
    """6 input vars (2 targets + 4 predictors) x 2 history weeks + 2 seasonal (cos,sin)."""
    cfg = _make_cfg(history_weeks=2)   # no SST extra lags configured
    in_channels, out_channels = in_out_channels(cfg)
    assert in_channels == 6 * 2 + 2
    assert out_channels == 2


def test_sst_extra_lags_add_channels_and_carry_lagged_values():
    """ADR-0006: sst_history_lags_weeks adds one channel per lag, in order, after
    the history stack and before the seasonal pair, carrying the lagged SST field."""
    from s2s.data.assemble import input_vars, sst_extra_lags
    cfg = _make_cfg(history_weeks=2)
    cfg.data.sst_history_lags_weeks = [4, 8, 12]
    assert sst_extra_lags(cfg) == [4, 8, 12]

    in_channels, _ = in_out_channels(cfg)
    n_in = len(input_vars(cfg))
    assert in_channels == n_in * 2 + 3 + 2   # history + 3 SST lags + (cos,sin)

    weekly = _make_weekly(n_weeks=30, lat=4, lon=8)
    out = assemble_arrays(weekly, cfg)
    assert out["inputs"].shape[1] == in_channels
    assert np.isfinite(out["inputs"]).all()

    # First valid init week = lo = max(history_weeks-1, max_lag) = 12.
    # Extra SST channels start right after the history body (n_in*history_weeks).
    body = n_in * 2
    sst = np.nan_to_num(
        weekly["sea_surface_temperature"].transpose("time", "latitude", "longitude").values,
        nan=0.0,
    )
    for j, lag in enumerate([4, 8, 12]):
        np.testing.assert_allclose(out["inputs"][0, body + j], sst[12 - lag], rtol=1e-5)


def test_seasonal_pair_is_last_two_channels():
    """doy_cos then doy_sin occupy the final two channels (MosaicBackbone contract)."""
    cfg = _make_cfg(history_weeks=2)
    weekly = _make_weekly(n_weeks=12, lat=3, lon=5)
    out = assemble_arrays(weekly, cfg)
    doy = weekly.time.dt.dayofyear.values.astype(np.float64)
    lo = 1  # history_weeks-1, no SST lags
    n = out["inputs"].shape[0]
    exp_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)[lo:lo + n]
    exp_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)[lo:lo + n]
    np.testing.assert_allclose(out["inputs"][:, -2, 0, 0], exp_cos, rtol=1e-5)
    np.testing.assert_allclose(out["inputs"][:, -1, 0, 0], exp_sin, rtol=1e-5)


def test_assemble_arrays_shapes():
    cfg = _make_cfg(history_weeks=2, lead_weeks=(1, 2, 3))
    weekly = _make_weekly(n_weeks=10, lat=4, lon=8)
    out = assemble_arrays(weekly, cfg)

    in_channels, out_channels = in_out_channels(cfg)
    n_time = weekly.sizes["time"]
    expected_n = n_time - (2 - 1) - 3  # minus history lookback, minus max lead lookahead
    assert out["inputs"].shape == (expected_n, in_channels, 4, 8)
    assert out["targets"].shape == (expected_n, 3, out_channels, 4, 8)
    assert out["time"].shape == (expected_n,)
    assert np.isfinite(out["inputs"]).all()


def test_assemble_arrays_targets_align_with_lead():
    """targets[i, li] must equal the actual value at init-week-index + lead, not some other week."""
    cfg = _make_cfg(history_weeks=1, lead_weeks=(1, 2))
    weekly = _make_weekly(n_weeks=6, lat=3, lon=5)
    out = assemble_arrays(weekly, cfg)

    target_var = "2m_temperature"
    out_idx = cfg.data.variables.targets.surface.index(target_var)
    raw = weekly[target_var].values  # (time, lat, lon)

    # First valid sample's init week is index (history_weeks - 1) = 0.
    init_idx = 0
    for li, lead in enumerate(cfg.data.lead_weeks):
        np.testing.assert_allclose(out["targets"][0, li, out_idx], raw[init_idx + lead])
