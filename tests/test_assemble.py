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
    """6 input vars (2 targets + 4 predictors) x 2 history weeks + 2 doy channels (cos,sin)."""
    cfg = _make_cfg(history_weeks=2)
    in_channels, out_channels = in_out_channels(cfg)
    assert in_channels == 6 * 2 + 2
    assert out_channels == 2


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


def test_doy_cos_sin_channels_present_and_correct():
    """Last two input channels are doy_cos (-2) and doy_sin (-1) at the init week (Fix 5b)."""
    cfg = _make_cfg(history_weeks=2, lead_weeks=(1, 2, 3))
    weekly = _make_weekly(n_weeks=12, lat=4, lon=8)
    out = assemble_arrays(weekly, cfg)
    init_times = pd.DatetimeIndex(out["time"])
    doy = init_times.dayofyear.values.astype(float)
    exp_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    exp_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    np.testing.assert_allclose(out["inputs"][:, -2, 0, 0], exp_cos, atol=1e-6)
    np.testing.assert_allclose(out["inputs"][:, -1, 0, 0], exp_sin, atol=1e-6)


def test_train_lead_window_ends_at_t_plus_7L():
    """Fix 5a: daily-strided TRAIN lead-L target is the 7-day mean ending at t+7L
    (aligned to the W-MON test bin), NOT ending 6 days later at t+7L+6."""
    from s2s.data.windows import daily_init_weekly_windows

    cfg = _make_cfg(history_weeks=2, lead_weeks=(1, 2, 3, 4, 5, 6))
    n = 300
    time = pd.date_range("2010-01-01", periods=n, freq="D")
    rng = np.random.default_rng(1)
    names = ["2m_temperature", "total_precipitation_24hr", "sea_surface_temperature",
             "geopotential_500", "u_component_of_wind_850", "u_component_of_wind_200"]
    ds = xr.Dataset(
        {nm: (("time", "latitude", "longitude"), rng.normal(size=(n, 2, 3)).astype("float32")) for nm in names},
        coords={"time": time, "latitude": np.linspace(5, 35, 2), "longitude": np.linspace(70, 95, 3)},
    )
    out = daily_init_weekly_windows(ds, cfg, stride_days=1)
    t2m = ds["2m_temperature"].transpose("time", "latitude", "longitude").values
    init_idx = pd.DatetimeIndex(time).get_loc(pd.DatetimeIndex(out["time"])[0])
    for li, L in enumerate([1, 2, 3, 4, 5, 6]):
        end = init_idx + 7 * L
        expect = t2m[end - 6:end + 1].mean(axis=0)             # [t+7L-6 .. t+7L]
        np.testing.assert_allclose(out["targets"][0, li, 0], expect, atol=1e-5)
        old_window = t2m[end:end + 7].mean(axis=0)             # the old (misaligned) window
        assert not np.allclose(out["targets"][0, li, 0], old_window, atol=1e-3)
    assert out["inputs"].shape[1] == 6 * 2 + 2                 # 12 history + doy_cos + doy_sin


def test_sst_indices_add_channels_and_track_enso():
    """Phase-C lever b: sst_indices add low-dim broadcast channels between the history
    stack and the seasonal pair; nino34 tracks the Nino 3.4 SST box (32x64 grid)."""
    from s2s.data.assemble import input_vars, sst_index_names, n_sst_index_channels

    cfg = _make_cfg(history_weeks=2, lead_weeks=(1, 2, 3))
    cfg.data.sst_indices = ["nino34", "dmi"]
    cfg.data.sst_index_lags_weeks = [0]
    assert sst_index_names(cfg) == ["nino34", "dmi"]
    assert n_sst_index_channels(cfg) == 2

    n_in = len(input_vars(cfg))
    in_channels, _ = in_out_channels(cfg)
    assert in_channels == n_in * 2 + 2 + 2   # history + 2 indices + (cos, sin)

    lat = -90 + 5.625 / 2 + np.arange(32) * 5.625
    lon = np.arange(64) * 5.625

    def _weekly(nino_warm):
        rng = np.random.default_rng(0)
        t = pd.date_range("2010-01-04", periods=20, freq="7D")
        sh = (20, 32, 64)
        dv = {k: (("time", "latitude", "longitude"), rng.normal(size=sh).astype(np.float32))
              for k in ["2m_temperature", "total_precipitation_24hr", "sea_surface_temperature",
                        "geopotential_500", "u_component_of_wind_850", "u_component_of_wind_200"]}
        ds = xr.Dataset(dv, coords={"time": t, "latitude": lat, "longitude": lon})
        if nino_warm:
            m = (lat >= -5) & (lat <= 5); nn = (lon >= 190) & (lon <= 240)
            ds["sea_surface_temperature"].loc[dict(latitude=lat[m], longitude=lon[nn])] += nino_warm
        return ds

    warm = assemble_arrays(_weekly(2.0), cfg)
    neut = assemble_arrays(_weekly(0.0), cfg)
    assert warm["inputs"].shape[1] == in_channels
    assert np.isfinite(warm["inputs"]).all()
    body = n_in * 2  # index channels sit right after the history stack
    assert warm["inputs"][:, body, 0, 0].mean() > neut["inputs"][:, body, 0, 0].mean() + 1.0
    assert np.allclose(warm["inputs"][0, body], warm["inputs"][0, body, 0, 0])  # spatially uniform
    assert not np.allclose(warm["inputs"][:, -2, 0, 0], warm["inputs"][:, -1, 0, 0])  # cos != sin (pair last)
