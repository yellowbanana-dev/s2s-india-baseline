"""Data-integrity tests - the highest-ROI tests in ML (task #3 guardrails).

These encode the cardinal rule so leakage fails loudly instead of silently
inflating skill. No network: everything runs on small synthetic datasets.
"""
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf

from s2s.data.climatology import fit_climatology, fit_normalizer, to_anomaly
from s2s.data.splits import split_by_year


def _make_cfg(embargo_weeks: int = 4):
    return OmegaConf.create(
        {
            "data": {
                "splits": {
                    "train": [1979, 2012],
                    "val": [2013, 2017],
                    "test": [2018, 2020],
                    "embargo_weeks": embargo_weeks,
                }
            }
        }
    )


def _make_dataset(start="1979-01-01", end="2020-12-31", seed=0):
    rng = np.random.default_rng(seed)
    time = pd.date_range(start, end, freq="D")
    lat = np.linspace(-85, 85, 4)
    lon = np.linspace(0, 350, 8)
    t2m = 280 + 10 * np.sin(2 * np.pi * time.dayofyear.values / 365.25)[:, None, None]
    t2m = t2m + rng.normal(scale=0.5, size=(len(time), len(lat), len(lon)))
    precip = np.abs(rng.normal(scale=1.0, size=(len(time), len(lat), len(lon))))
    return xr.Dataset(
        {
            "2m_temperature": (("time", "latitude", "longitude"), t2m),
            "total_precipitation_24hr": (("time", "latitude", "longitude"), precip),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def test_climatology_uses_train_years_only():
    """fit_climatology must not touch any val/test timestamp."""
    cfg = _make_cfg()
    ds = _make_dataset()
    splits = split_by_year(ds, cfg)

    train_years = splits["train"].time.dt.year.values
    assert train_years.max() <= 2012
    assert train_years.min() >= 1979

    # Corrupting val/test data must not change a climatology fit on train only.
    clim_before = fit_climatology(splits["train"], cfg)
    corrupted = ds.copy()
    val_test_mask = ds.time.dt.year.values > 2012
    corrupted["2m_temperature"].values[val_test_mask] += 1000.0
    splits_after = split_by_year(corrupted, cfg)
    clim_after = fit_climatology(splits_after["train"], cfg)

    xr.testing.assert_allclose(clim_before, clim_after)


def test_normalizer_stats_from_train_only():
    """Normalization mean/std come only from training anomalies."""
    cfg = _make_cfg()
    ds = _make_dataset()
    splits = split_by_year(ds, cfg)

    clim = fit_climatology(splits["train"], cfg)
    train_anom = to_anomaly(splits["train"], clim)
    full_anom = to_anomaly(ds, clim)

    train_stats = fit_normalizer(train_anom, cfg)
    full_stats = fit_normalizer(full_anom, cfg)

    # Train-only stats must differ from stats computed over the whole record
    # (val/test years exist and have a different distribution / length).
    assert train_stats["2m_temperature"]["mean"] != full_stats["2m_temperature"]["mean"]


def test_embargo_gap_between_splits():
    """No sample window spans a split boundary; embargo gap is respected."""
    embargo_weeks = 4
    cfg = _make_cfg(embargo_weeks=embargo_weeks)
    ds = _make_dataset()
    splits = split_by_year(ds, cfg)

    train_end = splits["train"].time.values.max()
    val_start = splits["val"].time.values.min()
    val_end = splits["val"].time.values.max()
    test_start = splits["test"].time.values.min()

    gap_1 = (pd.Timestamp(val_start) - pd.Timestamp(train_end)).days
    gap_2 = (pd.Timestamp(test_start) - pd.Timestamp(val_end)).days

    assert gap_1 >= embargo_weeks * 7
    assert gap_2 >= embargo_weeks * 7


def test_no_overlap_between_split_indices():
    """train/val/test sample indices are disjoint."""
    cfg = _make_cfg()
    ds = _make_dataset()
    splits = split_by_year(ds, cfg)

    train_t = set(splits["train"].time.values)
    val_t = set(splits["val"].time.values)
    test_t = set(splits["test"].time.values)

    assert train_t.isdisjoint(val_t)
    assert val_t.isdisjoint(test_t)
    assert train_t.isdisjoint(test_t)
