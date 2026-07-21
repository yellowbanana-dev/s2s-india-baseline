"""MIN-6 / MIN-8 guards (review 2026-07-14). Assertion-shaped; no numerical tolerances."""
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from s2s.data.download import verify_pull


def _cfg(res=5.625):
    return SimpleNamespace(data=SimpleNamespace(resolution_deg=res))


def _ds(lat_ascending: bool, nlat=32, nlon=64):
    lat = -87.1875 + 5.625 * np.arange(nlat)
    if not lat_ascending:
        lat = lat[::-1]
    lon = 5.625 * np.arange(nlon)
    t = np.arange("2000-01-01", "2000-01-03", np.timedelta64(6, "h"), dtype="datetime64[ns]")
    data = np.zeros((t.size, nlat, nlon), dtype="float32")
    return xr.Dataset(
        {"2m_temperature": (("time", "latitude", "longitude"), data)},
        coords={"time": t, "latitude": lat, "longitude": lon},
    )


# ---------------- MIN-8: verify_pull must pin latitude ordering ----------------

def test_verify_pull_rejects_descending_latitude():
    with pytest.raises(AssertionError, match="ASCENDING"):
        verify_pull(_ds(lat_ascending=False), _cfg())


def test_verify_pull_accepts_ascending_latitude_past_the_lat_check():
    # Ascending lat must get PAST the new assert. Later checks (physical ranges) may still
    # fire on this synthetic all-zero field -- we only require that it is not the lat error.
    try:
        verify_pull(_ds(lat_ascending=True), _cfg())
    except AssertionError as e:
        assert "ASCENDING" not in str(e), f"ascending grid wrongly rejected by the lat check: {e}"


# ---------------- MIN-6: longitude is required for mosaic ----------------

def test_mosaic_without_longitude_raises():
    pytest.importorskip("torch")
    from omegaconf import OmegaConf
    from s2s.models.lit import S2SLitModule

    cfg = OmegaConf.create({
        "train": {"lr": 3.0e-5, "weight_decay": 0.1, "warmup_epochs": 2,
                  "max_epochs": 50, "min_lr": 1.0e-6},
        "model": {"name": "mosaic"},
    })
    with pytest.raises(ValueError, match="longitude"):
        S2SLitModule(in_channels=6, out_channels=2, lead=6,
                     latitude=np.linspace(-87.1875, 87.1875, 32), cfg=cfg)


def test_patch_vit_without_longitude_still_allowed():
    # patch_vit never reads longitude, so None must stay legal (an existing test relies on it).
    pytest.importorskip("torch")
    from omegaconf import OmegaConf
    from s2s.models.lit import S2SLitModule

    cfg = OmegaConf.create({
        "train": {"lr": 3.0e-4, "weight_decay": 0.1, "warmup_epochs": 2,
                  "max_epochs": 50, "min_lr": 1.0e-6},
        "model": {"patch_size": 2, "embed_dim": 32, "depth": 2,
                  "num_heads": 4, "mlp_ratio": 4.0, "drop_rate": 0.1},
    })
    lit = S2SLitModule(in_channels=6, out_channels=2, lead=6,
                       latitude=np.linspace(-87.1875, 87.1875, 32), cfg=cfg)
    assert lit is not None
