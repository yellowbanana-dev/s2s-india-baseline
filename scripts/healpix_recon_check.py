"""HEALPix interpolation round-trip diagnostic (Phase-B Stage-B safeguard).

The Mosaic adapter maps our lon/lat grid onto a HEALPix mesh and back via two
cross-attention interpolators (interp_to_hp, interp_to_ll). If that round-trip is
lossy -- especially over the India box -- it would cap regional skill and could
explain the deterministic mean-ACC gap vs patch-ViT, and would poison the Stage-B
India-box CRPS. This script measures the identity-reconstruction error of the
UNTRAINED interpolation path (a fixed geometric operation, independent of weights):
push a known field lon/lat -> HEALPix -> lon/lat and compare to the input.

Reports latitude-weighted RMSE, GLOBAL and over the India eval box, for a few
controlled test fields. Run:  python scripts/healpix_recon_check.py
"""
from __future__ import annotations

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from s2s.data.datamodule import S2SDataModule
from s2s.models.mosaic.mosaic import Transformer, ModelConfig, StageConfig, BottleneckConfig


def _latw_rmse(pred, truth, lat_deg, lat_mask=None, lon_mask=None):
    """Latitude-weighted RMSE over (lat, lon), optionally restricted by masks."""
    w = np.cos(np.deg2rad(lat_deg)).reshape(-1, 1)  # (lat,1)
    err2 = (pred - truth) ** 2
    if lat_mask is not None:
        err2 = err2[lat_mask, :]; w = w[lat_mask, :]
    if lon_mask is not None:
        err2 = err2[:, lon_mask]
    return float(np.sqrt((err2 * w).sum() / (w.sum() * err2.shape[1])))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    dm = S2SDataModule(cfg)
    dm.prepare_data(); dm.setup()
    lat, lon = dm.latitude, dm.lon  # (32,), (64,)
    H, W = len(lat), len(lon)

    mc = cfg.model
    dim = int(mc.dim)
    stage = StageConfig(
        nside=int(mc.nside), dim=dim, num_heads=int(mc.num_heads),
        block_attn_size=int(mc.block_attn_size), sparse_block_size=int(mc.sparse_block_size),
        sparse_block_count=int(mc.sparse_block_count), encoder_depth=1, decoder_depth=1,
        mlp_ratio=float(mc.mlp_ratio), gqa_ratio=int(mc.gqa_ratio),
    )
    bn = BottleneckConfig(
        nside=int(mc.bottleneck_nside), dim=int(mc.bottleneck_dim),
        num_heads=int(mc.bottleneck_num_heads), block_attn_size=int(mc.bottleneck_block_attn_size),
        sparse_block_size=int(mc.sparse_block_size), sparse_block_count=int(mc.sparse_block_count),
        depth=1, mlp_ratio=float(mc.mlp_ratio), gqa_ratio=int(mc.gqa_ratio),
    )
    model_cfg = ModelConfig(
        dim=dim, num_heads=int(mc.num_heads), k_neighbors=int(mc.k_neighbors),
        qk_norm=bool(mc.qk_norm), rope=bool(mc.rope), rope_theta=int(mc.rope_theta),
        sparse_every=int(mc.sparse_every), variables=["f"], static_variables=[],
        qkv_compress_ratio=int(mc.qkv_compress_ratio), cg_stage_cfgs=[stage],
        bottleneck_cfg=bn, num_history_steps=1, noise_dim=0,
        drop_rate=0.0, ortho_init=False, rmsnorm_elementwise_affine=True,
        no_compression=bool(getattr(mc, "no_compression", False)),
    )
    net = Transformer(model_cfg, seed=0).eval()
    lon_t = torch.tensor(lon, dtype=torch.float32)
    lat_t = torch.tensor(lat, dtype=torch.float32)
    net.initialize_interpolation(lon_t, lat_t)

    # India-box masks on the (lat, lon) grid.
    box = cfg.data.eval_box
    lat_mask = (lat >= min(box.lat_min, box.lat_max)) & (lat <= max(box.lat_min, box.lat_max))
    lon_mask = (lon >= box.lon_min) & (lon <= box.lon_max)

    # Controlled fields: a smooth planetary wave, and white noise (worst case).
    LON, LAT = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))  # (lat, lon)
    fields = {
        "planetary_wave_k3": (np.cos(3 * LON) * np.cos(LAT)).astype(np.float32),
        "white_noise": np.random.default_rng(0).normal(size=(H, W)).astype(np.float32),
    }

    print("\n=== HEALPix round-trip identity reconstruction (untrained interp) ===")
    print(f"grid {H}x{W}, nside={int(mc.nside)} -> npix={12*int(mc.nside)**2}\n")
    for name, f in fields.items():
        # Build a (lon*lat, batch=1, dim) feature by broadcasting the scalar field
        # into `dim` channels, run interp_to_hp then interp_to_ll, take channel mean.
        field_lonlat = torch.tensor(f.T, dtype=torch.float32).reshape(-1, 1, 1)  # (lon*lat,1,1)
        feat = field_lonlat.expand(-1, 1, dim).contiguous()                      # (lon*lat,1,dim)
        with torch.no_grad():
            hp = net.interp_to_hp(feat)      # -> (npix,1,dim)
            ll = net.interp_to_ll(hp)        # -> (lon*lat,1,dim)
        recon = ll.mean(dim=-1).reshape(W, H).T.numpy()  # (lat, lon)
        g = _latw_rmse(recon, f, lat)
        ib = _latw_rmse(recon, f, lat, lat_mask, lon_mask)
        rng = float(f.max() - f.min())
        print(f"{name:20s}  global RMSE={g:.4f}  India-box RMSE={ib:.4f}  "
              f"(field range {rng:.3f}; relative global={g/rng:.3f}, India={ib/rng:.3f})")
    print("\nInterpretation: relative RMSE << 1 => interpolation is near-lossless. If the")
    print("India-box relative error is large (e.g. > ~0.3) the mesh mapping distorts our")
    print("eval region and Stage-B India CRPS is compromised -- flag before trusting eval.\n")


if __name__ == "__main__":
    main()
