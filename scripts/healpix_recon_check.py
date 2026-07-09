"""HEALPix interpolation round-trip diagnostic (Phase-B Stage-B safeguard).

The Mosaic adapter maps our lon/lat grid onto a HEALPix mesh and back via two
cross-attention interpolators (interp_to_hp, interp_to_ll). If that round-trip is
lossy -- especially over the India box -- it would cap regional skill and could
poison the Stage-B India-box CRPS. This script measures the reconstruction error
of pushing a known field lon/lat -> HEALPix -> lon/lat.

IMPORTANT (Fix 7/M1): those interpolators are LEARNED (CrossAttentionInterpolate:
to_q/to_kv/to_o Linear layers), NOT a fixed geometric operation. On an UNTRAINED
network the round-trip differs from the input by an essentially arbitrary affine
map (gain + bias), so a *raw* RMSE measures that random gain, not interpolation
geometry. (The earlier untrained "~0.3 relative RMSE" figure was exactly this
artifact and is retracted.) This script therefore:

  1. reports a gain/bias-CORRECTED RMSE (least-squares a,b removed per region)
     so the residual reflects spatial distortion, not a trivial scale mismatch;
  2. also prints the raw RMSE for transparency;
  3. with eval.checkpoint=<path> (the "--checkpoint" mode), loads the TRAINED
     interpolator weights and measures the trained round-trip -- the number that
     actually bears on a trained model's regional skill.

Run (untrained geometry):  python scripts/healpix_recon_check.py model=mosaic
Run (trained round-trip):  python scripts/healpix_recon_check.py model=mosaic \\
                               eval.checkpoint=/path/to/best_crps.ckpt
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from s2s.data.datamodule import S2SDataModule
from s2s.eval.recon import affine_correct, interp_state_subset, latweighted_rmse
from s2s.models.mosaic.mosaic import Transformer, ModelConfig, StageConfig, BottleneckConfig


def _build_net(cfg, lon, lat):
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
    return net


def _load_trained_interp(net, ckpt_path) -> int:
    """Copy trained interp_to_hp / interp_to_ll weights from a Lightning checkpoint
    into `net` (those modules' shapes depend only on dim/num_heads/k_neighbors, not
    the variable count, so the 1-channel diagnostic net accepts them). Returns the
    number of tensors loaded."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    loaded = interp_state_subset(sd)  # re-keyed interp_to_hp/ll tensors
    missing, unexpected = net.load_state_dict(loaded, strict=False)
    n = len([k for k in loaded if k not in unexpected])
    return n


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    dm = S2SDataModule(cfg)
    dm.prepare_data(); dm.setup()
    lat, lon = dm.latitude, dm.lon  # (32,), (64,)
    H, W = len(lat), len(lon)
    dim = int(cfg.model.dim)

    net = _build_net(cfg, lon, lat)

    ckpt_path = cfg.eval.get("checkpoint", None)
    trained = bool(ckpt_path)
    if trained:
        n = _load_trained_interp(net, Path(ckpt_path))
        print(f"Loaded {n} trained interpolator tensors from {ckpt_path}")

    box = cfg.data.eval_box
    lat_mask = (lat >= min(box.lat_min, box.lat_max)) & (lat <= max(box.lat_min, box.lat_max))
    lon_mask = (lon >= box.lon_min) & (lon <= box.lon_max)

    LON, LAT = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))  # (lat, lon)
    fields = {
        "planetary_wave_k3": (np.cos(3 * LON) * np.cos(LAT)).astype(np.float32),
        "white_noise": np.random.default_rng(0).normal(size=(H, W)).astype(np.float32),
    }

    mode = "TRAINED" if trained else "UNTRAINED (random-init; gain/bias-corrected)"
    print(f"\n=== HEALPix round-trip reconstruction [{mode}] ===")
    print(f"grid {H}x{W}, nside={int(cfg.model.nside)} -> npix={12*int(cfg.model.nside)**2}")
    print("relative = corrected RMSE / field range; raw shown for transparency.\n")
    for name, f in fields.items():
        field_lonlat = torch.tensor(f.T, dtype=torch.float32).reshape(-1, 1, 1)  # (lon*lat,1,1)
        feat = field_lonlat.expand(-1, 1, dim).contiguous()                      # (lon*lat,1,dim)
        with torch.no_grad():
            hp = net.interp_to_hp(feat)      # -> (npix,1,dim)
            ll = net.interp_to_ll(hp)        # -> (lon*lat,1,dim)
        recon = ll.mean(dim=-1).reshape(W, H).T.numpy()  # (lat, lon)

        raw_g = latweighted_rmse(recon, f, lat)
        raw_ib = latweighted_rmse(recon, f, lat, lat_mask, lon_mask)
        # Gain/bias-corrected, fitted PER region so each region's scale is removed.
        _, _, corr_g = affine_correct(recon, f, lat)
        _, _, corr_ib = affine_correct(recon, f, lat, lat_mask, lon_mask)
        cg = latweighted_rmse(corr_g, f, lat)
        cib = latweighted_rmse(corr_ib, f, lat, lat_mask, lon_mask)
        rng = float(f.max() - f.min())
        print(f"{name:20s}  global: raw={raw_g:.4f} corrected={cg:.4f} (rel={cg/rng:.3f})  "
              f"India: raw={raw_ib:.4f} corrected={cib:.4f} (rel={cib/rng:.3f})")

    print("\nInterpretation: read the CORRECTED relative RMSE (the raw number on an")
    print("untrained net is dominated by random-init gain and is not meaningful). A small")
    print("corrected India-box relative error => the mesh mapping preserves our eval region.")
    if not trained:
        print("For the number that bears on a TRAINED model, rerun with eval.checkpoint=<path>.\n")
    else:
        print("This is the TRAINED round-trip -- the figure relevant to regional skill.\n")


if __name__ == "__main__":
    main()
