"""HEALPix round-trip loss DECOMPOSITION (Phase-C lever (a) diagnostic).

scripts/healpix_recon_check.py reported ~0.307 India-box relative RMSE for the
lon/lat <-> HEALPix round-trip -- but it measured UNTRAINED CrossAttentionInterpolate
modules, confounding mesh geometry with random projections. This script decomposes
that number into three references so we can tell whether lever (a) is real headroom
or a measurement artifact:

  1. geom_idw       -- weight-free inverse-distance round-trip on the same haversine
                       k-NN graph. The pure GEOMETRIC FLOOR (nside sweep 16, 32).
  2. attn_untrained -- the model's interpolators at init (what recon_check measures).
  3. attn_trained   -- the same modules loaded from a Stage-B checkpoint (+ckpt=...),
                       i.e. the loss the DEPLOYED model actually incurs. Skipped if
                       no checkpoint is given.

Decision rule (Phase-C, ADR-0005):
  * trained India-box relative RMSE small (< ~0.1)  -> 0.307 is an untrained artifact;
    lever (a) is a mirage -> pivot (record a no-op ADR).
  * geometric floor small but trained loss large    -> interpolators fail to learn the
    mapping; fix = geometric (IDW) init or an IDW skip around the mesh.
  * geometric floor itself large at nside=16, small at nside=32 -> fix = finer mesh.

Run (cluster CLI):
  python scripts/healpix_recon_decomp.py
  python scripts/healpix_recon_decomp.py +ckpt=/Datastorage/scdlds_bharat/s2s/checkpoints/seed_0/epoch=13-val_loss=0.2728.ckpt
"""
from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig

from s2s.eval.healpix_recon import (
    build_lonlat_grid,
    idw_round_trip,
    india_box_masks,
    latw_rmse,
    sample_fields,
)

NSIDE_SWEEP = (16, 32)  # NESTED HEALPix requires powers of two


def _get_grid(cfg: DictConfig):
    """Real datamodule coords when ERA5 is present; data-free fallback otherwise."""
    try:
        from s2s.data.datamodule import S2SDataModule

        dm = S2SDataModule(cfg)
        dm.prepare_data()
        dm.setup()
        return np.asarray(dm.lon, float), np.asarray(dm.latitude, float), "datamodule"
    except Exception as exc:  # noqa: BLE001 - diagnostic fallback is intentional
        lon, lat = build_lonlat_grid(float(cfg.data.resolution_deg))
        return lon, lat, f"synthetic ({type(exc).__name__})"


def _build_interpolators(cfg: DictConfig, lon, lat):
    """Construct a Transformer's two CrossAttentionInterpolate modules (untrained)."""
    import torch
    from s2s.models.mosaic.mosaic import (
        BottleneckConfig,
        ModelConfig,
        StageConfig,
        Transformer,
    )

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
    net.initialize_interpolation(
        torch.tensor(lon, dtype=torch.float32), torch.tensor(lat, dtype=torch.float32)
    )
    return net


def _load_trained_interpolators(net, ckpt_path):
    """Load interp_to_hp/interp_to_ll weights from a Lightning checkpoint in-place."""
    import torch

    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    # Lightning: self.model (S2SLitModule) -> transformer (MosaicBackbone) -> interp_to_*
    prefixes = ["model.transformer.", "transformer.", ""]
    loaded = {"interp_to_hp": 0, "interp_to_ll": 0}
    for sub in ("interp_to_hp", "interp_to_ll"):
        target = getattr(net, sub)
        for pre in prefixes:
            want = f"{pre}{sub}."
            picked = {k[len(want):]: v for k, v in sd.items() if k.startswith(want)}
            # keep only learned params (skip geometry buffers rebuilt by init)
            picked = {k: v for k, v in picked.items() if k in target.state_dict()
                      and target.state_dict()[k].shape == v.shape
                      and k not in ("neighbors", "rel_pos")}
            if picked:
                target.load_state_dict(picked, strict=False)
                loaded[sub] = len(picked)
                break
    if loaded["interp_to_hp"] == 0 and loaded["interp_to_ll"] == 0:
        raise KeyError(
            "No interp_to_hp/interp_to_ll weights found in checkpoint; keys look like: "
            + ", ".join(list(sd.keys())[:6])
        )
    return loaded


def _attn_round_trip(net, field_latlon, lon, lat):
    """lon/lat -> HEALPix -> lon/lat through the learned cross-attention interpolators."""
    import torch

    dim = net.interp_to_hp.to_kv.in_features
    W, H = len(lon), len(lat)
    field_lonlat = torch.tensor(field_latlon.T, dtype=torch.float32).reshape(-1, 1, 1)
    feat = field_lonlat.expand(-1, 1, dim).contiguous()
    with torch.no_grad():
        hp = net.interp_to_hp(feat)
        ll = net.interp_to_ll(hp)
    return ll.mean(dim=-1).reshape(W, H).T.numpy()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    lon, lat, src = _get_grid(cfg)
    box = cfg.data.eval_box
    latm, lonm = india_box_masks(
        lat, lon, box.lat_min, box.lat_max, box.lon_min, box.lon_max
    )
    fields = sample_fields(lon, lat)
    ckpt = cfg.get("ckpt", None)
    k = int(cfg.model.get("k_neighbors", 8))

    print("\n=== HEALPix round-trip loss decomposition (Phase-C lever (a)) ===")
    print(f"grid src={src}  {len(lat)}x{len(lon)}  India box={int(latm.sum())}x{int(lonm.sum())} pts  k={k}")
    hdr = f"{'reference':16s} {'field':18s} {'nside':>5s} {'globRMSE':>9s} {'globRel':>8s} {'IndiaRMSE':>9s} {'IndiaRel':>8s}"

    # 1. Geometric floor (torch-free) across the nside sweep.
    print("\n-- 1. geom_idw (weight-free inverse-distance = geometric floor) --")
    print(hdr)
    for name, f in fields.items():
        rng = float(f.max() - f.min())
        for nside in NSIDE_SWEEP:
            recon = idw_round_trip(f, lon, lat, nside, k=k, power=2.0)
            g = latw_rmse(recon, f, lat)
            ib = latw_rmse(recon, f, lat, latm, lonm)
            print(f"{'geom_idw':16s} {name:18s} {nside:5d} {g:9.4f} {g/rng:8.3f} {ib:9.4f} {ib/rng:8.3f}")

    # 2/3. Learned interpolators (untrained, and trained if a checkpoint is given).
    try:
        net = _build_interpolators(cfg, lon, lat)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[attention path skipped: {type(exc).__name__}: {exc}]")
        return

    nside = int(cfg.model.nside)
    print(f"\n-- 2. attn_untrained (learned interp at init, nside={nside}) --")
    print(hdr)
    for name, f in fields.items():
        rng = float(f.max() - f.min())
        recon = _attn_round_trip(net, f, lon, lat)
        g = latw_rmse(recon, f, lat)
        ib = latw_rmse(recon, f, lat, latm, lonm)
        print(f"{'attn_untrained':16s} {name:18s} {nside:5d} {g:9.4f} {g/rng:8.3f} {ib:9.4f} {ib/rng:8.3f}")

    if ckpt:
        info = _load_trained_interpolators(net, ckpt)
        print(f"\n-- 3. attn_trained (loaded {info}, nside={nside}) --")
        print(hdr)
        for name, f in fields.items():
            rng = float(f.max() - f.min())
            recon = _attn_round_trip(net, f, lon, lat)
            g = latw_rmse(recon, f, lat)
            ib = latw_rmse(recon, f, lat, latm, lonm)
            print(f"{'attn_trained':16s} {name:18s} {nside:5d} {g:9.4f} {g/rng:8.3f} {ib:9.4f} {ib/rng:8.3f}")
    else:
        print("\n[attn_trained skipped: pass +ckpt=/path/to/seed_0/epoch=..ckpt to enable]")

    print("\nDecision: if attn_trained India-box rel RMSE < ~0.1 the 0.307 is an untrained")
    print("artifact (lever (a) = mirage). If geom floor << attn_trained, fix = IDW init/skip.\n")


if __name__ == "__main__":
    main()
