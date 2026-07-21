"""Is the sparse attention path suppressing the ensemble-noise pathway? (MAJ-2 follow-up)

Both sparse ablation runs collapsed to spread_error_ratio ~0.001 while the 1.5deg model --
which uses the SAME sparse path -- sits at 0.77-0.84. So the sparse path per se does not kill
noise; something about this configuration does.

Noise enters at exactly one place per block (primitives.cSwiGLU):
    noise = noise_bias(z)                       # (members, hidden)
    out   = w2( SiLU(x1 + noise) * x3 )         # x1, x3 = w13(x).chunk(2)
so the noise pathway is suppressed if EITHER
  (a) noise_bias collapses toward 0 relative to w13  -> noise is negligible vs signal, or
  (b) NoiseGenerator.to_noise collapses              -> z itself is degenerate.
Both are readable directly off the checkpoint weights: no data, no forward pass, no GPU.

Usage (CPU, seconds):
    python scripts/_diag_noise_path.py DENSE.ckpt SPARSE.ckpt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load(p: Path):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return ck.get("state_dict", ck)


def _report(name: str, sd: dict) -> dict:
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    noise_bias = {k: v for k, v in sd.items() if k.endswith("noise_bias.weight")}
    w13 = {k: v for k, v in sd.items() if k.endswith("w13.weight")}
    to_noise = {k: v for k, v in sd.items() if k.endswith("to_noise.weight")}
    gate = {k: v for k, v in sd.items() if "to_strategy_combine_mlp" in k}

    print(f"noise_bias tensors: {len(noise_bias)} | w13: {len(w13)} | "
          f"to_noise: {len(to_noise)} | strategy-gate: {len(gate)}")

    for k, v in sorted(to_noise.items()):
        print(f"  NoiseGenerator {k}: ||W||={v.norm():.6f}  rms={v.pow(2).mean().sqrt():.6f}")

    ratios = []
    print(f"\n  {'block':<48} {'||noise_bias||':>14} {'||w13||':>10} {'ratio':>10}")
    for k in sorted(noise_bias):
        stem = k[: -len("noise_bias.weight")]
        wk = stem + "w13.weight"
        if wk not in w13:
            continue
        nb = float(noise_bias[k].norm())
        ww = float(w13[wk].norm())
        # per-input-unit comparison: noise_bias maps noise_dim->hidden, w13 maps dim->2*hidden
        nb_rms = float(noise_bias[k].pow(2).mean().sqrt())
        ww_rms = float(w13[wk].pow(2).mean().sqrt())
        r = nb_rms / ww_rms if ww_rms > 0 else float("nan")
        ratios.append(r)
        print(f"  {stem[:46]:<48} {nb:>14.5f} {ww:>10.5f} {r:>10.5f}")

    if ratios:
        t = torch.tensor(ratios)
        print(f"\n  noise_bias/w13 RMS ratio: mean={t.mean():.6f}  min={t.min():.6f}  "
              f"max={t.max():.6f}  n={len(ratios)}")
    if gate:
        for k, v in sorted(gate.items()):
            print(f"  gate {k[:56]}: ||W||={v.norm():.5f} rms={v.pow(2).mean().sqrt():.6f}")
    return {"ratios": ratios}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dense", type=Path)
    ap.add_argument("sparse", type=Path)
    args = ap.parse_args()

    d = _report(f"DENSE  {args.dense.name}", _load(args.dense))
    s = _report(f"SPARSE {args.sparse.name}", _load(args.sparse))

    print(f"\n{'='*70}\nVERDICT\n{'='*70}")
    if d["ratios"] and s["ratios"]:
        dm = torch.tensor(d["ratios"]).mean()
        sm = torch.tensor(s["ratios"]).mean()
        print(f"mean noise_bias/w13 RMS ratio   dense={dm:.6f}   sparse={sm:.6f}   "
              f"sparse/dense={sm/dm if dm > 0 else float('nan'):.4f}")
        if dm > 0 and sm / dm < 0.25:
            print("=> SPARSE noise pathway is SUPPRESSED relative to dense: training drove")
            print("   noise_bias toward zero, so members are near-identical. Consistent with")
            print("   the observed spread_error_ratio ~0.001. The collapse is LEARNED, and a")
            print("   noise-pathway/optimisation problem -- NOT evidence about attention capacity.")
        elif dm > 0 and sm / dm > 0.75:
            print("=> noise pathway is INTACT in the sparse model (weights comparable to dense).")
            print("   The spread collapse must arise downstream (e.g. x3 gating the noise term")
            print("   to zero, or SiLU saturation); next step is forward-pass instrumentation.")
        else:
            print("=> partial suppression; inconclusive from weights alone, instrument forward.")
    print()


if __name__ == "__main__":
    main()
