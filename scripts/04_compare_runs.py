"""Paired model-vs-model CRPSS comparison on a COMMON grid (MAJ-3 / ADR-0007 f3).

Two eval runs scored on the SAME grid over the SAME test inits are compared with a paired
moving-block bootstrap on the CRPSS DIFFERENCE. This is the pre-registered test for the
1.5deg-vs-5.625deg readout: overlapping marginal CIs do NOT establish that two models are
indistinguishable, so the marginal CIs in metrics.csv cannot settle f3 on their own.

Usage (A = baseline, B = candidate; delta > 0 means B is more skilful):
    python scripts/04_compare_runs.py RUN_A_DIR RUN_B_DIR [--block-len 8] [--n-boot 5000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from s2s.eval.bootstrap import paired_delta_crpss_bootstrap


def _load(run_dir: Path):
    f = run_dir / "per_sample_crps.npz"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found -- re-run scripts/03_evaluate.py at this commit or later; "
            "older runs did not persist per-sample CRPS."
        )
    z = np.load(f)
    return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path, help="baseline run dir (A)")
    ap.add_argument("run_b", type=Path, help="candidate run dir (B)")
    ap.add_argument("--block-len", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", choices=["prob", "trend"], default="prob")
    args = ap.parse_args()

    A, B = _load(args.run_a), _load(args.run_b)

    ta, tb = A.get("init_times"), B.get("init_times")
    if ta is None or tb is None or ta.shape != tb.shape or not np.array_equal(ta, tb):
        raise SystemExit(
            "init_times differ between runs -- the paired test requires the SAME test "
            f"inits (A n={None if ta is None else ta.size}, B n={None if tb is None else tb.size})."
        )

    keys = sorted(k for k in A if k.endswith("|model") and k in B)
    if not keys:
        raise SystemExit("no overlapping (variable, lead) cells between the two runs")

    rows = []
    for km in keys:
        var, lead, _ = km.split("|")
        kr = f"{var}|{lead}|{args.reference}"
        if kr not in A or kr not in B:
            continue
        r = paired_delta_crpss_bootstrap(
            A[km], A[kr], B[km], B[kr],
            block_len=args.block_len, n_boot=args.n_boot, seed=args.seed,
        )
        sig = "yes" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "no"
        rows.append({
            "variable": var, "lead_week": int(lead),
            "crpss_A": r["crpss_a"], "crpss_B": r["crpss_b"], "delta_B_minus_A": r["delta"],
            "delta_ci_lo": r["ci_lo"], "delta_ci_hi": r["ci_hi"],
            "delta_boot_se": r["boot_se"], "p_two_sided": r["p_two_sided"],
            "ci_excludes_zero": sig, "n": r["n"],
        })

    out = pd.DataFrame(rows).sort_values(["variable", "lead_week"])
    print(f"\nA (baseline) = {args.run_a}")
    print(f"B (candidate)= {args.run_b}")
    print(f"reference    = climatology_{args.reference};  paired moving-block bootstrap "
          f"(block_len={args.block_len}, n_boot={args.n_boot})")
    print("delta = CRPSS_B - CRPSS_A;  delta > 0 means B more skilful\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    dest = args.run_b / "paired_comparison_vs_A.csv"
    out.to_csv(dest, index=False)
    print(f"\nSaved: {dest}")

    gate = out[out.lead_week.isin([3, 4])]
    if len(gate):
        wins = (gate.delta_B_minus_A > 0) & (gate.ci_excludes_zero == "yes")
        loses = (gate.delta_B_minus_A < 0) & (gate.ci_excludes_zero == "yes")
        print(f"\nGATE LEADS (wk3-4): B significantly better in {int(wins.sum())}/{len(gate)} "
              f"cells, significantly worse in {int(loses.sum())}/{len(gate)}, "
              f"indistinguishable in {int((~(wins | loses)).sum())}/{len(gate)}.")


if __name__ == "__main__":
    main()
