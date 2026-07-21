"""04_compare_runs.py --series selection (torch-free, runs the script end-to-end).

Guards the calibration-attribution path: --series model_cal must score the calibrated
per-sample series and write a separate file, while the default stays byte-identical.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _mk_run(d: Path, model_vals, cal_vals, prob_vals, times):
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "per_sample_crps.npz", **{
        "2m_temperature|3|model": model_vals,
        "2m_temperature|3|model_cal": cal_vals,
        "2m_temperature|3|prob": prob_vals,
        "init_times": times,
    })


def _run(a, b, extra=()):
    cmd = [sys.executable, str(REPO / "scripts" / "04_compare_runs.py"), str(a), str(b),
           "--n-boot", "200", *extra]
    env = {"PYTHONPATH": str(REPO / "src")}
    import os
    e = dict(os.environ); e.update(env)
    r = subprocess.run(cmd, capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_series_flag_selects_calibrated_and_writes_separate_file(tmp_path):
    n = 40
    times = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
    prob = np.full(n, 1.0)
    # B's calibrated series is much better than its as-forecast series -> the two runs
    # must produce different deltas, proving the flag actually switches series.
    _mk_run(tmp_path / "A", np.full(n, 0.80), np.full(n, 0.80), prob, times)
    _mk_run(tmp_path / "B", np.full(n, 0.95), np.full(n, 0.60), prob, times)

    _run(tmp_path / "A", tmp_path / "B")
    _run(tmp_path / "A", tmp_path / "B", ("--series", "model_cal"))

    default_csv = tmp_path / "B" / "paired_comparison_vs_A.csv"
    cal_csv = tmp_path / "B" / "paired_comparison_vs_A_model_cal.csv"
    assert default_csv.exists() and cal_csv.exists(), "series must write separate files"

    d_default = pd.read_csv(default_csv)["delta_B_minus_A"].iloc[0]
    d_cal = pd.read_csv(cal_csv)["delta_B_minus_A"].iloc[0]
    # as-forecast: B worse (0.95 vs 0.80) -> negative; calibrated: B better -> positive
    assert d_default < 0 < d_cal


def test_unknown_series_is_rejected(tmp_path):
    n = 10
    times = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
    _mk_run(tmp_path / "A", np.full(n, 0.8), np.full(n, 0.8), np.full(n, 1.0), times)
    _mk_run(tmp_path / "B", np.full(n, 0.9), np.full(n, 0.7), np.full(n, 1.0), times)
    cmd = [sys.executable, str(REPO / "scripts" / "04_compare_runs.py"),
           str(tmp_path / "A"), str(tmp_path / "B"), "--series", "bogus"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0 and "invalid choice" in r.stderr
