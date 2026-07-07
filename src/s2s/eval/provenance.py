"""Run provenance for eval outputs (Fix 8/M3). Torch-free, unit-testable.

Stamps every metrics row with enough to reproduce it: the code commit, the
checkpoint scored, whether fair-CRPS was used, and the ensemble size.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def git_sha(repo_root=None, short: bool = False) -> str:
    """Current commit SHA of the repo containing this file (or `repo_root`).

    Returns 'unknown' if git is unavailable or the tree isn't a repo. Uses the
    file's own location by default so it works even under Hydra's changed cwd.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]  # src/s2s/eval/ -> repo root
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.check_output(cmd, cwd=str(repo_root), stderr=subprocess.DEVNULL)
        return out.decode().strip() or "unknown"
    except Exception:
        return "unknown"


def provenance_columns(checkpoint_path, crps_fair: bool, n_members: int,
                       member_seed=None, repo_root=None) -> dict:
    """Constant per-row provenance dict for metrics.csv."""
    cols = {
        "commit_sha": git_sha(repo_root=repo_root),
        "checkpoint": str(checkpoint_path),
        "crps_fair": bool(crps_fair),
        "n_members": int(n_members),
    }
    if member_seed is not None:
        cols["member_seed"] = int(member_seed)
    return cols
