"""Fix 8 (M3): eval provenance stamping. Torch-free."""
import subprocess
from pathlib import Path

from s2s.eval.provenance import git_sha, provenance_columns

REPO = Path(__file__).resolve().parents[1]


def test_git_sha_matches_repo_head():
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO)
    ).decode().strip()
    assert git_sha(repo_root=REPO) == expected
    assert len(git_sha(repo_root=REPO)) == 40


def test_git_sha_unknown_outside_repo(tmp_path):
    assert git_sha(repo_root=tmp_path) == "unknown"


def test_provenance_columns_shape_and_types():
    cols = provenance_columns(
        checkpoint_path="/ckpt/seed_0/best_crps.ckpt",
        crps_fair=True, n_members=16, member_seed=42, repo_root=REPO,
    )
    assert cols["checkpoint"] == "/ckpt/seed_0/best_crps.ckpt"
    assert cols["crps_fair"] is True and isinstance(cols["crps_fair"], bool)
    assert cols["n_members"] == 16
    assert cols["member_seed"] == 42
    assert cols["commit_sha"] == git_sha(repo_root=REPO)


def test_provenance_omits_member_seed_when_none():
    cols = provenance_columns("/x.ckpt", crps_fair=False, n_members=1, repo_root=REPO)
    assert "member_seed" not in cols
    assert cols["crps_fair"] is False
