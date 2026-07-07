"""Static guard: scripts must have no undefined names (Fix 2 follow-up).

Unit tests import modules in isolation, so they never exercise a script's own
import block. That gap let scripts/03_evaluate.py call block_bootstrap_crpss /
year_bootstrap_crpss without importing them — the unit tests passed while the
script NameError'd on every run. pyflakes detects undefined names statically
(no execution, so no torch/CUDA/data needed), catching this class of bug.
"""
import glob
import os

import pytest

pyflakes_api = pytest.importorskip("pyflakes.api")
from pyflakes import messages as pf_messages  # noqa: E402
from pyflakes.reporter import Reporter  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = sorted(glob.glob(os.path.join(REPO, "scripts", "*.py")))


class _UndefCollector(Reporter):
    def __init__(self):
        super().__init__(open(os.devnull, "w"), open(os.devnull, "w"))
        self.undef = []

    def flake(self, message):
        if isinstance(message, pf_messages.UndefinedName):
            self.undef.append(f"{message.filename}:{message.lineno} {message.message % message.message_args}")


@pytest.mark.parametrize("path", SCRIPTS, ids=[os.path.basename(p) for p in SCRIPTS])
def test_script_has_no_undefined_names(path):
    with open(path) as fh:
        src = fh.read()
    rep = _UndefCollector()
    pyflakes_api.check(src, path, rep)
    assert not rep.undef, "undefined names (missing imports?):\n" + "\n".join(rep.undef)
