"""Exact test init-week time reconstruction for evaluation (Fix 1 / m1).

`assemble_arrays` (src/s2s/data/assemble.py) keeps only the init weeks that have
BOTH a full history window behind them and a full lead window ahead of them:

    lo = history_weeks - 1          # drop leading weeks lacking full history
    hi = n_time - max_lead          # drop trailing weeks lacking full lead
    valid_idx = arange(lo, hi)
    time = weekly.time[valid_idx]

The evaluation script needs the SAME per-sample init timestamps to do week-of-year
climatology pooling and physical-event reconstruction. The previous inline code in
scripts/03_evaluate.py trimmed only `max_lead` off the END of the weekly axis and
relied on an `init_times[-n_samples:]` fallback to silently repair the missing
`history_weeks - 1` leading drop. That fallback masked any real divergence between
eval and assemble. This module is the single, testable source of truth for the
reconstruction so the two can never quietly disagree.
"""
from __future__ import annotations

import numpy as np


def valid_init_index(n_time: int, history_weeks: int, max_lead: int) -> np.ndarray:
    """Return valid_idx = arange(history_weeks-1, n_time-max_lead).

    Identical to assemble_arrays' valid_idx. Raises if no week is valid.
    """
    lo = int(history_weeks) - 1
    hi = int(n_time) - int(max_lead)
    if hi <= lo:
        raise ValueError(
            f"not enough weeks ({n_time}) for history_weeks={history_weeks} "
            f"and max lead={max_lead} (lo={lo}, hi={hi})"
        )
    return np.arange(lo, hi)


def reconstruct_init_times(weekly_time, history_weeks: int, max_lead: int) -> np.ndarray:
    """Init-week timestamps for the test set, matching assemble_arrays exactly.

    weekly_time : 1-D array of the test split's weekly-mean time coordinate
                  (the same axis assemble_arrays consumes).
    Returns weekly_time[valid_idx].
    """
    weekly_time = np.asarray(weekly_time)
    idx = valid_init_index(len(weekly_time), history_weeks, max_lead)
    return weekly_time[idx]
