"""Stage 2a - Temporal splits with embargo (task #3).

Split FIRST, before computing any statistic. This is the safeguard for the
cardinal rule: climatology + normalization come from TRAIN years only.
"""
from __future__ import annotations
import xarray as xr


def split_by_year(ds: xr.Dataset, cfg) -> dict[str, xr.Dataset]:
    """Return {'train','val','test'} subsets by year block.

    Apply an embargo gap (cfg.data.splits.embargo_weeks) between blocks so a
    sample near a boundary cannot 'see' the next split via weather autocorrelation.
    """
    raise NotImplementedError
