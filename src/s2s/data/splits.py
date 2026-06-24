"""Stage 2a - Temporal splits with embargo (task #3).

Split FIRST, before computing any statistic. This is the safeguard for the
cardinal rule: climatology + normalization come from TRAIN years only.
"""
from __future__ import annotations

import pandas as pd
import xarray as xr

_ORDER = ("train", "val", "test")


def split_by_year(ds: xr.Dataset, cfg) -> dict[str, xr.Dataset]:
    """Return {'train','val','test'} subsets by year block.

    Apply an embargo gap (cfg.data.splits.embargo_weeks) between blocks so a
    sample near a boundary cannot 'see' the next split via weather autocorrelation.
    The gap is carved out of the END of every block except the last -- those
    days belong to neither split, which is what makes the boundary safe.
    """
    splits_cfg = cfg.data.splits
    embargo = pd.Timedelta(weeks=int(splits_cfg.embargo_weeks))

    out: dict[str, xr.Dataset] = {}
    for i, name in enumerate(_ORDER):
        lo, hi = tuple(getattr(splits_cfg, name))
        start = pd.Timestamp(f"{lo}-01-01")
        end = pd.Timestamp(f"{hi}-12-31")
        if i < len(_ORDER) - 1:
            end = end - embargo
        out[name] = ds.sel(time=slice(start, end))
    return out
