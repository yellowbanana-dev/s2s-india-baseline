"""Data-integrity tests - the highest-ROI tests in ML (task #3 guardrails).

These encode the cardinal rule so leakage fails loudly instead of silently
inflating skill. Fill in once the pipeline exists; keep them green forever.
"""
import pytest


@pytest.mark.skip(reason="implement alongside the Stage 2 pipeline")
def test_climatology_uses_train_years_only():
    """fit_climatology must not touch any val/test timestamp."""
    raise NotImplementedError


@pytest.mark.skip(reason="implement alongside the Stage 2 pipeline")
def test_normalizer_stats_from_train_only():
    """Normalization mean/std come only from training anomalies."""
    raise NotImplementedError


@pytest.mark.skip(reason="implement alongside the Stage 2 pipeline")
def test_embargo_gap_between_splits():
    """No sample window spans a split boundary; embargo gap is respected."""
    raise NotImplementedError


@pytest.mark.skip(reason="implement alongside the Stage 2 pipeline")
def test_no_overlap_between_split_indices():
    """train/val/test sample indices are disjoint."""
    raise NotImplementedError
