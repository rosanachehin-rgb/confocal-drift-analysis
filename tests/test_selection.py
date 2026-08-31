"""A fixed threshold selecting on brightness rather than measuring it."""
import numpy as np
import pandas as pd
import pytest

from acqdrift import selection


def counted_session(coupling=0.0, n=24, seed=0):
    """Object counts that either do or do not follow the image background."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 200, n)
    background = 150.0 - 8.0 * t / t.max() + rng.normal(0, 0.4, n)
    counts = 6 + coupling * (background - background.mean())
    return pd.DataFrame({"t_min": t, "group": "Control",
                         "background": background,
                         "n_objects": np.round(counts).astype(int)})


def test_headroom_is_measured_in_background_noise():
    background = np.full(20, 100.0)
    background[0] = 110.0
    value = selection.headroom(background, threshold=120.0)
    assert np.isfinite(value) and value > 0


def test_headroom_is_infinite_when_the_background_never_moves():
    assert not np.isfinite(selection.headroom([100.0] * 10, 250.0))


def test_counts_that_track_brightness_are_flagged():
    df = counted_session(coupling=1.2)
    report = selection.audit_threshold(df, "n_objects", "background")
    assert not report.coupled.empty
    assert report.verdict == "THRESHOLD SELECTING"
    assert "relative to each image" in report.remedy


def test_counts_independent_of_brightness_are_not_flagged():
    df = counted_session(coupling=0.0)
    df["n_objects"] = [6, 7] * (len(df) // 2)
    report = selection.audit_threshold(df, "n_objects", "background")
    assert report.coupled.empty
    assert report.verdict != "THRESHOLD SELECTING"


def test_counts_above_a_biological_ceiling_are_false_positives():
    """No ground truth needed: the preparation cannot contain them."""
    df = counted_session()
    df["n_objects"] = 9
    report = selection.audit_threshold(df, "n_objects", "background", ceiling=6)
    assert report.over_ceiling == len(df)
    assert report.verdict == "THRESHOLD SELECTING"


def test_a_constant_count_is_untestable_not_passing():
    df = counted_session()
    df["n_objects"] = 6
    report = selection.audit_threshold(df, "n_objects", "background")
    assert report.verdict == "NOT TESTABLE"
    assert report.table["untestable_because"].notna().all()


def test_missing_columns_are_reported_not_raised():
    report = selection.audit_threshold(pd.DataFrame({"group": ["a"]}),
                                       "n_objects", "background")
    assert report.verdict == "NOT TESTABLE"
    assert any("missing column" in n for n in report.notes)
