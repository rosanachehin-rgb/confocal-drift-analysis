"""The sentinel check must reject exactly the datasets it cannot serve."""
import numpy as np
import pandas as pd
import pytest

from acqdrift import check_sentinel, occupancy, partial_spearman


def synthetic_session(signal_fraction, n_images=20, seed=0, jitter=0.5,
                      drift_pct=0.0, noise=0.0):
    """Images whose signal content varies, with an optional background drift."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_images):
        t = i * 5.0
        n_vox = 60000
        base = 150.0 * (1 + drift_pct / 100 * t / 100) + rng.normal(0, noise)
        frac = signal_fraction * rng.uniform(1 - jitter, 1 + jitter)
        v = rng.normal(base, 3, n_vox)
        k = int(frac * n_vox)
        if k:
            v[:k] = rng.normal(900, 200, k)
        rng.shuffle(v)
        rows.append({"group": "ctrl", "t_min": t,
                     "ch.background": float(np.median(v)),
                     "ch.p99": float(np.percentile(v, 99)),
                     "ch.p999": float(np.percentile(v, 99.9))})
    return pd.DataFrame(rows)


def test_sparse_labelling_passes():
    check = check_sentinel(synthetic_session(0.005, seed=1), channel="ch")
    assert check.verdict == "SENTINEL VALID"
    assert check.valid


def test_dense_labelling_is_rejected():
    check = check_sentinel(synthetic_session(0.35, seed=2), channel="ch")
    assert check.verdict == "SENTINEL INVALID"
    assert not check.valid
    assert "ctrl" in set(check.failing["group"])


def test_rejection_survives_a_shared_time_trend():
    """Drift in the background must not be mistaken for content dependence."""
    sparse = check_sentinel(synthetic_session(0.02, seed=3, drift_pct=-10.0,
                                              noise=1.0), channel="ch")
    assert sparse.verdict.startswith("SENTINEL VALID"), (
        "a drifting but sparse session must still pass the sentinel check")


def test_proxy_inside_the_background_is_refused_not_passed():
    """At very sparse labelling a p99 is still background, and must not be
    silently correlated against the sentinel as if it were signal."""
    check = check_sentinel(synthetic_session(0.005, seed=3), channel="ch",
                           signal_proxy="p99")
    assert check.verdict == "NOT TESTABLE"
    assert any("not a signal proxy" in n for n in check.notes)


def test_untestable_group_is_named_with_its_reason():
    df = synthetic_session(0.05, seed=1, n_images=4)
    check = check_sentinel(df, channel="ch")
    assert check.verdict == "NOT TESTABLE"
    assert any("fewer than 6 images" in n for n in check.notes)


def test_occupancy_tracks_truth_below_one_half():
    rng = np.random.default_rng(0)
    for truth in (0.01, 0.10, 0.30, 0.45):
        v = rng.normal(150, 3, 100000)
        k = int(truth * v.size)
        v[:k] = rng.normal(900, 200, k)
        rng.shuffle(v)
        assert occupancy(v) == pytest.approx(truth, abs=0.02)


def test_occupancy_collapses_when_signal_dominates():
    """Documented failure mode: past half the field the estimate reads zero.

    This is why occupancy is descriptive only and never decides a verdict.
    """
    rng = np.random.default_rng(0)
    v = rng.normal(150, 3, 100000)
    k = int(0.75 * v.size)
    v[:k] = rng.normal(900, 200, k)
    rng.shuffle(v)
    assert occupancy(v) < 0.01


def test_partial_correlation_removes_the_shared_trend():
    t = np.arange(30, dtype=float)
    rng = np.random.default_rng(0)
    x = t + rng.normal(0, 1, 30)      # both driven by time only
    y = t + rng.normal(0, 1, 30)
    raw = np.corrcoef(x, y)[0, 1]
    r, p, reason = partial_spearman(x, y, t)
    assert reason is None
    assert raw > 0.9
    assert abs(r) < 0.5, "the shared trend with time should be removed"


def test_missing_columns_are_reported_not_raised():
    df = pd.DataFrame({"group": ["a"] * 8, "t_min": range(8)})
    check = check_sentinel(df, channel="ch")
    assert check.verdict == "NOT TESTABLE"
    assert check.notes
