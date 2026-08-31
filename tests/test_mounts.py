"""Recovering batches from timestamps, and locating the drift."""
import numpy as np
import pandas as pd
import pytest

from acqdrift import mounts


def batched_session(n_mounts=6, per_mount=8, spacing=3.5, gap=25.0,
                    between=-0.02, within=0.0, seed=0):
    rng = np.random.default_rng(seed)
    times, level = [], []
    clock = 0.0
    for _ in range(n_mounts):
        start = clock
        for j in range(per_mount):
            times.append(clock)
            level.append(100.0 + between * clock + within * (clock - start)
                         + rng.normal(0, 0.05))
            clock += spacing
        clock += gap
    return pd.DataFrame({"t_min": times, "level": level})


def test_find_mounts_recovers_the_batches():
    df = batched_session(n_mounts=6, per_mount=8)
    index, gaps, threshold = mounts.find_mounts(df["t_min"].to_numpy())
    assert index.max() + 1 == 6
    assert gaps.size == 5
    assert threshold > 3.5


def test_find_mounts_needs_sorted_times():
    with pytest.raises(ValueError, match="sorted"):
        mounts.find_mounts([0.0, 5.0, 2.0])


def test_a_regular_session_is_one_batch():
    t = np.arange(0, 40, 2.0)
    index, gaps, _ = mounts.find_mounts(t)
    assert index.max() == 0
    assert gaps.size == 0


def test_clock_driven_drift_lands_between_batches():
    report = mounts.decompose(batched_session(between=-0.02, within=0.0),
                              "level")
    assert report.locus == "BETWEEN MOUNTS"
    assert "Randomising" in report.remedy


def test_batch_driven_drift_lands_within_batches():
    report = mounts.decompose(batched_session(between=0.0, within=-0.3),
                              "level")
    assert report.locus in ("WITHIN MOUNTS", "BOTH")
    assert "interleaving will not help" in report.remedy


def test_an_explicit_batch_column_overrides_the_inference():
    df = batched_session()
    df["mount"] = np.repeat(np.arange(6), 8)
    report = mounts.decompose(df, "level", mount_col="mount")
    assert report.n_mounts == 6


def test_too_few_images_refuses_rather_than_guesses():
    with pytest.raises(ValueError, match="at least 6"):
        mounts.decompose(pd.DataFrame({"t_min": [0, 1, 2], "level": [1, 2, 3]}),
                         "level")
