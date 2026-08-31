"""The eta^2 bands are only meaningful for a few groups over many images.

These tests came out of running Level 1 against public depositions in the Image
Data Resource, where an automatically chosen "group" column was sometimes a
replicate identifier with fifty levels. eta^2 came back at 1.000 and the audit
called the session confounded, which said more about the number of groups than
about the schedule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acqdrift import audit_schedule, fitness
from acqdrift.schedule import MAX_ETA_NULL, MIN_GROUP_N
from acqdrift.stats import epsilon_squared, eta_squared, eta_squared_null


def _session(n_per_group, n_groups, blocked, start=0.0, step=1.0, seed=0):
    """Build a schedule with a given block structure.

    blocked=True gives each group its own contiguous window; blocked=False
    interleaves them in a fixed round-robin, which is the cleanest possible
    design and should never be called confounded.
    """
    rng = np.random.default_rng(seed)
    labels = np.concatenate([[f"g{g}"] * n_per_group for g in range(n_groups)])
    if not blocked:
        labels = np.array([f"g{i % n_groups}" for i in range(len(labels))])
    t = start + step * np.arange(len(labels), dtype=float)
    return pd.DataFrame({"group": labels, "t_min": t})


# --- the statistic itself -------------------------------------------------

def test_eta_null_is_the_chance_floor():
    # assigning n observations at random to k groups explains (k-1)/(n-1)
    assert eta_squared_null(81, 5) == pytest.approx(4 / 80)
    assert eta_squared_null(200, 50) == pytest.approx(49 / 199)
    assert np.isnan(eta_squared_null(3, 5))     # more groups than images
    assert np.isnan(eta_squared_null(10, 1))    # a single group explains nothing


def test_random_labels_land_near_the_chance_floor_not_near_zero():
    """The floor is a real effect, not a theoretical curiosity."""
    rng = np.random.default_rng(0)
    t = np.arange(200, dtype=float)
    observed = []
    for _ in range(60):
        lab = rng.permutation(np.repeat(np.arange(50), 4))
        observed.append(eta_squared(t, lab))
    mean = float(np.mean(observed))
    assert mean == pytest.approx(eta_squared_null(200, 50), abs=0.05)
    assert mean > 0.20   # nowhere near zero, which is the trap


def test_epsilon_squared_removes_the_floor():
    rng = np.random.default_rng(1)
    t = np.arange(200, dtype=float)
    vals = [epsilon_squared(t, rng.permutation(np.repeat(np.arange(50), 4)))
            for _ in range(60)]
    assert float(np.mean(vals)) == pytest.approx(0.0, abs=0.05)


# --- the guard ------------------------------------------------------------

def test_many_tiny_groups_are_not_called_confounded():
    """The IDR case: 50 groups of 4, each acquired as a consecutive batch."""
    d = _session(n_per_group=4, n_groups=50, blocked=True)
    r = audit_schedule(d, n_perm=200)
    assert r.eta_squared > 0.99          # the raw number really is ~1
    assert r.min_group_n == 4
    assert r.eta_null > MAX_ETA_NULL
    assert not r.bands_reliable
    assert r.verdict == "INCONCLUSIVE"
    assert "chance floor" in r.verdict_reason or "group holds" in r.verdict_reason


def test_the_day5_shape_still_reaches_a_verdict():
    """Five groups, ~16 images each, strictly blocked: the real session."""
    d = _session(n_per_group=16, n_groups=5, blocked=True)
    r = audit_schedule(d, n_perm=200)
    assert r.min_group_n >= MIN_GROUP_N
    assert r.eta_null <= MAX_ETA_NULL
    assert r.bands_reliable
    assert r.verdict == "CONFOUNDED"


def test_a_well_interleaved_session_still_passes():
    d = _session(n_per_group=16, n_groups=5, blocked=False)
    r = audit_schedule(d, n_perm=200)
    assert r.bands_reliable
    assert r.verdict == "OK"
    assert r.separable


def test_a_single_undersized_group_withholds_the_verdict():
    """One group with fewer than MIN_GROUP_N images is enough to withhold."""
    d = pd.concat([_session(n_per_group=20, n_groups=3, blocked=True),
                   pd.DataFrame({"group": ["tiny"] * 2, "t_min": [61.0, 62.0]})])
    r = audit_schedule(d, n_perm=200)
    assert r.min_group_n == 2
    assert r.verdict == "INCONCLUSIVE"
    assert "at least" in r.verdict_reason


def test_separable_is_false_when_the_bands_do_not_apply():
    """`separable` must not read as clean on a design it cannot judge."""
    d = _session(n_per_group=4, n_groups=50, blocked=False)
    r = audit_schedule(d, n_perm=200)
    assert not r.bands_reliable
    assert not r.separable          # even though this design is interleaved


# --- it reaches the combined verdict --------------------------------------

def test_fitness_reports_inconclusive_rather_than_a_clean_bill():
    d = _session(n_per_group=4, n_groups=50, blocked=True)
    r = audit_schedule(d, n_perm=200)
    f = fitness.assess(r)
    assert f.verdict == fitness.INCONCLUSIVE
    assert not f.unfit
    joined = " ".join(f.reasons).lower()
    assert "schedule" in joined
    assert "chance floor" in joined or "replicate" in joined


def test_fitness_still_flags_the_blocked_five_group_session():
    d = _session(n_per_group=16, n_groups=5, blocked=True)
    f = fitness.assess(audit_schedule(d, n_perm=200))
    assert f.verdict == fitness.INCONCLUSIVE     # confounded, no drift supplied
    assert "separate blocks" in " ".join(f.reasons)


# --- the printed report ---------------------------------------------------

def test_report_prints_the_floor_and_the_reason():
    from acqdrift.report import render_schedule
    d = _session(n_per_group=4, n_groups=50, blocked=True)
    text = render_schedule(audit_schedule(d, n_perm=200))
    assert "Chance floor" in text
    assert "Bias-corrected" in text
    assert "VERDICT: INCONCLUSIVE" in text
    assert "not a" in text and "clean result" in text
