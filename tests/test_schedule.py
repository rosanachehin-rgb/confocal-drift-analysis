"""The schedule audit must separate a blocked design from a randomised one."""
import numpy as np
import pandas as pd
import pytest

from acqdrift import audit_schedule, eta_squared
from acqdrift.schedule import interleaved_order

GROUPS = {"ctrl": 20, "a": 15, "b": 15, "c": 10}


def blocked_session():
    labels, times, clock = [], [], 0.0
    for group, n in GROUPS.items():
        for _ in range(n):
            labels.append(group)
            times.append(clock)
            clock += 5.0
        clock += 30.0  # gap between blocks
    return pd.DataFrame({"group": labels, "t_min": times})


def randomised_session(seed=0):
    rng = np.random.default_rng(seed)
    labels = np.concatenate([[g] * n for g, n in GROUPS.items()])
    rng.shuffle(labels)
    return pd.DataFrame({"group": labels,
                         "t_min": np.arange(labels.size) * 5.0})


def test_blocked_design_is_flagged_as_confounded():
    report = audit_schedule(blocked_session(), n_perm=2000)
    assert report.eta_squared > 0.90
    assert report.verdict == "CONFOUNDED"
    assert not report.separable
    assert (report.overlap["overlap_min"] == 0).all()


def test_randomised_design_passes():
    report = audit_schedule(randomised_session(), n_perm=2000)
    assert report.eta_squared < 0.20
    assert report.verdict == "OK"
    assert report.separable


def test_eta_squared_bounds():
    times = np.arange(20, dtype=float)
    assert eta_squared(times, ["a"] * 20) == pytest.approx(0.0, abs=1e-9)
    perfect = ["a"] * 10 + ["b"] * 10
    assert eta_squared(times, perfect) > 0.75


def test_longest_streak_detects_blocking():
    blocked = audit_schedule(blocked_session(), n_perm=500)
    randomised = audit_schedule(randomised_session(), n_perm=500)
    assert blocked.max_run == 20            # the largest block
    assert randomised.max_run < blocked.max_run
    assert blocked.max_run > blocked.max_run_expected


def test_settings_change_is_reported():
    df = blocked_session()
    df["set_Voltage"] = ["580"] * (len(df) - 5) + ["600"] * 5
    df["set_Power"] = "10"
    report = audit_schedule(df, settings_cols=["set_Voltage", "set_Power"],
                            n_perm=200)
    assert "set_Voltage" in report.settings_varying
    assert "set_Power" in report.settings_constant


def test_interleaved_order_is_balanced_and_separable():
    order = interleaved_order(GROUPS, seed=3)
    assert len(order) == sum(GROUPS.values())
    for group, n in GROUPS.items():
        assert order.count(group) == n
    df = pd.DataFrame({"group": order,
                       "t_min": np.arange(len(order), dtype=float) * 5.0})
    report = audit_schedule(df, n_perm=1000)
    assert report.separable, "the proposed order should not confound group with time"
