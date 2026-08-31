"""The combined verdict must never turn an absence of findings into a pass."""
import numpy as np
import pandas as pd

from acqdrift import analyse, assess, audit_schedule, check_sentinel
from acqdrift.fitness import INCONCLUSIVE, NO_EVIDENCE, UNFIT

from test_drift import session as drift_session
from test_schedule import blocked_session, randomised_session
from test_sentinel import synthetic_session


def _drift(**kwargs):
    return analyse(drift_session(**kwargs), metrics=["signal"])


def test_blocked_design_with_drift_is_unfit():
    sched = audit_schedule(blocked_session(), n_perm=500)
    verdict = assess(sched, _drift(drift_pct_per_hour=25.0, seed=2))
    assert verdict.verdict == UNFIT
    assert verdict.unfit
    assert "interleav" in verdict.remedy.lower()


def test_blocked_design_without_drift_is_inconclusive_not_unfit():
    """A confounded design offers no protection, but is not proof of damage."""
    sched = audit_schedule(blocked_session(), n_perm=500)
    verdict = assess(sched, _drift(drift_pct_per_hour=0.0, seed=1))
    assert verdict.verdict == INCONCLUSIVE
    assert not verdict.unfit


def test_clean_session_is_never_called_fit():
    sched = audit_schedule(randomised_session(), n_perm=500)
    verdict = assess(sched, _drift(drift_pct_per_hour=0.0, seed=1))
    assert verdict.verdict == NO_EVIDENCE
    text = " ".join(verdict.reasons + [verdict.remedy]).lower()
    for forbidden in ("pass", "fit for", "good quality", "validated", "clean bill"):
        assert forbidden not in text.replace("clean bill of health", "")


def test_underpowered_is_inconclusive_not_clean():
    sched = audit_schedule(randomised_session(), n_perm=500)
    verdict = assess(sched, _drift(n_per_group=3, drift_pct_per_hour=0.0, seed=5))
    assert verdict.verdict == INCONCLUSIVE


def test_changed_settings_short_circuit_everything():
    df = blocked_session()
    df["set_Voltage"] = ["580"] * (len(df) - 4) + ["620"] * 4
    sched = audit_schedule(df, settings_cols=["set_Voltage"], n_perm=200)
    verdict = assess(sched, _drift(drift_pct_per_hour=0.0, seed=1))
    assert verdict.verdict == UNFIT
    assert "settings" in verdict.reasons[0].lower()


def test_invalid_sentinel_blocks_the_drift_verdict():
    sched = audit_schedule(randomised_session(), n_perm=500)
    bad = check_sentinel(synthetic_session(0.35, seed=2), channel="ch")
    verdict = assess(sched, _drift(drift_pct_per_hour=25.0, seed=2), bad)
    assert verdict.verdict == INCONCLUSIVE
    assert "schedule audit above is unaffected" in " ".join(verdict.reasons)


def test_valid_sentinel_lets_the_drift_verdict_through():
    sched = audit_schedule(blocked_session(), n_perm=500)
    good = check_sentinel(synthetic_session(0.005, seed=1), channel="ch")
    verdict = assess(sched, _drift(drift_pct_per_hour=25.0, seed=2), good)
    assert verdict.verdict == UNFIT
