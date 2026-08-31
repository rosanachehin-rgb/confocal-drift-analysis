"""Sensitivity and specificity of the drift audit against known ground truth.

Real data can show that the audit detects something. Only synthetic data with
an injected slope can show what it detects and what it misses, which is what a
reader should want before trusting a negative result.
"""
import numpy as np
import pandas as pd
import pytest

from acqdrift import analyse, benjamini_hochberg, min_detectable_rho, trend


def session(n_per_group=20, drift_pct_per_hour=0.0, noise=0.02, seed=0,
            groups=("ctrl", "treated")):
    """Two groups, randomly interleaved, with an optional linear drift."""
    rng = np.random.default_rng(seed)
    n = n_per_group * len(groups)
    t = np.sort(rng.uniform(0, 120, n))
    labels = rng.permutation(np.repeat(list(groups), n_per_group))
    baseline = 100.0
    value = baseline * (1 + drift_pct_per_hour / 100 * t / 60)
    value = value * rng.normal(1.0, noise, n)
    return pd.DataFrame({"group": labels, "t_min": t, "signal": value})


def test_no_drift_is_not_reported():
    """Specificity: a clean session must not produce a finding."""
    report = analyse(session(drift_pct_per_hour=0.0, seed=1), metrics=["signal"])
    assert report.significant.empty
    assert report.verdict in ("NO DRIFT DETECTED", "UNDERPOWERED")


def test_false_positive_rate_is_controlled():
    """Across many clean sessions, findings should stay near alpha."""
    hits = 0
    runs = 60
    for seed in range(runs):
        report = analyse(session(drift_pct_per_hour=0.0, seed=seed),
                         metrics=["signal"], alpha=0.05)
        hits += int(not report.significant.empty)
    assert hits / runs < 0.15, f"{hits}/{runs} false positives"


def test_strong_drift_is_detected():
    report = analyse(session(drift_pct_per_hour=20.0, seed=2), metrics=["signal"])
    assert not report.significant.empty
    assert report.verdict == "DRIFT DETECTED"


def test_reported_magnitude_recovers_the_injected_slope():
    injected = 15.0
    report = analyse(session(drift_pct_per_hour=injected, noise=0.01, seed=4),
                     metrics=["signal"])
    for _, row in report.table.iterrows():
        assert row["pct_per_hour"] == pytest.approx(injected, rel=0.35)


def test_small_n_is_called_underpowered_not_clean():
    """A negative result on six images must not read as reassurance."""
    report = analyse(session(n_per_group=3, drift_pct_per_hour=0.0, seed=5),
                     metrics=["signal"])
    assert report.verdict == "UNDERPOWERED"


def test_detection_floor_matches_reported_power():
    """Drift just below the stated detection floor should usually be missed."""
    floor = min_detectable_rho(40)
    assert 0.2 < floor < 0.5
    detected = 0
    for seed in range(40):
        report = analyse(session(n_per_group=20, drift_pct_per_hour=1.5,
                                 noise=0.05, seed=seed), metrics=["signal"])
        detected += int(not report.significant.empty)
    assert detected < 20, "a sub-floor drift was detected too often"


def test_benjamini_hochberg_is_monotone_and_conservative():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.60, 0.99])
    q = benjamini_hochberg(p)
    assert np.all(np.diff(q) >= -1e-12), "q-values must not decrease with p"
    assert np.all(q >= p - 1e-12), "q-values must not fall below raw p"
    assert q.max() <= 1.0


def test_trend_reports_ties():
    t = np.arange(30, dtype=float)
    quantised = np.round(np.linspace(0, 3, 30))
    result = trend(t, quantised)
    assert result["ties_frac"] > 0.5
    assert result["rho"] > 0.9


def offset_session(offset, n=20, drift_counts=-3.0, seed=0):
    """A channel whose real signal sits `offset` counts above zero."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * 5.0
    real = 47.0 + drift_counts * t / t.max()
    return pd.DataFrame({
        "group": "ctrl", "t_min": t,
        "ch.background": offset + real + rng.normal(0, 0.3, n),
        "ch.offset": offset,
        "ch.n_voxels": 60000.0,
    })


def test_offset_is_removed_from_the_denominator():
    """The same absolute drift must read larger once the offset is out."""
    from acqdrift import analyse

    raw = analyse(offset_session(0.0), metrics=["ch.background"]).table.iloc[0]
    shifted = analyse(offset_session(100.0), metrics=["ch.background"]).table.iloc[0]

    assert raw["rho"] == pytest.approx(shifted["rho"], abs=1e-9), (
        "a constant added to every image must not change the rank correlation")
    assert shifted["offset_applied"]
    assert not raw["offset_applied"]
    assert shifted["pct_over_span"] == pytest.approx(raw["pct_over_span"], rel=0.02), (
        "with the offset removed both sessions describe the same real change")


def test_uncorrected_denominator_would_understate_the_drift():
    """Guards the reason the correction exists, with the factor made explicit."""
    from acqdrift import analyse

    df = offset_session(100.0)
    corrected = analyse(df, metrics=["ch.background"]).table.iloc[0]
    naive = analyse(df.drop(columns=["ch.offset"]),
                    metrics=["ch.background"]).table.iloc[0]

    assert not naive["offset_applied"]
    factor = corrected["pct_over_span"] / naive["pct_over_span"]
    assert factor == pytest.approx(147 / 47, rel=0.05)


def test_spread_and_fraction_are_left_alone():
    """mad and sat_frac do not carry the offset, so they must not be adjusted."""
    from acqdrift.drift import signal_level

    df = offset_session(100.0)
    df["ch.mad"] = 2.0
    df["ch.sat_frac"] = 0.0
    for stat in ("mad", "sat_frac"):
        level, applied = signal_level(df, f"ch.{stat}")
        assert not applied, f"{stat} must not be offset-corrected"


def test_sum_carries_the_offset_once_per_voxel():
    from acqdrift.drift import signal_level

    df = offset_session(100.0)
    df["ch.total"] = 100.0 * 60000 + 5_000_000
    level, applied = signal_level(df, "ch.total")
    assert applied
    assert level == pytest.approx(5_000_000, rel=1e-9)


def test_bookkeeping_columns_are_not_treated_as_metrics():
    from acqdrift.drift import metric_columns

    columns = metric_columns(offset_session(100.0), exclude=("t_min",))
    assert "ch.background" in columns
    assert "ch.offset" not in columns
    assert "ch.n_voxels" not in columns
