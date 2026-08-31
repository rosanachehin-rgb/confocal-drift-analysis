"""Fitting a correction, and refusing to believe it without a control."""
import numpy as np
import pandas as pd
import pytest

from acqdrift import sensitivity


def decaying(n=40, a=39.0, b=11.0, tau=170.0, offset=100.0):
    t = np.linspace(0, 400, n)
    return t, offset + a + b * np.exp(-t / tau)


def test_fit_recovers_the_curve_it_was_given():
    t, level = decaying()
    fit = sensitivity.fit_gain(t, level, offset=100.0)
    assert fit.a == pytest.approx(39.0, rel=0.02)
    assert fit.b == pytest.approx(11.0, rel=0.05)
    assert fit.tau_min == pytest.approx(170.0, rel=0.05)
    assert fit.k_first == pytest.approx(1.0)
    assert fit.k_last < 1.0


def test_fitting_with_the_pedestal_left_in_flattens_the_curve():
    """Why the offset argument exists at all."""
    t, level = decaying()
    honest = sensitivity.fit_gain(t, level, offset=100.0)
    naive = sensitivity.fit_gain(t, level, offset=0.0)
    assert abs(naive.span_pct) < abs(honest.span_pct) / 2


def test_fit_refuses_a_pedestal_that_swallows_the_signal():
    t, level = decaying()
    with pytest.raises(ValueError, match="offset"):
        sensitivity.fit_gain(t, level, offset=200.0)


def test_apply_gain_leaves_the_pedestal_unscaled():
    """(raw - offset)/k + offset, not raw/k."""
    corrected = sensitivity.apply_gain([150.0], k=0.5, offset=100.0)
    assert corrected[0] == pytest.approx(200.0)
    assert corrected[0] != pytest.approx(300.0)


def test_apply_gain_round_trips():
    raw = np.array([150.0, 140.0, 130.0])
    k = np.array([1.0, 0.9, 0.8])
    back = sensitivity.apply_gain(raw, k, 100.0)
    assert np.allclose((back - 100.0) * k + 100.0, raw)


def control_pair(raw_artefact, corrected_artefact, n=20):
    t = np.linspace(0, 100, n)
    ramp = np.linspace(0, 1, n)
    return pd.DataFrame({
        "t_min": t, "group": "Control",
        "raw": 100.0 * (1 + raw_artefact / 100.0 * ramp),
        "corrected": 100.0 * (1 + corrected_artefact / 100.0 * ramp)})


def test_a_correction_that_makes_the_artefact_worse_is_rejected():
    """The finding this module exists for."""
    check = sensitivity.validate(control_pair(51.5, 56.2), "raw", "corrected")
    assert check.verdict == "CORRECTION REJECTED"
    assert check.reduction_pct < 0
    assert "not what the correction models" in check.note


def test_a_correction_that_removes_the_artefact_is_reported_as_such():
    check = sensitivity.validate(control_pair(50.0, 2.0), "raw", "corrected")
    assert check.verdict == "CORRECTION REDUCES ARTEFACT"
    assert check.reduction_pct > 90
    assert "not a guarantee" in check.note


def test_a_half_measure_is_called_insufficient():
    check = sensitivity.validate(control_pair(50.0, 35.0), "raw", "corrected")
    assert check.verdict == "CORRECTION INSUFFICIENT"


def test_validate_needs_a_control_group():
    with pytest.raises(ValueError, match="no rows for control group"):
        sensitivity.validate(control_pair(10, 5), "raw", "corrected",
                             control_group="Missing")


def test_scale_invariant_endpoints_are_recognised():
    values = np.array([5.0, 6.0, 7.0])
    assert sensitivity.scale_invariance(values, values) == pytest.approx(0.0)
    assert sensitivity.scale_invariance(values, values * 1.2) == pytest.approx(20.0)


def test_sweep_tau_returns_one_row_per_tau():
    t, level = decaying()
    df = pd.DataFrame({"t_min": t, "level": level})
    out = sensitivity.sweep_tau(df, "level", offset=100.0, taus=(60, 170, 400))
    assert len(out) == 3
    assert out["r2"].max() == pytest.approx(1.0, abs=0.01)
