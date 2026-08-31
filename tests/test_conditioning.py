"""Error amplification of a background-subtracted integral."""
import numpy as np
import pytest

from acqdrift import conditioning


def test_amplification_is_f_over_one_minus_f():
    assert conditioning.amplification(0.5) == pytest.approx(1.0)
    assert conditioning.amplification(0.9) == pytest.approx(9.0)
    assert conditioning.amplification(0.947) == pytest.approx(17.9, abs=0.1)


def test_amplification_at_the_limits():
    assert conditioning.amplification(0.0) == 0.0
    assert not np.isfinite(conditioning.amplification(1.0))


def test_sparse_labelling_is_ill_conditioned():
    n_vox = 1000.0
    background = np.full(30, 100.0) + np.random.default_rng(0).normal(0, 3, 30)
    raw_sum = background * n_vox / 0.95
    report = conditioning.assess_integral(raw_sum, background, n_vox)
    assert report.background_fraction == pytest.approx(0.95, abs=0.01)
    assert report.amplification == pytest.approx(19.0, abs=1.0)
    assert report.verdict == "ILL-CONDITIONED"


def test_a_bright_field_is_well_conditioned():
    n_vox = 1000.0
    background = np.full(30, 100.0)
    raw_sum = background * n_vox / 0.30
    report = conditioning.assess_integral(raw_sum, background, n_vox)
    assert report.amplification < 0.5
    assert report.verdict == "WELL-CONDITIONED"


def test_tolerable_error_inverts_the_amplification():
    report = conditioning.assess_integral([1000.0] * 10, [90.0] * 10, 10.0)
    assert report.tolerable_background_error_pct(5.0) == pytest.approx(
        5.0 / report.amplification)


def test_peak_is_conditioned_far_better_than_the_integral():
    """Background ruins the integral and barely touches the peak."""
    share = conditioning.assess_peak([6473.0] * 10, [147.0] * 10)
    assert share == pytest.approx(2.3, abs=0.1)
