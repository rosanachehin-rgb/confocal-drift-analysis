"""Reproduce the C. elegans day-5 session from the acquisition record.

The session in `examples/celegans_day5` is the one the package was built
around: 81 stacks, five groups, acquired in five contiguous blocks on
13-08-2026. Every number asserted here was first obtained by hand, in the
analysis scripts that preceded the package, and is quoted in the README. The
tests exist so that a change to the code cannot quietly change the worked
example while the prose keeps claiming otherwise.

Tolerances are loose where an aggregation choice is involved and tight where
the quantity is determined. The gain fit and the regression against time are
determined and are asserted to three or four figures.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import acqdrift as ad

DATA = Path(__file__).resolve().parents[1] / "examples" / "celegans_day5"
OFFSET = 100.0
N_VOXELS = 512 * 512 * 10


@pytest.fixture(scope="module")
def record():
    table, _ = ad.from_csv(DATA / "acquisition_record.csv")
    return table


@pytest.fixture(scope="module")
def correction():
    return pd.read_csv(DATA / "correction_check.csv")


def test_the_record_supports_every_audit(record):
    check = ad.validate(record)
    assert len(record) == 81
    assert check.can_audit_schedule and check.can_audit_drift
    assert check.can_correct_offset
    assert check.channels == ["GFP", "ESID"]


def test_the_five_groups_never_overlap_in_time(record):
    report = ad.audit_schedule(record)
    assert report.n_groups == 5
    assert report.session_minutes == pytest.approx(427.7, abs=0.1)
    assert report.eta_squared == pytest.approx(0.962, abs=0.002)
    assert report.verdict == "CONFOUNDED"
    assert int((report.overlap["overlap_min"] > 0).sum()) == 0


def test_the_background_falls_linearly_across_the_session(record):
    from scipy import stats
    fit = stats.linregress(record["t_min"], record["GFP.background"])
    assert fit.slope == pytest.approx(-0.0205, abs=0.0002)
    assert fit.rvalue ** 2 == pytest.approx(0.791, abs=0.002)
    assert fit.pvalue < 1e-25


def test_the_pedestal_triples_the_reported_drift(record):
    report = ad.audit_offset(record, "GFP.background", offset=OFFSET)
    assert report.change_counts == pytest.approx(-8.5, abs=0.3)
    assert report.pct_raw == pytest.approx(-6.0, abs=0.3)
    assert report.pct_net == pytest.approx(-20.2, abs=0.6)
    assert report.inflation == pytest.approx(3.4, abs=0.15)
    assert report.matters


def test_the_transmitted_channel_stays_put_while_fluorescence_falls(record):
    """The contrast that moves the suspicion from excitation to detection."""
    gfp = ad.audit_offset(record, "GFP.background", offset=OFFSET)
    esid = ad.audit_offset(record, "ESID.background", offset=OFFSET)
    assert gfp.pct_net < -15.0
    assert abs(esid.pct_net) < 5.0
    assert esid.p > 0.05
    assert abs(gfp.pct_net) > 10 * abs(esid.pct_net)


def test_the_integral_is_ill_conditioned(record):
    report = ad.assess_integral(record["GFP.total"], record["GFP.background"],
                                N_VOXELS)
    assert report.background_fraction == pytest.approx(0.949, abs=0.005)
    assert report.amplification == pytest.approx(18.7, abs=0.5)
    assert report.induced_integral_cv_pct > 25
    assert report.verdict == "ILL-CONDITIONED"
    assert report.tolerable_background_error_pct(5.0) < 0.3


def test_nine_mounts_are_recovered_from_the_timestamps(record):
    report = ad.decompose(record, "GFP.background")
    assert report.n_mounts == 9
    assert report.median_interval_min == pytest.approx(3.6, abs=0.1)
    assert report.mean_mount_minutes == pytest.approx(29.4, abs=0.5)


def test_the_drift_lives_between_mounts_not_inside_them(record):
    report = ad.decompose(record, "GFP.background")
    assert report.between_rho == pytest.approx(-0.917, abs=0.01)
    assert report.between_p < 0.001
    assert report.between_slope == pytest.approx(-0.0230, abs=0.0005)

    assert report.within_rho == pytest.approx(-0.009, abs=0.01)
    assert report.within_f == pytest.approx(0.27, abs=0.05)
    assert report.within_f_p == pytest.approx(0.60, abs=0.03)
    assert report.r2_time == pytest.approx(0.7913, abs=0.001)
    assert report.r2_time_plus_within == pytest.approx(0.7920, abs=0.001)
    assert report.locus == "BETWEEN MOUNTS"


def test_the_within_mount_null_is_reported_with_its_confidence_interval(record):
    """A flat result on a short window must not read as a clean negative."""
    report = ad.decompose(record, "GFP.background")
    assert report.within_slope_lo == pytest.approx(-0.037, abs=0.002)
    assert report.within_slope_hi == pytest.approx(+0.021, abs=0.002)
    # the interval still admits an effect the size of the global drift
    assert report.within_slope_lo < -0.0205


def test_the_gain_curve_matches_the_published_fit(record):
    fit = ad.fit_gain(record["t_min"], record["GFP.background"], offset=OFFSET)
    assert fit.a == pytest.approx(39.01, abs=0.1)
    assert fit.b == pytest.approx(10.68, abs=0.1)
    assert fit.tau_min == pytest.approx(171.9, abs=1.0)
    assert fit.k_first == pytest.approx(1.0)
    assert fit.k_last == pytest.approx(0.803, abs=0.005)


def test_the_correction_is_rejected_by_the_negative_control(correction):
    """Clean fit, correct arithmetic, and the artefact grew."""
    check = ad.validate_correction(correction, "peak_raw", "peak_corrected",
                                   control_group="Control")
    assert check.artefact_raw_pct == pytest.approx(57.2, abs=1.0)
    assert check.artefact_corrected_pct == pytest.approx(62.1, abs=1.0)
    assert check.rho_raw == pytest.approx(0.201, abs=0.01)
    assert check.rho_corrected == pytest.approx(0.316, abs=0.01)
    assert check.reduction_pct < 0
    assert check.verdict == "CORRECTION REJECTED"


def test_area_is_invariant_to_the_correction_and_peak_is_not(correction):
    """An endpoint defined as a within-object ratio sidesteps the gain."""
    area = ad.scale_invariance(correction["area_raw"],
                               correction["area_corrected"])
    peak = ad.scale_invariance(correction["peak_raw"],
                               correction["peak_corrected"])
    assert abs(area) < 0.5
    assert peak > 10.0


def test_the_absolute_threshold_is_selecting(correction):
    record, _ = ad.from_csv(DATA / "acquisition_record.csv")
    merged = correction.merge(record[["file", "GFP.background"]], on="file")
    report = ad.audit_threshold(merged, count_col="n_objects_raw",
                                level_col="GFP.background", threshold=250.0,
                                ceiling=6)
    assert report.over_ceiling == 53
    assert report.n_images == 81
    assert report.verdict == "THRESHOLD SELECTING"


def test_the_correction_manufactures_detections_where_it_amplifies_most(correction):
    """Dividing by k raises the noise with the mean, and the threshold admits it."""
    assert (correction["n_objects_raw"] > 8).sum() == 18
    assert (correction["n_objects_corrected"] > 8).sum() == 21

    dad9 = correction[correction["group"] == "DAD9 140uM"]
    assert dad9["n_objects_raw"].mean() == pytest.approx(8.50, abs=0.01)
    assert dad9["n_objects_corrected"].mean() == pytest.approx(9.00, abs=0.01)


def test_the_session_is_declared_unfit(record, correction):
    schedule = ad.audit_schedule(record)
    merged = correction.merge(record[["file", "GFP.background"]], on="file")
    selection = ad.audit_threshold(merged, count_col="n_objects_raw",
                                   level_col="GFP.background", ceiling=6)
    verdict = ad.assess(schedule, selection_report=selection)
    assert verdict.unfit
    assert any("left no record" in r for r in verdict.reasons)


# --------------------------------------------------------------------------
# The prose and the code have to agree. Every figure quoted in the new README
# sections is recomputed here, so a change to a threshold or an estimator
# cannot leave the documentation claiming something the package no longer does.
# --------------------------------------------------------------------------

DOCS = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def readme():
    return (DOCS / "README.md").read_text()


@pytest.fixture(scope="module")
def output_md():
    return (DOCS / "OUTPUT.md").read_text()


def test_readme_quotes_the_pedestal_figures(record, readme):
    report = ad.audit_offset(record, "GFP.background", offset=OFFSET)
    assert f"{report.pct_raw:.1f}" == "-6.0"
    assert f"{report.pct_net:.1f}" == "-20.1"
    assert f"{report.inflation:.1f}" == "3.4"
    for quoted in ("-6.0 %", "-20.1 %", "factor of 3.4", "(142.0)", "(42.0)"):
        assert quoted in readme


def test_readme_quotes_the_conditioning_figures(record, readme):
    report = ad.assess_integral(record["GFP.total"], record["GFP.background"],
                                N_VOXELS)
    assert f"{100 * report.background_fraction:.1f}" == "94.9"
    assert f"{report.amplification:.1f}" == "18.7"
    for quoted in ("94.9 %", "18.7", "2.1 %", "40 % of noise"):
        assert quoted in readme


def test_readme_quotes_the_batch_figures(record, readme):
    report = ad.decompose(record, "GFP.background")
    assert report.n_mounts == 9
    for quoted in ("9 batches", "-0.917", "0.0005", "-0.0230",
                   "0.7913 -> 0.7920", "[-0.037, +0.021]"):
        assert quoted in readme


def test_readme_quotes_the_gain_fit(record, readme):
    fit = ad.fit_gain(record["t_min"], record["GFP.background"], offset=OFFSET)
    assert f"{fit.a:.2f}" == "39.01"
    assert f"{fit.b:.2f}" == "10.68"
    assert f"{fit.tau_min:.1f}" == "171.9"
    for quoted in ("39.01", "10.68", "171.9", "1.000 to 0.803"):
        assert quoted in readme


def test_readme_quotes_the_negative_control(correction, readme):
    check = ad.validate_correction(correction, "peak_raw", "peak_corrected")
    assert f"{check.artefact_raw_pct:.1f}" == "57.2"
    assert f"{check.artefact_corrected_pct:.1f}" == "62.1"
    for quoted in ("+57.2 %", "+62.1 %", "+0.201", "+0.316",
                   "CORRECTION REJECTED"):
        assert quoted in readme


def test_readme_quotes_the_selection_figures(record, correction, readme):
    merged = correction.merge(record[["file", "GFP.background"]], on="file")
    report = ad.audit_threshold(merged, count_col="n_objects_raw",
                                level_col="GFP.background", ceiling=6)
    assert report.over_ceiling == 53
    for quoted in ("53 of 81", "18 to 21 of 81", "8.50 to 9.00"):
        assert quoted in readme


def test_the_thresholds_named_in_the_docs_are_the_ones_applied(readme, output_md):
    """The two conventions the README admits are conventions."""
    from acqdrift.conditioning import ConditioningReport
    fragile = ConditioningReport(0.9, 0.1, 9.0, 1.2, 11.0, 10)
    ill = ConditioningReport(0.95, 0.05, 19.0, 1.4, 26.0, 10)
    assert fragile.verdict == "FRAGILE"
    assert ill.verdict == "ILL-CONDITIONED"
    assert "25 %" in readme and "1.5" in readme
    assert "25 % induced noise" in output_md
