"""Numeric claims about the worked example, recomputed from the shipped table.

Prose-level checks against README.md and OUTPUT.md live in
test_day5_reproduction.py, which is the authority on what the documents say.
This file covers the quantities that file does not: the shape of
session_metrics_day5.csv and the detector-offset correction applied to
percentages.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

from acqdrift import (analyse, audit_schedule, check_sentinel, compare_channels,
                      drift, min_detectable_rho)
from acqdrift.fitness import UNFIT
from acqdrift import assess

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "celegans_day5" / "session_metrics_day5.csv"

pytestmark = pytest.mark.skipif(not EXAMPLE.exists(),
                                reason="example table not shipped")


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(EXAMPLE)


@pytest.fixture(scope="module")
def results(table):
    metrics = drift.metric_columns(
        table, exclude=("t_min", "px_um", "z_um", "n_channels"))
    settings = [c for c in table.columns if c.startswith("set_")]
    return {
        "schedule": audit_schedule(table, settings_cols=settings, n_perm=2000),
        "drift": analyse(table, metrics=metrics),
        "sentinel": check_sentinel(table, channel="ZsGr1"),
        "metrics": metrics,
    }


@pytest.fixture(scope="module")
def docs():
    """Documents with whitespace collapsed, so a quoted phrase still matches
    when the source file happens to wrap it across two lines."""
    return {name: re.sub(r"\s+", " ", (ROOT / name).read_text())
            for name in ("README.md", "OUTPUT.md")}


def test_example_table_shape(table, docs):
    assert len(table) == 81
    # 31 base columns plus offset and n_voxels for each of the two channels
    assert len(table.columns) == 35
    for channel in ("ZsGr1", "ESID"):
        assert f"{channel}.offset" in table.columns
        assert f"{channel}.n_voxels" in table.columns


def test_session_length_and_group_windows(results, docs):
    sched = results["schedule"]
    assert round(sched.session_minutes) == 428
    windows = {
        "Control": (0.0, 97.3), "OliDA 1/5": (145.8, 223.8),
        "OliDAD9 1/5": (234.7, 328.2), "DA 140uM": (351.1, 386.8),
        "DAD9 140uM": (399.1, 427.7),
    }
    for group, (start, end) in windows.items():
        row = sched.blocks.loc[group]
        assert row["start"] == pytest.approx(start, abs=0.1)
        assert row["end"] == pytest.approx(end, abs=0.1)
        assert f"{start:.1f}" in docs["README.md"]


def test_confounding_statistics_as_quoted(results, docs):
    sched = results["schedule"]
    assert sched.eta_squared == pytest.approx(0.962, abs=0.001)
    assert (sched.overlap["overlap_min"] > 0).sum() == 0
    assert len(sched.overlap) == 10
    assert sched.max_run == 21
    assert sched.max_run_expected == pytest.approx(3.5, abs=0.4)
    assert sched.verdict == "CONFOUNDED"
    for quoted in ("0.962", "0 of 10", "21", "3.5"):
        assert quoted in docs["README.md"]


def test_control_drift_as_quoted(results, docs):
    control = results["drift"].table.query("group == 'Control'") \
                                   .set_index("metric")
    background = control.loc["ZsGr1.background"]
    assert background["rho"] == pytest.approx(-0.761, abs=0.001)
    assert background["q"] == pytest.approx(0.002, abs=0.001)
    assert control.loc["ZsGr1.p01", "rho"] == pytest.approx(-0.758, abs=0.001)
    assert control.loc["ESID.background", "rho"] == pytest.approx(-0.089, abs=0.001)
    assert control.loc["ESID.background", "q"] > 0.9
    for quoted in ("-0.761", "-0.758", "-0.089"):
        assert quoted in docs["README.md"]


def test_offset_correction_changes_magnitudes_only(results, table):
    """Percentages are taken against signal above the detector offset."""
    control = results["drift"].table.query("group == 'Control'") \
                                    .set_index("metric")
    background = control.loc["ZsGr1.background"]
    assert background["offset_applied"]
    assert background["pct_over_span"] == pytest.approx(-6.0, abs=0.15)
    assert control.loc["ZsGr1.p01", "pct_over_span"] == pytest.approx(-8.7, abs=0.15)
    assert not control.loc["ESID.background", "offset_applied"], (
        "ESID has no offset and must not be adjusted")
    # the same drift read against the raw background instead of against signal
    # above the offset, which is the mistake the correction exists to prevent
    raw_level = table.query("group == 'Control'")["ZsGr1.background"].median()
    net_level = raw_level - 100.0
    assert background["pct_over_span"] == pytest.approx(
        -1.9 * raw_level / net_level, abs=0.2)


def test_family_size_quoted_in_docs(results, docs):
    n_tests = len(results["drift"].table)
    assert n_tests == 70, f"family size changed to {n_tests}"
    assert "70 of them" in docs["README.md"]
    assert "5 groups × 14 metrics = 70 rows" in docs["OUTPUT.md"]


def test_sensitivity_claim_of_eleven_versus_fortynine(results, docs):
    """The headline comparison: the cheap sentinel beats the endpoint."""
    control = results["drift"].table.query("group == 'Control'") \
                                    .set_index("metric")
    rho_background = abs(control.loc["ZsGr1.background", "rho"])
    n_background = _n_required(rho_background)
    assert n_background == 11
    # the endpoint is not in the raw table; its rho comes from the segmented
    # analysis this package deliberately does not perform
    assert _n_required(0.391) == 49
    assert "11 images and the segmented cell area needs 49" in docs["OUTPUT.md"]
    assert "about 11 images" in docs["README.md"]
    assert "about 49" in docs["README.md"]


def test_sentinel_separation_range_as_quoted(results, docs):
    separation = results["sentinel"].table["separation"]
    assert separation.min() == pytest.approx(5.3, abs=0.15)
    assert separation.max() == pytest.approx(6.5, abs=0.15)
    assert results["sentinel"].valid
    assert "5.3 to 6.5" in docs["README.md"]


def test_detection_floor_table_as_quoted(docs):
    for n, quoted in ((10, 0.79), (15, 0.67), (30, 0.49)):
        assert min_detectable_rho(n) == pytest.approx(quoted, abs=0.005)
    assert "0.79 at n = 10" in docs["README.md"]


def test_channel_contrast_reads_fluorescence_path(results, docs):
    contrast = compare_channels(results["drift"], "ZsGr1", "ESID",
                                group="Control").set_index("metric")
    assert contrast.loc["background", "interpretation"] == "fluorescence path"
    assert "fluorescence path" in docs["README.md"]


def test_overall_verdict_is_unfit(results, docs):
    verdict = assess(results["schedule"], results["drift"], results["sentinel"])
    assert verdict.verdict == UNFIT
    assert "`UNFIT`" in docs["README.md"]


def test_confounding_thresholds_match_the_code(docs):
    """The bands quoted in OUTPUT.md must be the ones the code applies."""
    import inspect

    from acqdrift import schedule as schedule_module
    source = inspect.getsource(schedule_module.ScheduleReport)
    assert "0.90" in source and "0.60" in source
    assert "η² < 0.60" in docs["OUTPUT.md"]
    assert "0.60–0.90" in docs["OUTPUT.md"]


def test_settings_tag_count_matches_the_code(docs):
    from acqdrift.io import SETTING_TAGS

    assert len(SETTING_TAGS) == 10
    assert "Ten tags are looked for" in docs["OUTPUT.md"]
    for tag in SETTING_TAGS:
        assert tag in docs["OUTPUT.md"], f"{tag} missing from OUTPUT.md"


def _n_required(rho, alpha_z=1.96, power_z=0.84):
    import numpy as np
    z = np.arctanh(min(abs(rho), 0.999))
    return int(np.ceil((alpha_z + power_z) ** 2 / z ** 2 + 3))
