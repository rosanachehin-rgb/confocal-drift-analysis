"""The vendor-neutral record, and what it admits it cannot support."""
import pandas as pd
import pytest

from acqdrift import records


def minimal():
    return pd.DataFrame({
        "file": ["a", "b", "c"], "group": ["x", "x", "y"],
        "timestamp": ["2026-08-13T15:00:00Z", "2026-08-13T15:05:00Z",
                      "2026-08-13T15:20:00Z"]})


def test_finalise_anchors_time_on_the_first_image():
    out = records.finalise(minimal())
    assert out["t_min"].tolist() == [0.0, 5.0, 20.0]


def test_finalise_needs_some_notion_of_time():
    with pytest.raises(ValueError, match="acquisition order"):
        records.finalise(minimal().drop(columns=["timestamp"]))


def test_a_bare_record_still_supports_the_schedule_audit():
    check = records.validate(records.finalise(minimal()))
    assert check.can_audit_schedule
    assert not check.can_audit_drift
    assert not check.can_correct_offset


def test_missing_pedestal_is_a_warning_not_a_failure():
    check = records.validate(records.finalise(minimal()))
    assert any("pedestal" in w for w in check.warnings)


def test_filesystem_times_are_flagged_as_provisional():
    check = records.validate(records.finalise(minimal()),
                             time_source="filesystem")
    assert any("provisional" in w for w in check.warnings)


def test_channels_are_read_off_the_metric_column_names():
    df = records.finalise(minimal())
    df["GFP.background"] = 1.0
    df["ESID.background"] = 2.0
    check = records.validate(df)
    assert check.channels == ["GFP", "ESID"]
    assert check.can_audit_drift


def test_settings_columns_are_recognised_by_prefix():
    df = records.finalise(minimal())
    df["set_Zoom"] = 1.4
    assert records.validate(df).settings == ["set_Zoom"]


def test_a_record_without_the_required_fields_cannot_be_audited():
    check = records.validate(pd.DataFrame({"t_min": [0, 1, 2]}))
    assert check.missing_required == ["file", "group"]
    assert not check.can_audit_schedule


def test_the_message_for_an_unsupported_format_points_somewhere_useful():
    message = records.missing_reader_message(".lif")
    assert "from_csv" in message and "SCHEMA" in message


def test_describe_schema_lists_every_field():
    text = records.describe_schema()
    for name in records.SCHEMA:
        assert name in text
