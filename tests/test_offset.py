"""The pedestal changes the answer, and the tests say by how much."""
import numpy as np
import pandas as pd
import pytest

from acqdrift import offset


def session(level_start=150.0, fall=9.0, n=40, pedestal=100.0):
    t = np.linspace(0, 400, n)
    values = level_start - fall * t / t.max()
    return pd.DataFrame({"t_min": t, "background": values,
                         "set_Offset": pedestal})


def test_net_subtracts_the_pedestal():
    assert offset.net([150.0, 140.0], 100.0).tolist() == [50.0, 40.0]


def test_net_refuses_a_pedestal_larger_than_the_signal():
    with pytest.raises(ValueError, match="below the stated offset"):
        offset.net([90.0], 100.0)


def test_constant_offset_reads_metadata():
    value, source = offset.constant_offset(session())
    assert value == 100.0
    assert "set_Offset" in source


def test_constant_offset_refuses_a_pedestal_that_moved():
    df = session()
    df.loc[df.index[-5:], "set_Offset"] = 200.0
    value, source = offset.constant_offset(df)
    assert value is None
    assert "not constant" in source


def test_pedestal_inflates_the_percentage_it_hides():
    """The whole point: same photons, two very different percentages."""
    report = offset.audit_offset(session(), "background")
    # median level 145.5, so 9 counts is 6.2 % of the raw level and 19.8 %
    # of the 45.5 counts that were actually light
    assert report.pct_raw == pytest.approx(-6.2, abs=0.1)
    assert report.pct_net == pytest.approx(-19.8, abs=0.2)
    assert report.inflation == pytest.approx(3.2, abs=0.1)
    assert report.matters


def test_a_negligible_pedestal_does_not_trigger_the_warning():
    df = session(pedestal=1.0)
    report = offset.audit_offset(df, "background")
    assert not report.matters
    assert report.inflation == pytest.approx(1.0, abs=0.05)


def test_missing_pedestal_is_an_error_not_a_guess():
    df = session().drop(columns=["set_Offset"])
    with pytest.raises(ValueError, match="could not establish"):
        offset.audit_offset(df, "background")


def test_add_net_columns_leaves_the_originals_alone():
    df = session()
    out = offset.add_net_columns(df, ["background"], 100.0)
    assert out["background"].equals(df["background"])
    assert out["background_net"].iloc[0] == pytest.approx(50.0)
