"""Worked example - C. elegans dopaminergic neurons, 81 stacks, one session.

Runs both audits from the raw .czi files, with no prior processing. The
numbers this prints were the reason the package exists: the session that
produced them was analysed to completion before anyone checked whether the
groups were separable from the time of day.
"""
import sys
from pathlib import Path

from acqdrift import (analyse, audit_schedule, compare_channels, drift,
                      read_session, render_drift, render_power, render_schedule)


def group_from_name(filename):
    """Condition from the filename, tolerating both naming conventions used."""
    n = "".join(filename.lower().split())
    if n.startswith(("control", "buffer")):
        return "Control"
    if n.startswith("olidad9"):
        return "OliDAD9 1/5" if "1en5" in n else "OliDAD9 1/7"
    if n.startswith("olida"):
        return "OliDA 1/5" if "1en5" in n else "OliDA 1/7"
    if n.startswith("dad9"):
        return "DAD9 140uM"
    if n.startswith("da"):
        return "DA 140uM"
    return "unassigned"


def main(raw_dir):
    table = read_session(raw_dir, group_from_name=group_from_name)
    table.to_csv(Path(__file__).parent / "session_metrics.csv", index=False)

    settings = [c for c in table.columns if c.startswith("set_")]
    print(render_schedule(audit_schedule(table, settings_cols=settings)))
    print()

    metrics = drift.metric_columns(
        table, exclude=("t_min", "px_um", "z_um", "n_channels"))
    result = analyse(table, metrics=metrics)
    print(render_drift(result, top=14))
    print()
    print(render_power(result))
    print()
    print("CHANNEL CONTRAST  ZsGr1 (fluorescence) against ESID (transmitted)")
    print(compare_channels(result, "ZsGr1", "ESID",
                           group="Control").to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
