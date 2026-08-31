"""Command line entry point."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from . import (conditioning, drift, fitness, io, mounts, offset, records,
               report, schedule, selection, sentinel)


def group_from_regex(pattern):
    rx = re.compile(pattern, re.IGNORECASE)
    def inner(name):
        m = rx.search(name)
        return m.group(1) if m else "unassigned"
    return inner


def build_parser():
    p = argparse.ArgumentParser(
        prog="acqdrift",
        description="Pre-flight quality control for grouped imaging sessions.")
    p.add_argument("directory", type=Path, nargs="?",
                   help="folder of acquisitions (.czi)")
    p.add_argument("--from-csv", type=Path, default=None,
                   help="audit an acquisition record you built yourself, for "
                        "any microscope this package has no reader for")
    p.add_argument("--schema", action="store_true",
                   help="print the acquisition record layout and exit")
    p.add_argument("--pattern", default="*.czi")
    p.add_argument("--group-regex", default=r"^([A-Za-z0-9 ]+?)\s*stack",
                   help="regex whose first capture group is the condition")
    p.add_argument("--schedule-only", action="store_true",
                   help="metadata only; never opens pixel data")
    p.add_argument("--signal-channel", default=None)
    p.add_argument("--reference-channel", default=None,
                   help="treatment-independent channel, e.g. transmitted light")
    p.add_argument("--sentinel", default="background",
                   help="metric used as the drift sentinel")
    p.add_argument("--offset", type=float, default=None,
                   help="detector pedestal in counts. Read from metadata when "
                        "omitted. Without it every drift percentage is "
                        "understated, often several-fold")
    p.add_argument("--count-col", default=None,
                   help="per-image object count, for the threshold check")
    p.add_argument("--threshold", type=float, default=None,
                   help="the absolute detection threshold that was applied")
    p.add_argument("--ceiling", type=float, default=None,
                   help="largest object count the preparation can contain")
    p.add_argument("--raw-sum-col", default=None,
                   help="per-image raw sum, for the conditioning check")
    p.add_argument("--n-voxels", type=float, default=None,
                   help="voxels per image, for the conditioning check")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--csv", type=Path, default=None,
                   help="write the per-image table here")
    return p


def _load(args):
    """Return (table, record_check). Either route ends in the same record."""
    if args.from_csv:
        return records.from_csv(args.from_csv)
    if not args.directory:
        raise SystemExit("give a directory of acquisitions, or --from-csv")
    if not args.directory.is_dir():
        raise SystemExit(f"{args.directory} is not a directory")
    table = io.read_session(args.directory,
                            group_from_name=group_from_regex(args.group_regex),
                            pattern=args.pattern,
                            with_pixels=not args.schedule_only)
    return table, records.validate(table, time_source="czi metadata")


def _sentinel_column(table, args):
    """The column the pedestal and batch checks are run on."""
    if args.signal_channel:
        column = f"{args.signal_channel}.{args.sentinel}"
        if column in table.columns:
            return column
    for column in table.columns:
        if column.endswith(f".{args.sentinel}"):
            return column
    return None


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.schema:
        print(records.describe_schema())
        return 0

    table, check = _load(args)
    if args.csv:
        table.to_csv(args.csv, index=False)

    print(report.render_records(check))
    if not check.can_audit_schedule:
        print("\nNot enough to audit. See the record layout with --schema.")
        return 2
    print()

    settings = [c for c in table.columns if c.startswith("set_")]
    sched = schedule.audit_schedule(table, settings_cols=settings)
    print(report.render_schedule(sched))

    if args.schedule_only:
        verdict = fitness.assess(sched)
        print()
        print(report.render_fitness(verdict))
        return 1 if verdict.unfit else 0

    print()
    metrics = drift.metric_columns(table, exclude=("t_min", "px_um", "z_um",
                                                   "n_channels"))
    dr = drift.analyse(table, metrics=metrics, alpha=args.alpha)
    print(report.render_drift(dr))
    print()
    print(report.render_power(dr))

    column = _sentinel_column(table, args)
    pedestal = args.offset
    if pedestal is None:
        pedestal, _ = offset.constant_offset(table)

    if column and pedestal is not None:
        print()
        print(report.render_offset(
            offset.audit_offset(table, column, offset=pedestal,
                                alpha=args.alpha)))
    elif column:
        print()
        print("No detector pedestal found or supplied, so drift percentages "
              "below use the raw level as the denominator and are understated. "
              "Pass --offset.")

    if column:
        try:
            print()
            print(report.render_mounts(mounts.decompose(table, column)))
        except ValueError as exc:
            print(f"\nBatch structure not resolved: {exc}")

    cond = None
    if args.raw_sum_col and column and args.n_voxels:
        cond = conditioning.assess_integral(table[args.raw_sum_col],
                                            table[column], args.n_voxels)
        print()
        print(report.render_conditioning(cond))

    sel = None
    if args.count_col and column:
        sel = selection.audit_threshold(table, count_col=args.count_col,
                                        level_col=column,
                                        threshold=args.threshold,
                                        ceiling=args.ceiling, alpha=args.alpha)
        print()
        print(report.render_selection(sel))

    sent = None
    if args.signal_channel:
        sent = sentinel.check_sentinel(table, channel=args.signal_channel,
                                       sentinel=args.sentinel, alpha=args.alpha)
        print()
        print(report.render_sentinel(sent))

        if args.reference_channel:
            print()
            print(report.RULE)
            print("CHANNEL CONTRAST  %s against %s"
                  % (args.signal_channel, args.reference_channel))
            print(report.RULE)
            print()
            contrast = drift.compare_channels(dr, args.signal_channel,
                                              args.reference_channel)
            notable = contrast[contrast["interpretation"] != "no drift in either"] \
                if not contrast.empty else contrast
            if notable.empty:
                print("No drift in either channel, in any group.")
            else:
                print(notable.to_string(index=False))

    print()
    verdict = fitness.assess(sched, dr, sent, selection_report=sel,
                             conditioning_report=cond)
    print(report.render_fitness(verdict))
    return 1 if verdict.unfit else 0


if __name__ == "__main__":
    sys.exit(main())
