"""The vendor-neutral acquisition record, and how to get one from any microscope.

Nothing in the audits needs to know what wrote the files. They need a table
with one row per image and a small number of named fields. Everything that is
specific to a manufacturer exists to fill that table in, and lives in a reader.

    reader  ->  acquisition record  ->  audits

The package ships one reader, for Zeiss CZI, because that is the format it was
developed and tested against. Claiming more would be dishonest: a reader that
has never been run against a real file from that instrument is a guess, and a
guess about where a timestamp lives in a proprietary container is a bad one.

There are two supported ways to audit data from any other microscope.

The first is to build the record yourself and hand it over as a CSV. Every
column below can be filled from an export, a lab notebook, or a few lines of
whatever library already reads your format. `from_csv` validates it and tells
you which audits it can support. This route needs no code from us and it works
today for Leica, Nikon, Olympus, a light sheet, or a spinning disk.

The second is to write a reader. It is one function returning one dictionary
per file, keyed as below, and it is the whole of what a new manufacturer
requires. `io.czi_metadata` is the reference implementation; anything the
reader cannot find should be left absent rather than guessed, since the
validation below is built to degrade gracefully around missing fields and
cannot protect anyone from a fabricated one.

A note on the field that is hardest to obtain and most important to have.
`timestamp` must be the moment the instrument acquired the image. File
modification time is not that: it survives a copy, a sync, or an export, and
it silently reorders a session. Where only file times exist, say so with
`time_source="filesystem"` and read every schedule verdict as provisional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: Fields of one acquisition record. Name, whether it is required, and what it
#: is for. A reader fills what it can and omits the rest.
SCHEMA = {
    "file":       ("required", "identifier of the image, unique in the session"),
    "group":      ("required", "experimental condition this image belongs to"),
    "timestamp":  ("required*", "acquisition time, ISO 8601. Required unless "
                                "t_min is supplied directly"),
    "t_min":      ("derived", "minutes from the first image of the session"),
    "px_um":      ("optional", "lateral pixel size in micrometres"),
    "z_um":       ("optional", "axial step in micrometres"),
    "offset":     ("recommended", "detector pedestal in counts. Without it, "
                                  "every drift percentage is understated"),
    "n_voxels":   ("optional", "voxels per image, for the conditioning check"),
    "mount":      ("optional", "batch or mount identifier. Inferred from gaps "
                               "in the timestamps when absent"),
}

#: Any column named `set_*` is treated as an instrument setting and checked for
#: constancy across the session. The names are the manufacturer's own; the
#: audit only asks whether they changed, so no cross-vendor vocabulary is
#: needed for this to work.
SETTING_PREFIX = "set_"

#: Per-image image statistics are named `<channel>.<statistic>`. The channel
#: name is whatever the acquisition called it. `background` is the sentinel the
#: drift audit uses by default.
METRIC_SEPARATOR = "."

REQUIRED = ("file", "group")


@dataclass
class RecordCheck:
    n_rows: int
    columns: list
    missing_required: list = field(default_factory=list)
    time_source: str = "unknown"
    channels: list = field(default_factory=list)
    settings: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def can_audit_schedule(self) -> bool:
        return not self.missing_required and self.time_source != "none"

    @property
    def can_audit_drift(self) -> bool:
        return self.can_audit_schedule and bool(self.channels)

    @property
    def can_correct_offset(self) -> bool:
        return "offset" in self.columns or f"{SETTING_PREFIX}Offset" in self.columns

    def summary(self) -> str:
        lines = [f"{self.n_rows} images, acquisition time from {self.time_source}"]
        if self.missing_required:
            lines.append("missing required: " + ", ".join(self.missing_required))
        lines.append("schedule audit: " + ("yes" if self.can_audit_schedule else "no"))
        lines.append("drift audit: " + ("yes" if self.can_audit_drift
                                        else "no, no per-image metrics found"))
        lines.append("pedestal-corrected percentages: "
                     + ("yes" if self.can_correct_offset
                        else "no, drift will be understated"))
        if self.channels:
            lines.append("channels: " + ", ".join(self.channels))
        lines.extend(f"warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def channels_in(df):
    """Channel names implied by the `<channel>.<statistic>` metric columns."""
    names = []
    for col in df.columns:
        if METRIC_SEPARATOR in col and not col.startswith(SETTING_PREFIX):
            head = col.split(METRIC_SEPARATOR)[0]
            if head and head not in names:
                names.append(head)
    return names


def validate(df, time_source="unknown") -> RecordCheck:
    """Report which audits this table can support, and what it is missing.

    Never raises for a merely incomplete table. A session with timestamps and
    group labels and nothing else still supports the schedule audit, which is
    the cheapest check in the package and usually the one that decides the
    outcome.
    """
    columns = list(df.columns)
    missing = [c for c in REQUIRED if c not in columns]
    warnings = []

    if "t_min" in columns:
        source = time_source if time_source != "unknown" else "t_min column"
    elif "timestamp" in columns:
        source = time_source if time_source != "unknown" else "timestamp column"
    else:
        source = "none"
        warnings.append("no timestamp and no t_min; the schedule audit cannot run")

    if source == "filesystem":
        warnings.append(
            "acquisition times came from the filesystem, which a copy or a "
            "sync can rewrite; treat the schedule verdict as provisional")

    if "group" in columns and df["group"].nunique() < 2:
        warnings.append("fewer than two groups; there is nothing to confound")

    if "offset" not in columns and f"{SETTING_PREFIX}Offset" not in columns:
        warnings.append(
            "no detector pedestal; drift percentages will use the raw level as "
            "the denominator and will therefore be understated")

    settings = [c for c in columns if c.startswith(SETTING_PREFIX)]
    if not settings:
        warnings.append(
            "no set_* columns, so the audit cannot verify that acquisition "
            "settings stayed constant during the session")

    return RecordCheck(n_rows=len(df), columns=columns, missing_required=missing,
                       time_source=source, channels=channels_in(df),
                       settings=settings, warnings=warnings)


def finalise(df, timestamp_col="timestamp", time_col="t_min"):
    """Add `t_min` from the timestamps and sort the session into acquisition order.

    Times are anchored on the earliest image rather than on midnight, so a
    session that runs past midnight does not fold back on itself.
    """
    out = df.copy()
    if time_col not in out.columns:
        if timestamp_col not in out.columns:
            raise ValueError(
                f"need either {time_col} or {timestamp_col} to place the "
                f"images in acquisition order")
        stamps = pd.to_datetime(out[timestamp_col], format="ISO8601",
                                utc=True, errors="coerce")
        if stamps.isna().all():
            stamps = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
        if stamps.isna().all():
            raise ValueError(f"could not parse any value in {timestamp_col}")
        out[time_col] = (stamps - stamps.min()).dt.total_seconds() / 60.0
    return out.sort_values(time_col).reset_index(drop=True)


def from_csv(path, timestamp_col="timestamp", time_col="t_min",
             time_source="csv"):
    """Read an acquisition record someone else produced, and validate it.

    The route for any microscope this package has no reader for. Returns
    (table, check); print `check.summary()` before trusting a verdict.
    """
    df = pd.read_csv(Path(path))
    table = finalise(df, timestamp_col=timestamp_col, time_col=time_col)
    return table, validate(table, time_source=time_source)


def from_records(records, time_source="reader"):
    """Build the table from a sequence of per-file dictionaries.

    The interface a new reader targets. Each dictionary uses the keys in
    `SCHEMA`, plus any `set_*` settings and any `<channel>.<statistic>`
    metrics it was able to compute.
    """
    df = pd.DataFrame(list(records))
    if df.empty:
        raise ValueError("no records")
    table = finalise(df)
    return table, validate(table, time_source=time_source)


def describe_schema() -> str:
    """The record layout, for the documentation and the command line."""
    width = max(len(k) for k in SCHEMA)
    lines = [f"{'field'.ljust(width)}  {'status'.ljust(11)}  meaning",
             f"{'-' * width}  {'-' * 11}  {'-' * 40}"]
    for name, (status, meaning) in SCHEMA.items():
        lines.append(f"{name.ljust(width)}  {status.ljust(11)}  {meaning}")
    lines.append("")
    lines.append(f"{SETTING_PREFIX}*  instrument settings, checked for constancy")
    lines.append("<channel>.<statistic>  per-image metrics, e.g. GFP.background")
    return "\n".join(lines)


def missing_reader_message(suffix) -> str:
    """What to tell someone holding a format the package cannot open."""
    return (
        f"no reader for {suffix} files. acqdrift ships a reader only for the "
        f"format it has been tested against (.czi). Build the acquisition "
        f"record yourself and load it with `acqdrift.records.from_csv`, or "
        f"write a reader returning the fields in `acqdrift.records.SCHEMA`. "
        f"Run `acqdrift --schema` to print the layout.")
