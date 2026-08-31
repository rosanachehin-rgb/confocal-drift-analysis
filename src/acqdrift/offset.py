"""The detector offset, and why a drift percentage is wrong without it.

Every photon-counting or analogue detector adds a constant pedestal to the
digitised value so that noise below zero is not clipped. Zeiss calls it the
digital offset, Leica and Nikon call it something else, and on many systems it
is simply left at whatever the last user set. Whatever it is called, it is a
number of counts that was never light.

The pedestal is harmless for anything computed as a difference, and it is
poison for anything computed as a ratio. A drift expressed as a percentage is
a ratio, so it is exactly the case that goes wrong:

    background falls from 151 to 142 counts over a session
    with the pedestal in the denominator   ->   5.8 % drift
    with the pedestal removed              ->  20.5 % drift

Same photons, same session, two numbers that differ by a factor of three and
a half. The first one gets a session waved through; the second one does not.

This module is small on purpose. It does one arithmetic correction, and it
reports how much that correction changed the answer, because the size of that
change is itself the argument for having done it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import trend


@dataclass
class OffsetReport:
    column: str
    offset: float
    raw_level: float
    net_level: float
    change_counts: float
    pct_raw: float
    pct_net: float
    rho: float
    p: float
    span_min: float
    source: str

    @property
    def inflation(self) -> float:
        """How many times larger the honest percentage is than the naive one."""
        if not np.isfinite(self.pct_raw) or self.pct_raw == 0:
            return np.nan
        return float(self.pct_net / self.pct_raw)

    @property
    def matters(self) -> bool:
        """Whether ignoring the pedestal would have changed the reading.

        Threshold at 1.5x rather than at some significance test: this is not a
        hypothesis, it is an arithmetic fact about the denominator, and the
        only question is whether it is large enough to bother reporting.
        """
        return bool(np.isfinite(self.inflation) and self.inflation >= 1.5)


def net(values, offset):
    """Subtract the detector pedestal, refusing to return negative levels.

    A level below the pedestal means the pedestal is wrong: either it was read
    from the wrong metadata field, or it was changed mid-session. Silently
    clamping to zero would hide that, and a zero denominator downstream would
    produce an infinite percentage that looks like a catastrophic finding.
    """
    v = np.asarray(values, dtype=float) - float(offset)
    if np.nanmin(v) < 0:
        raise ValueError(
            f"values fall below the stated offset of {offset}; the offset is "
            f"wrong or it changed during the session")
    return v


def constant_offset(df, column="set_Offset", tolerance=0.0):
    """Read a single offset from per-image metadata, or refuse.

    Returns (value, source). A session in which the pedestal moved is not one
    session for this purpose, and the caller is told so rather than handed an
    average that describes none of the images.
    """
    if column not in df.columns:
        return None, f"no column {column}"
    values = pd.to_numeric(df[column], errors="coerce").dropna().unique()
    if values.size == 0:
        return None, f"{column} is empty"
    if values.size > 1 and (values.max() - values.min()) > tolerance:
        return None, (f"{column} is not constant: "
                      f"{', '.join(str(v) for v in sorted(values))}")
    return float(values[0]), f"metadata column {column}"


def audit_offset(df, column, offset=None, offset_column="set_Offset",
                 time_col="t_min", alpha=0.05) -> OffsetReport:
    """Express the drift of one metric with and without the pedestal removed.

    `offset` overrides the metadata. Pass it when the reader could not find the
    field, or when the pedestal is known from a dark-frame measurement, which
    is the more trustworthy source of the two.
    """
    if offset is None:
        offset, source = constant_offset(df, offset_column)
        if offset is None:
            raise ValueError(
                f"could not establish a detector offset: {source}. Pass "
                f"offset= explicitly, or measure it from a dark frame.")
    else:
        source = "supplied by caller"

    values = df[column].to_numpy(dtype=float)
    result = trend(df[time_col], values, alpha=alpha)

    raw_level = float(np.nanmedian(values))
    net_level = raw_level - float(offset)
    span = result["span"]
    change = (result["slope"] * span
              if np.isfinite(result["slope"]) and np.isfinite(span) else np.nan)

    def pct(denominator):
        if not np.isfinite(change) or not np.isfinite(denominator) or denominator == 0:
            return np.nan
        return float(100.0 * change / denominator)

    return OffsetReport(
        column=column, offset=float(offset),
        raw_level=raw_level, net_level=net_level,
        change_counts=float(change) if np.isfinite(change) else np.nan,
        pct_raw=pct(raw_level), pct_net=pct(net_level),
        rho=result["rho"], p=result["p"], span_min=span, source=source)


def add_net_columns(df, columns, offset, suffix="_net"):
    """Return a copy with pedestal-free versions of the named columns.

    Downstream code should prefer these. Nothing in the package rewrites the
    original columns: the raw numbers are what the instrument reported, and a
    table that quietly no longer matches the files is worse than one that is
    merely incomplete.
    """
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[f"{column}{suffix}"] = net(out[column], offset)
    return out
