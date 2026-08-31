"""Level 2 - drift in raw image statistics.

Tests every metric against acquisition time *within* each group. Because the
treatment is constant inside a group, any monotone trend there cannot be the
treatment effect: it is drift in the instrument, the mount or the specimen.

The control group is the sharpest instrument for this, since it is the group
where the experiment asserts that nothing is happening.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import as_percent_change, benjamini_hochberg, trend


@dataclass
class DriftReport:
    table: pd.DataFrame
    alpha: float
    session_minutes: float

    @property
    def significant(self) -> pd.DataFrame:
        return self.table[self.table["q"] < self.alpha]

    @property
    def verdict(self) -> str:
        if self.table.empty:
            return "UNKNOWN"
        if not self.significant.empty:
            return "DRIFT DETECTED"
        floor = self.table["min_detectable_rho"].min()
        if not np.isfinite(floor) or floor > 0.7:
            return "UNDERPOWERED"
        return "NO DRIFT DETECTED"


# Bookkeeping carried alongside the metrics, not metrics themselves.
BOOKKEEPING = ("offset", "n_voxels")

# How the detector offset enters each statistic. A location statistic carries
# it once; a sum carries it once per voxel; a spread and a fraction do not
# carry it at all.
OFFSET_PER_VOXEL = ("background", "p01", "p99", "p999")
OFFSET_SUMMED = ("total",)


def metric_columns(df, exclude=("t_min",)):
    """Numeric columns that look like image metrics rather than bookkeeping."""
    numeric = df.select_dtypes(include="number").columns
    return [c for c in numeric
            if c not in exclude
            and not c.startswith("set_")
            and not c.rsplit(".", 1)[-1] in BOOKKEEPING]


def signal_level(df, metric):
    """Median of `metric` with any detector offset removed.

    This is the denominator for every percentage the audit reports. Using the
    raw median instead would divide a real change by a number partly made of a
    constant the detector adds, shrinking every reported magnitude. Rank
    statistics are untouched by this: adding a constant changes no ordering.

    Returns (level, offset_applied).
    """
    raw = float(np.nanmedian(df[metric]))
    if "." not in metric:
        return raw, False
    channel, stat = metric.rsplit(".", 1)

    offset_col = f"{channel}.offset"
    if offset_col not in df.columns:
        return raw, False
    offset = float(np.nanmedian(df[offset_col]))
    if not np.isfinite(offset) or offset == 0:
        return raw, False

    if stat in OFFSET_PER_VOXEL:
        return raw - offset, True
    if stat in OFFSET_SUMMED:
        voxels_col = f"{channel}.n_voxels"
        if voxels_col in df.columns:
            voxels = float(np.nanmedian(df[voxels_col]))
            return raw - offset * voxels, True
    return raw, False


def analyse(df, metrics=None, group_col="group", time_col="t_min", alpha=0.05):
    """Within-group trend of every metric against acquisition time.

    Reported per group and per metric:
      rho, p              monotone association and its significance
      q                   Benjamini-Hochberg adjusted p across the whole family
      pct_over_span       Theil-Sen slope expressed as a percent of the
                          metric's median, across the window the group was
                          actually acquired in. Not extrapolated.
      pct_per_hour        the same slope per hour. Where a group spans much
                          less than an hour this is an extrapolation, and
                          `extrapolation` records by how much.
      ties_frac           fraction of repeated values. A zero slope alongside
                          a strong rho means the metric is heavily tied, so
                          the trend is real but its magnitude is not resolved
                          at this quantisation.
      min_detectable_rho  the smallest effect this group's n could have found
    """
    metrics = metrics or metric_columns(df)
    session = float(df[time_col].max() - df[time_col].min())

    rows = []
    for group, sub in df.groupby(group_col):
        for metric in metrics:
            if metric not in sub.columns:
                continue
            result = trend(sub[time_col], sub[metric], alpha=alpha)
            level, offset_applied = signal_level(sub, metric)
            rows.append({
                "group": group,
                "metric": metric,
                "n": result["n"],
                "span_min": result["span"],
                "level": level,
                "offset_applied": offset_applied,
                "rho": result["rho"],
                "p": result["p"],
                "pct_over_span": as_percent_change(result["slope"], level,
                                                   result["span"]),
                "pct_per_hour": as_percent_change(result["slope"], level, 60.0),
                "extrapolation": (60.0 / result["span"]
                                  if result["span"] and result["span"] > 0
                                  else np.nan),
                "ties_frac": result["ties_frac"],
                "slope_lo": result["slope_lo"],
                "slope_hi": result["slope_hi"],
                "min_detectable_rho": result["min_detectable_rho"],
            })

    table = pd.DataFrame(rows)
    if not table.empty:
        testable = table["p"].notna()
        table.loc[:, "q"] = np.nan
        table.loc[testable, "q"] = benjamini_hochberg(table.loc[testable, "p"])
        table = table.sort_values(["q", "p"], na_position="last").reset_index(drop=True)
    return DriftReport(table=table, alpha=alpha, session_minutes=session)


def compare_channels(report, signal, reference, group=None):
    """Contrast drift in a signal channel against a reference channel.

    A transmitted-light or otherwise treatment-independent channel recorded
    alongside the fluorescence gives a control that most quality checks lack.
    The two channels see the same specimen through the same mount at the same
    instant, so:

      signal drifts, reference flat  -> the fluorescence path: excitation,
                                        detection, or photophysics
      both drift together            -> the specimen or the mount
      reference drifts, signal flat  -> unusual; suspect focus or the stage

    This does not identify a cause on its own. It narrows the search, and it
    rules out the explanation that the whole preparation was changing.

    The contrast only means anything within a group, since across groups the
    treatment differs. With `group=None` every group is returned separately.
    """
    groups = [group] if group is not None else sorted(report.table["group"].unique())
    frames = []
    for name in groups:
        sub = report.table[report.table["group"] == name]
        frames.append(_contrast_one(sub, name, signal, reference, report.alpha))
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()
    return out


def _contrast_one(sub, group, signal, reference, alpha):
    def pick(prefix):
        hit = sub[sub["metric"].str.startswith(f"{prefix}.")].copy()
        hit["stat"] = hit["metric"].str.split(".").str[-1]
        return hit.drop_duplicates("stat").set_index("stat")

    sig, ref = pick(signal), pick(reference)
    shared = sig.index.intersection(ref.index)
    if shared.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "group": group,
        "metric": list(shared),
        f"{signal}_rho": sig.loc[shared, "rho"].to_numpy(),
        f"{signal}_q": sig.loc[shared, "q"].to_numpy(),
        f"{reference}_rho": ref.loc[shared, "rho"].to_numpy(),
        f"{reference}_q": ref.loc[shared, "q"].to_numpy(),
    })
    out["interpretation"] = [
        _interpret(a, b, alpha=alpha)
        for a, b in zip(out[f"{signal}_q"], out[f"{reference}_q"])
    ]
    return out.reset_index(drop=True)


def _interpret(signal_q, reference_q, alpha):
    sig = np.isfinite(signal_q) and signal_q < alpha
    ref = np.isfinite(reference_q) and reference_q < alpha
    if sig and not ref:
        return "fluorescence path"
    if sig and ref:
        return "specimen or mount"
    if ref and not sig:
        return "reference only - check focus or stage"
    return "no drift in either"
