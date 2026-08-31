"""Recovering the batch structure of a session, and using it to locate the drift.

Nobody acquires a long session in one unbroken run. Slides get remounted,
animals get anaesthetised in batches, dishes get swapped. Those interruptions
leave a signature in the timestamps: a run of images a few minutes apart, then
a gap, then another run. The batches can be recovered from the file times
alone, without anyone having written them down.

That recovery is worth doing because it splits drift into two kinds that
demand different responses:

  between batches   the level at the start of each batch marches in one
                    direction across the whole session. A clock-driven
                    process: detector warm-up, laser ageing, room temperature.

  within batches    the level moves as time passes inside a batch and resets
                    when the next one is mounted. A preparation-driven
                    process: anaesthetic exposure, evaporation, photobleaching
                    of the mounted field.

The distinction decides what to do next. A clock-driven drift is defeated by
randomising acquisition order. A batch-driven one is not: it recurs identically
in every batch, so interleaving spreads it evenly over the groups instead of
removing it, and the fix is to shorten the batch or to standardise the delay
between mounting and imaging.

One asymmetry is worth stating plainly, because the arithmetic invites the
wrong conclusion. Batches are short and the session is long, so the
within-batch test always has less power than the between-batch one. A flat
within-batch result is therefore weak evidence on its own, and the confidence
interval is reported next to it so that it cannot be read as a clean negative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MountReport:
    column: str
    n_mounts: int
    sizes: list
    gap_threshold_min: float
    median_interval_min: float
    mean_mount_minutes: float

    between_rho: float
    between_p: float
    between_slope: float

    within_rho: float
    within_p: float
    within_slope: float
    within_slope_lo: float
    within_slope_hi: float
    within_f: float
    within_f_p: float
    r2_time: float
    r2_time_plus_within: float

    @property
    def locus(self) -> str:
        """Where the drift lives, as far as this decomposition can tell."""
        between = np.isfinite(self.between_p) and self.between_p < 0.05
        within = np.isfinite(self.within_f_p) and self.within_f_p < 0.05
        if between and within:
            return "BOTH"
        if between:
            return "BETWEEN MOUNTS"
        if within:
            return "WITHIN MOUNTS"
        return "NEITHER RESOLVED"

    @property
    def remedy(self) -> str:
        return {
            "BETWEEN MOUNTS":
                "Clock-driven. Randomising and interleaving the acquisition "
                "order makes this drift harmless without removing it.",
            "WITHIN MOUNTS":
                "Batch-driven and it resets with each mount, so interleaving "
                "will not help. Shorten the batch, or fix the delay between "
                "mounting and imaging so it is identical for every image.",
            "BOTH":
                "Randomise the order to defeat the clock-driven part, and fix "
                "the mounting-to-imaging delay to defeat the rest.",
            "NEITHER RESOLVED":
                "The decomposition did not localise the drift. Check the "
                "confidence interval on the within-mount term before reading "
                "this as an absence.",
        }[self.locus]


def find_mounts(times, min_gap=None, gap_multiple=3.0, mad_multiple=5.0):
    """Split a session into batches at the unusually long gaps between images.

    With `min_gap=None` the threshold is derived from the session itself, as
    the larger of `gap_multiple` times the median interval and the median plus
    `mad_multiple` robust deviations. Two rules rather than one because either
    alone fails on a real session: a regular session has a near-zero MAD and
    the second rule fires on nothing, while a ragged one has a large median and
    the first rule fires on nothing.

    Returns (mount_index, gaps, threshold). `mount_index` is a 0-based label
    per image in the order given, so sort by time first.
    """
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        return np.array([], dtype=int), np.array([]), np.nan
    if np.any(np.diff(t) < 0):
        raise ValueError("times must be sorted ascending")

    intervals = np.diff(t)
    if intervals.size == 0:
        return np.zeros(1, dtype=int), intervals, np.nan

    median = float(np.median(intervals))
    mad = float(np.median(np.abs(intervals - median))) * 1.4826

    if min_gap is None:
        threshold = max(gap_multiple * median, median + mad_multiple * mad)
    else:
        threshold = float(min_gap)

    breaks = intervals > threshold
    index = np.concatenate([[0], np.cumsum(breaks)]).astype(int)
    return index, intervals[breaks], float(threshold)


def _ols(X, y):
    """Least squares with the residual sum of squares and the R^2."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    return beta, rss, (1.0 - rss / tss if tss > 0 else np.nan)


def decompose(df, column, time_col="t_min", mount_col=None, alpha=0.05,
              **mount_kwargs) -> MountReport:
    """Split the drift of one metric into between-mount and within-mount parts.

    If `mount_col` is absent the batches are recovered from the timestamps by
    `find_mounts`. Supply the column instead whenever the real batch structure
    was recorded: an inferred partition is a convenience, not a substitute.
    """
    d = df.dropna(subset=[time_col, column]).sort_values(time_col).copy()
    if len(d) < 6:
        raise ValueError(f"need at least 6 images, got {len(d)}")

    if mount_col and mount_col in d.columns:
        codes = pd.factorize(d[mount_col])[0]
        gaps, threshold = np.array([]), np.nan
        intervals = np.diff(d[time_col].to_numpy())
    else:
        codes, gaps, threshold = find_mounts(d[time_col].to_numpy(),
                                             **mount_kwargs)
        intervals = np.diff(d[time_col].to_numpy())
    d["_mount"] = codes

    # Between mounts: does the mount-level average march across the session?
    per_mount = d.groupby("_mount").agg(t=(time_col, "mean"),
                                        level=(column, "mean"),
                                        n=(column, "size"))
    if len(per_mount) >= 3:
        b_rho, b_p = stats.spearmanr(per_mount["t"], per_mount["level"])
        b_slope = float(np.polyfit(per_mount["t"], per_mount["level"], 1)[0])
    else:
        b_rho, b_p, b_slope = np.nan, np.nan, np.nan

    # Within mounts: does elapsed time inside a mount add anything to a model
    # that already knows the time of day?
    d["_within"] = d[time_col] - d.groupby("_mount")[time_col].transform("min")
    t = d[time_col].to_numpy(dtype=float)
    w = d["_within"].to_numpy(dtype=float)
    y = d[column].to_numpy(dtype=float)

    ones = np.ones_like(t)
    _, rss_a, r2_a = _ols(np.column_stack([ones, t]), y)
    beta_b, rss_b, r2_b = _ols(np.column_stack([ones, t, w]), y)

    n, p_a, p_b = len(y), 2, 3
    if n > p_b and rss_b > 0 and np.ptp(w) > 0:
        f = ((rss_a - rss_b) / (p_b - p_a)) / (rss_b / (n - p_b))
        f_p = float(stats.f.sf(f, p_b - p_a, n - p_b))
        sigma2 = rss_b / (n - p_b)
        X = np.column_stack([ones, t, w])
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(cov[2, 2]))
        crit = stats.t.ppf(1 - alpha / 2, n - p_b)
        lo, hi = beta_b[2] - crit * se, beta_b[2] + crit * se
        w_slope = float(beta_b[2])
    else:
        f, f_p, lo, hi, w_slope = np.nan, np.nan, np.nan, np.nan, np.nan

    resid = y - np.column_stack([ones, t]) @ np.linalg.lstsq(
        np.column_stack([ones, t]), y, rcond=None)[0]
    if np.ptp(w) > 0 and np.ptp(resid) > 0:
        w_rho, w_p = stats.spearmanr(w, resid)
    else:
        w_rho, w_p = np.nan, np.nan

    return MountReport(
        column=column,
        n_mounts=int(per_mount.shape[0]),
        sizes=[int(v) for v in per_mount["n"]],
        gap_threshold_min=threshold,
        median_interval_min=float(np.median(intervals)) if intervals.size else np.nan,
        mean_mount_minutes=float(d.groupby("_mount")[time_col]
                                 .agg(lambda s: s.max() - s.min()).mean()),
        between_rho=float(b_rho), between_p=float(b_p), between_slope=b_slope,
        within_rho=float(w_rho), within_p=float(w_p),
        within_slope=w_slope, within_slope_lo=float(lo), within_slope_hi=float(hi),
        within_f=float(f), within_f_p=f_p,
        r2_time=float(r2_a), r2_time_plus_within=float(r2_b))
