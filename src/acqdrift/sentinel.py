"""Does the sentinel metric actually measure what the audit assumes?

Level 2 uses the median voxel value as a background sentinel. That only works
where the labelling is sparse enough that the median sits in the background
population, so the biology under test cannot move it. In a densely labelled
preparation the median is signal, it responds to the treatment, and every
conclusion drawn from it is circular.

The package does not assume this holds. It tests it, and refuses to report a
Level 2 verdict when the test fails.

Two statistics are computed, and only the second is a test:

`occupancy` is descriptive. It estimates the fraction of the field above
background, and is reliable only while that fraction stays below about a half.
Past that the median crosses into the signal population and the estimate
collapses to zero, which is indistinguishable from an empty field. It is
reported for orientation and is never used to pass or fail a dataset.

`independence` is the test. Across images within a group it asks whether the
sentinel moves with the amount of bright material in the frame, after removing
any common trend with acquisition time. If it does, the sentinel is not
independent of the content and cannot serve as a control.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .stats import benjamini_hochberg


def robust_sigma(values):
    """Scale estimate that survives a field full of bright objects."""
    v = np.asarray(values, dtype=float)
    mad = np.median(np.abs(v - np.median(v)))
    if mad > 0:
        return float(mad * 1.4826)
    q75, q25 = np.percentile(v, [75, 25])
    return float((q75 - q25) / 1.349) if q75 > q25 else np.nan


def occupancy(values, k=5.0):
    """Fraction of voxels more than k robust sigmas above the median.

    Reliable below roughly 0.5. Above that the median has moved into the signal
    population and this returns near zero, so a zero reading is ambiguous and
    must be read together with `independence`.
    """
    v = np.asarray(values, dtype=float).ravel()
    med, sigma = np.median(v), robust_sigma(v)
    if not np.isfinite(sigma) or sigma == 0:
        return np.nan
    return float(np.mean(v > med + k * sigma))


def partial_spearman(x, y, z):
    """Spearman correlation of x with y, with z removed from both.

    Rank-transform all three, regress the ranks of x and y on the ranks of z,
    and correlate the residuals. Removing acquisition time matters here: both
    the sentinel and the signal proxy may drift, and a raw correlation between
    them would be that shared drift rather than any dependence of one on the
    other.

    Returns (rho, p, reason). `reason` is None when the test ran, and otherwise
    names why it could not, so that an untestable case is never mistaken for a
    passing one.
    """
    x, y, z = (np.asarray(a, dtype=float) for a in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    n = x.size

    if n < 6:
        return np.nan, np.nan, f"n = {n}, fewer than 6 images"
    if np.ptp(z) == 0:
        return np.nan, np.nan, "all images share one acquisition time"
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, "sentinel or signal proxy is constant"

    rx, ry, rz = (stats.rankdata(a) for a in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    # A sentinel that is a perfect monotone function of time leaves no residual
    # variation to test against. That is a real limit, not a clean result.
    scale = max(np.ptp(rx), 1.0)
    if np.ptp(ex) < 1e-6 * scale:
        return np.nan, np.nan, "sentinel is perfectly ordered by time"
    if np.ptp(ey) < 1e-6 * max(np.ptp(ry), 1.0):
        return np.nan, np.nan, "signal proxy is perfectly ordered by time"

    r = float(np.corrcoef(ex, ey)[0, 1])
    t = r * np.sqrt((n - 3) / max(1e-12, 1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), n - 3)), None


@dataclass
class SentinelCheck:
    channel: str
    table: pd.DataFrame
    alpha: float
    notes: list = field(default_factory=list)

    @property
    def failing(self) -> pd.DataFrame:
        """Groups where the sentinel tracks the amount of signal."""
        t = self.table
        return t[(t["q"] < self.alpha) & (t["rho_partial"] > 0)]

    @property
    def valid(self) -> bool:
        return self.failing.empty

    @property
    def tested(self):
        return self.table[self.table["p"].notna()] if not self.table.empty \
            else self.table

    @property
    def verdict(self) -> str:
        if self.table.empty or self.tested.empty:
            return "NOT TESTABLE"
        if not self.valid:
            return "SENTINEL INVALID"
        if len(self.tested) < len(self.table):
            return "SENTINEL VALID WHERE TESTED"
        return "SENTINEL VALID"


def check_sentinel(df, channel="ZsGr1", sentinel="background",
                   signal_proxy="p999", group_col="group", time_col="t_min",
                   alpha=0.05, min_separation=1.5) -> SentinelCheck:
    """Test the sparse-labelling assumption on a real session, group by group.

    `signal_proxy` must be a percentile high enough to land in the signal
    population. Where labelling is very sparse a p99 still sits in the
    background, and correlating the sentinel against it would be correlating
    background with background. The separation between the two is checked
    before the test runs, and the test is declared untestable if it fails.
    """
    col_sentinel = f"{channel}.{sentinel}"
    col_proxy = f"{channel}.{signal_proxy}"
    notes = []
    for col in (col_sentinel, col_proxy):
        if col not in df.columns:
            notes.append(f"missing column {col}")
    if notes:
        return SentinelCheck(channel, pd.DataFrame(), alpha, notes)

    rows = []
    for group, sub in df.groupby(group_col):
        level = float(np.nanmedian(sub[col_sentinel]))
        proxy_level = float(np.nanmedian(sub[col_proxy]))
        separation = proxy_level / level if level else np.nan

        if not np.isfinite(separation) or separation < min_separation:
            rho, p, reason = (np.nan, np.nan,
                              f"{signal_proxy} is only {separation:.2f}x the "
                              f"sentinel, so it is not a signal proxy here")
        else:
            rho, p, reason = partial_spearman(sub[col_sentinel], sub[col_proxy],
                                              sub[time_col])

        span = np.ptp(sub[col_sentinel].to_numpy())
        rows.append({
            "group": group, "n": len(sub), "rho_partial": rho, "p": p,
            "separation": separation,
            "sentinel_range_pct": 100 * span / level if level else np.nan,
            "untestable_because": reason,
        })

    table = pd.DataFrame(rows)
    testable = table["p"].notna()
    table["q"] = np.nan
    if testable.any():
        table.loc[testable, "q"] = benjamini_hochberg(table.loc[testable, "p"])
    for _, row in table[~testable].iterrows():
        notes.append(f"{row['group']}: not tested, {row['untestable_because']}")

    return SentinelCheck(channel, table.sort_values("q", na_position="last")
                                     .reset_index(drop=True), alpha, notes)
