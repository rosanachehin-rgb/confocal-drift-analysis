"""Level 1 - acquisition schedule audit.

Reads acquisition timestamps, group labels and instrument settings. Never
opens pixel data, so it runs in under a second on a full session and can be
used the moment acquisition finishes, before any analysis decision is made.

This is the cheapest check in the package and usually the most consequential.
A design in which each group occupies its own contiguous block of the session
confounds group with time so completely that no downstream statistics can
undo it. That is detectable from filenames and timestamps alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .stats import eta_squared, eta_squared_permutation_p


@dataclass
class ScheduleReport:
    n_images: int
    n_groups: int
    session_minutes: float
    eta_squared: float
    eta_p: float
    max_run: int
    max_run_expected: float
    blocks: pd.DataFrame
    overlap: pd.DataFrame
    settings_varying: dict = field(default_factory=dict)
    settings_constant: dict = field(default_factory=dict)

    @property
    def separable(self) -> bool:
        """Whether group and acquisition time can be told apart at all."""
        return bool(np.isfinite(self.eta_squared) and self.eta_squared < 0.60)

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.eta_squared):
            return "UNKNOWN"
        if self.eta_squared >= 0.90:
            return "CONFOUNDED"
        if self.eta_squared >= 0.60:
            return "AT RISK"
        return "OK"


def _pairwise_overlap(blocks: pd.DataFrame) -> pd.DataFrame:
    """Temporal overlap between every pair of group acquisition windows.

    Expressed as a fraction of the shorter of the two windows, so a small
    group nested inside a long one still scores 1.0. Zero everywhere means a
    strictly blocked design.
    """
    rows = []
    names = list(blocks.index)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lo = max(blocks.loc[a, "start"], blocks.loc[b, "start"])
            hi = min(blocks.loc[a, "end"], blocks.loc[b, "end"])
            shared = max(0.0, hi - lo)
            shorter = min(blocks.loc[a, "end"] - blocks.loc[a, "start"],
                          blocks.loc[b, "end"] - blocks.loc[b, "start"])
            frac = shared / shorter if shorter > 0 else np.nan
            rows.append({"group_a": a, "group_b": b,
                         "overlap_min": shared, "overlap_frac": frac})
    return pd.DataFrame(rows)


def _max_run(labels) -> int:
    """Longest streak of consecutive images from the same group."""
    best = run = 0
    previous = object()
    for lab in labels:
        run = run + 1 if lab == previous else 1
        previous = lab
        best = max(best, run)
    return best


def _expected_max_run(counts, n_perm=2000, seed=0) -> float:
    """Mean longest streak under random ordering, for comparison."""
    rng = np.random.default_rng(seed)
    labels = np.concatenate([[g] * int(n) for g, n in counts.items()])
    runs = []
    for _ in range(n_perm):
        rng.shuffle(labels)
        runs.append(_max_run(labels))
    return float(np.mean(runs))


def audit_schedule(df, group_col="group", time_col="t_min",
                   settings_cols=None, n_perm=10000, seed=0) -> ScheduleReport:
    """Audit an acquisition schedule.

    Parameters
    ----------
    df : DataFrame with one row per image, a group label and a time in minutes
         from the start of the session.
    settings_cols : columns holding instrument settings to check for constancy.
    """
    d = df.dropna(subset=[time_col]).sort_values(time_col)
    if d.empty:
        raise ValueError("no rows with a usable acquisition time")

    eta_p, eta = eta_squared_permutation_p(d[time_col].to_numpy(),
                                           d[group_col].to_numpy(),
                                           n_perm=n_perm, seed=seed)

    blocks = (d.groupby(group_col)[time_col]
                .agg(n="count", start="min", end="max")
                .sort_values("start"))
    blocks["duration_min"] = blocks["end"] - blocks["start"]

    counts = blocks["n"].to_dict()
    report = ScheduleReport(
        n_images=len(d),
        n_groups=d[group_col].nunique(),
        session_minutes=float(d[time_col].max() - d[time_col].min()),
        eta_squared=eta,
        eta_p=eta_p,
        max_run=_max_run(d[group_col].tolist()),
        max_run_expected=_expected_max_run(counts, seed=seed),
        blocks=blocks,
        overlap=_pairwise_overlap(blocks),
    )

    for col in (settings_cols or []):
        if col not in d.columns:
            continue
        values = d[col].dropna().unique()
        if len(values) > 1:
            report.settings_varying[col] = sorted(map(str, values))
        elif len(values) == 1:
            report.settings_constant[col] = str(values[0])

    return report


def interleaved_order(counts, seed=0):
    """Propose a randomised, interleaved acquisition order.

    Draws without replacement from the group still furthest from finishing,
    with ties broken at random, so no group ends up concentrated in one part
    of the session. Run this before the microscope is switched on; it is the
    only part of the package that prevents the problem rather than detecting
    it after the fact.
    """
    rng = np.random.default_rng(seed)
    remaining = {g: int(n) for g, n in counts.items() if int(n) > 0}
    total = sum(remaining.values())
    done = {g: 0 for g in remaining}
    order = []
    for step in range(total):
        target = (step + 1) / total
        deficits = {g: target * remaining_total_share(counts, g) - done[g]
                    for g in remaining}
        best = max(deficits.values())
        candidates = [g for g, v in deficits.items() if v >= best - 1e-9]
        pick = candidates[int(rng.integers(len(candidates)))]
        order.append(pick)
        done[pick] += 1
        remaining[pick] -= 1
        if remaining[pick] == 0:
            del remaining[pick]
    return order


def remaining_total_share(counts, group):
    total = sum(int(n) for n in counts.values())
    return int(counts[group]) / total if total else 0.0
