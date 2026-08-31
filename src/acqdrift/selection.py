"""When a fixed detection threshold turns drift into a change of sample.

A detector applied at a fixed number of counts asks the same question of every
image. If the images are not equally bright, that is not fairness, it is a
moving admission standard. Objects near the threshold enter the sample in the
bright images and fall out of it in the dim ones, so the set of objects that
gets measured changes with the state of the instrument.

The damage is worse than a bias in the measured value, and it is a different
kind of damage. A gain change scales brightness and can, in principle, be
divided back out. A change in which objects were admitted cannot: the ones
that fell below the line left no record, and nothing downstream can recover
them. A correction applied afterwards will faithfully rescale the survivors
and leave the selection untouched.

The signature is visible without ever opening an image, from the per-image
object counts alone:

  counts track the brightness of the image within a group, where the
  treatment is constant and the true count cannot be changing;

  counts exceed whatever number the preparation can actually contain;

  a per-object property drifts across a group that received no treatment.

Each is checked separately below, because they fail independently and a
session can show one without the others.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .stats import benjamini_hochberg


@dataclass
class SelectionReport:
    count_col: str
    level_col: str
    table: pd.DataFrame
    alpha: float
    ceiling: float
    over_ceiling: int
    n_images: int
    headroom_sigma: float
    headroom_shift_sigma: float
    notes: list = field(default_factory=list)

    @property
    def coupled(self) -> pd.DataFrame:
        """Groups where the object count tracks how bright the image was."""
        t = self.table
        if t.empty:
            return t
        return t[(t["q"] < self.alpha) & (t["rho"] > 0)]

    @property
    def over_ceiling_frac(self) -> float:
        if not self.n_images:
            return np.nan
        return float(self.over_ceiling / self.n_images)

    @property
    def verdict(self) -> str:
        if not self.coupled.empty:
            return "THRESHOLD SELECTING"
        if np.isfinite(self.over_ceiling_frac) and self.over_ceiling_frac > 0.05:
            return "THRESHOLD SELECTING"
        if self.table.empty or self.table["p"].isna().all():
            return "NOT TESTABLE"
        if np.isfinite(self.headroom_shift_sigma) and self.headroom_shift_sigma >= 1.0:
            return "AT RISK"
        return "NO EVIDENCE OF SELECTION"

    @property
    def remedy(self) -> str:
        if self.verdict in ("NO EVIDENCE OF SELECTION", "NOT TESTABLE"):
            return ""
        return ("Set the detection threshold relative to each image - a "
                "quantile of that image, or a fixed multiple of its own "
                "background scatter - so that admission does not depend on the "
                "state of the instrument. Rescaling the images afterwards does "
                "not undo a selection that already happened.")


def headroom(background, threshold, noise=None):
    """Distance from the background to the detection threshold, in noise units.

    Small headroom means the threshold sits inside the shoulder of the
    background distribution, where a change of a few counts moves a large
    number of objects across it. `noise` defaults to the image-to-image
    standard deviation of the background, which is the scatter the threshold
    actually has to survive across a session.
    """
    back = np.asarray(background, dtype=float)
    ok = np.isfinite(back)
    if ok.sum() < 2:
        return np.nan
    sigma = float(np.std(back[ok], ddof=1)) if noise is None else float(noise)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.inf
    return float((float(threshold) - float(np.median(back[ok]))) / sigma)


def audit_threshold(df, count_col, level_col, threshold=None,
                    group_col="group", time_col="t_min", ceiling=None,
                    alpha=0.05) -> SelectionReport:
    """Test whether a fixed detection threshold is selecting on image brightness.

    Parameters
    ----------
    count_col : per-image number of detected objects.
    level_col : per-image brightness the threshold competes against, usually
        the background sentinel. Pass the pedestal-free column where one
        exists; the correlation is unaffected but the headroom is not.
    threshold : the absolute threshold that was applied, in the units of
        `level_col`. Optional; without it the headroom is not computed and the
        two statistical checks still run.
    ceiling : the largest number of objects the preparation can contain. Six
        for the dopaminergic head neurons of C. elegans. Counts above a stated
        ceiling are false positives by construction, and their rate measures
        how loose the threshold is without needing any ground truth.
    """
    notes = []
    for col in (count_col, level_col):
        if col not in df.columns:
            notes.append(f"missing column {col}")
    if notes:
        return SelectionReport(count_col, level_col, pd.DataFrame(), alpha,
                               np.nan, 0, 0, np.nan, np.nan, notes)

    rows = []
    for group, sub in df.groupby(group_col):
        s = sub.dropna(subset=[count_col, level_col])
        if len(s) < 6 or np.ptp(s[count_col]) == 0 or np.ptp(s[level_col]) == 0:
            reason = (f"n = {len(s)}" if len(s) < 6
                      else "count or level is constant")
            rows.append({"group": group, "n": len(s), "rho": np.nan,
                         "p": np.nan, "count_median": np.nan,
                         "untestable_because": reason})
            continue
        rho, p = stats.spearmanr(s[level_col], s[count_col])
        rows.append({"group": group, "n": len(s), "rho": float(rho),
                     "p": float(p),
                     "count_median": float(np.median(s[count_col])),
                     "untestable_because": None})

    table = pd.DataFrame(rows)
    testable = table["p"].notna()
    table["q"] = np.nan
    if testable.any():
        table.loc[testable, "q"] = benjamini_hochberg(table.loc[testable, "p"])
    for _, row in table[~testable].iterrows():
        notes.append(f"{row['group']}: not tested, {row['untestable_because']}")

    counts = df[count_col].dropna()
    over = int((counts > ceiling).sum()) if ceiling is not None else 0
    if ceiling is not None and over:
        notes.append(
            f"{over} of {len(counts)} images report more than {ceiling} "
            f"objects, which the preparation cannot contain")

    head = shift = np.nan
    if threshold is not None:
        levels = df[level_col].dropna()
        head = headroom(levels, threshold)
        sigma = float(np.std(levels, ddof=1)) if len(levels) > 1 else np.nan
        if np.isfinite(sigma) and sigma > 0:
            shift = float(np.ptp(levels) / sigma)

    return SelectionReport(
        count_col=count_col, level_col=level_col,
        table=table.sort_values("q", na_position="last").reset_index(drop=True),
        alpha=alpha,
        ceiling=np.nan if ceiling is None else float(ceiling),
        over_ceiling=over, n_images=int(len(counts)),
        headroom_sigma=head, headroom_shift_sigma=shift, notes=notes)
