"""How much a background-subtracted integral amplifies the error in the background.

Integrated intensity above background is the difference between two large,
similar numbers:

    integral = raw_sum - background * n_voxels

In a sparsely labelled field the subtrahend is most of the minuend. When the
background accounts for a fraction f of the raw sum, a relative error e in the
background estimate arrives in the integral multiplied by

    amplification = f / (1 - f)

At f = 0.947, which is an ordinary number for a few labelled neurons in a
512x512 field, the amplification is 17.9. A background estimate good to one
percent yields an integral good to eighteen. The integral looks like a
measurement of the specimen and is mostly a measurement of the background.

None of this is a finding about any particular microscope. It is a property of
the estimator, it is computable before any comparison is run, and it decides
whether integrated intensity is a usable endpoint at all. The package reports
it for that reason, and reports it whether or not any drift was detected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConditioningReport:
    background_fraction: float
    signal_fraction: float
    amplification: float
    background_cv_pct: float
    induced_integral_cv_pct: float
    n_images: int

    @property
    def verdict(self) -> str:
        """Whether the integral survives the noise already present in the background.

        The comparison is against the measured image-to-image scatter of the
        background, not against an assumed error. If that scatter alone, once
        amplified, exceeds the effect sizes anyone would look for, the endpoint
        is unusable no matter how the groups were scheduled.
        """
        induced = self.induced_integral_cv_pct
        if not np.isfinite(induced):
            return "UNKNOWN"
        if induced >= 25.0:
            return "ILL-CONDITIONED"
        if induced >= 10.0:
            return "FRAGILE"
        return "WELL-CONDITIONED"

    def tolerable_background_error_pct(self, target_integral_pct=5.0):
        """How well the background must be known for a stated integral accuracy.

        Usually the most useful number in the report, because it is the one an
        experimenter can act on: it says what a background estimator has to
        achieve before the integral means anything.
        """
        if not np.isfinite(self.amplification) or self.amplification == 0:
            return np.nan
        return float(target_integral_pct / self.amplification)


def amplification(background_fraction):
    """f / (1 - f), the error multiplier of a background-subtracted integral."""
    f = float(background_fraction)
    if not np.isfinite(f) or f >= 1.0:
        return np.inf
    if f <= 0.0:
        return 0.0
    return float(f / (1.0 - f))


def assess_integral(raw_sum, background, n_voxels):
    """Condition of the integral estimator over a set of images.

    Parameters
    ----------
    raw_sum : per-image sum of the unmodified volume.
    background : per-image background level, in the same units per voxel.
    n_voxels : voxels per image, scalar or per-image.

    The background fraction is taken as the median across images rather than
    computed from pooled totals, so that one unusually bright or unusually
    empty field cannot set the conditioning number for the whole session.
    """
    total = np.asarray(raw_sum, dtype=float)
    back = np.asarray(background, dtype=float)
    n = np.broadcast_to(np.asarray(n_voxels, dtype=float), total.shape)

    ok = np.isfinite(total) & np.isfinite(back) & np.isfinite(n) & (total > 0)
    if ok.sum() < 2:
        return ConditioningReport(np.nan, np.nan, np.nan, np.nan, np.nan,
                                  int(ok.sum()))

    fraction = np.median((back[ok] * n[ok]) / total[ok])
    amp = amplification(fraction)

    level = np.median(back[ok])
    scatter = np.std(back[ok], ddof=1)
    cv = 100.0 * scatter / level if level else np.nan

    return ConditioningReport(
        background_fraction=float(fraction),
        signal_fraction=float(1.0 - fraction),
        amplification=float(amp),
        background_cv_pct=float(cv),
        induced_integral_cv_pct=float(cv * amp) if np.isfinite(cv) else np.nan,
        n_images=int(ok.sum()))


def assess_peak(peak_level, background_level, offset=0.0):
    """The same question for a peak-minus-background measurement.

    Peak height is conditioned very differently from an integral and the two
    are routinely confused. Subtracting a background of 147 counts from a peak
    of 6473 moves the peak by 2.3 percent, so a background error that ruins the
    integral leaves the peak essentially untouched.

    Returned as the fraction of the peak that the background accounts for. Both
    arguments should be raw, with the pedestal still in: it is present in both
    and cancels in the subtraction. `offset` is accepted only to express the
    result against the net peak where that is the quantity of interest.
    """
    peak = float(np.nanmedian(np.asarray(peak_level, dtype=float)))
    back = float(np.nanmedian(np.asarray(background_level, dtype=float)))
    net_peak = peak - float(offset)
    if not np.isfinite(net_peak) or net_peak <= 0:
        return np.nan
    return float(100.0 * (back - float(offset)) / net_peak)
