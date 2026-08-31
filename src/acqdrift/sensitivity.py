"""Correcting drift, and the negative control that decides whether it worked.

A multiplicative correction is easy to fit and easy to believe. Model the
level as decaying with the session, divide it back out, and the background
flattens: the correlation with time collapses from rho = -0.861 to +0.036 and
the difference between groups goes from P < 1e-13 to P = 0.111. Every number a
report would print gets better.

None of that is evidence that the correction worked. It is evidence that a
curve fitted to the background describes the background, which it must, since
that is what it was fitted to. Judging a correction on the quantity used to
fit it is circular, and it is the standard way this goes wrong.

The test that is not circular is a group that received no treatment, split
against itself by acquisition time. Whatever separates its early images from
its late ones is artefact by construction. A correction that works shrinks it.
In the session this package was built from, the artefact was +51.5 % before
correction and +56.2 % after: the fit was clean, the arithmetic was right, and
the correction did nothing for the thing that needed fixing. It could not, as
it turned out, because the artefact came from a fixed detection threshold
choosing different objects rather than from the gain scaling the same ones.

Hence the shape of this module. `fit_gain` and `apply_gain` are ordinary and
short. `validate` is the part that matters, and nothing here reports a
correction as successful on the strength of the fit alone.

One requirement cannot be met inside this module and has to be met by the
caller: the corrected measurement must be recomputed from the corrected
images, running the same detection and the same thresholds. Rescaling numbers
already extracted from the raw images tests nothing, because the selection
that did the damage has already happened by then.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats


@dataclass
class GainFit:
    a: float
    b: float
    tau_min: float
    offset: float
    k_first: float
    k_last: float
    r2: float
    n: int

    def k(self, t):
        """Relative gain at time t, normalised to 1.0 at the session start."""
        t = np.asarray(t, dtype=float)
        return (self.a + self.b * np.exp(-t / self.tau_min)) / (self.a + self.b)

    @property
    def span_pct(self) -> float:
        """Total attenuation across the session, as a percentage."""
        return float(100.0 * (self.k_last - self.k_first) / self.k_first)


@dataclass
class NegativeControl:
    group: str
    value_col: str
    n_early: int
    n_late: int
    artefact_raw_pct: float
    artefact_corrected_pct: float
    rho_raw: float
    rho_corrected: float
    p_raw: float
    p_corrected: float

    @property
    def reduction_pct(self) -> float:
        """How much of the artefact the correction removed. Negative if it grew."""
        a, b = abs(self.artefact_raw_pct), abs(self.artefact_corrected_pct)
        if not np.isfinite(a) or a == 0:
            return np.nan
        return float(100.0 * (a - b) / a)

    @property
    def verdict(self) -> str:
        r = self.reduction_pct
        if not np.isfinite(r):
            return "NOT TESTABLE"
        if r <= 0:
            return "CORRECTION REJECTED"
        if r < 50:
            return "CORRECTION INSUFFICIENT"
        return "CORRECTION REDUCES ARTEFACT"

    @property
    def note(self) -> str:
        return {
            "CORRECTION REJECTED":
                "The artefact in an untreated group is no smaller after "
                "correction. Whatever drives it is not what the correction "
                "models, so applying it adds a step without fixing anything.",
            "CORRECTION INSUFFICIENT":
                "The artefact shrank but survives at a size comparable to the "
                "effects being looked for.",
            "CORRECTION REDUCES ARTEFACT":
                "The artefact in an untreated group shrank substantially. This "
                "is the strongest available evidence for the correction and it "
                "is still not a guarantee: it covers the failure mode that was "
                "tested and no other.",
            "NOT TESTABLE":
                "No usable artefact in the control group to measure against.",
        }[self.verdict]


def fit_gain(times, levels, offset=0.0, tau_init=150.0) -> GainFit:
    """Fit level(t) = A + B exp(-t/tau) to a sentinel and return the gain curve.

    The pedestal is removed before fitting. Leaving it in flattens the curve
    towards a constant and biases tau upwards, because a fixed number of counts
    that never decays is being asked to decay.

    An exponential-plus-constant is a description, not a mechanism. It is used
    because detector and laser warm-up settle that way and because it has few
    enough parameters to fit on one session; a straight line usually does
    almost as well over a single session and is easier to defend.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(levels, dtype=float) - float(offset)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if t.size < 5:
        raise ValueError(f"need at least 5 images to fit a gain curve, got {t.size}")
    if np.any(y <= 0):
        raise ValueError("levels fall to or below the offset; check the offset")

    def model(x, a, b, tau):
        return a + b * np.exp(-x / tau)

    guess = [float(y.min()), float(max(y.max() - y.min(), 1e-6)), float(tau_init)]
    try:
        popt, _ = optimize.curve_fit(model, t, y, p0=guess, maxfev=20000)
        a, b, tau = (float(v) for v in popt)
    except (RuntimeError, ValueError):
        slope, intercept = np.polyfit(t, y, 1)
        tau = 1e6
        a, b = float(intercept + slope * tau), 0.0

    resid = y - model(t, a, b, tau)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (resid @ resid) / tss) if tss > 0 else np.nan

    fit = GainFit(a=a, b=b, tau_min=tau, offset=float(offset),
                  k_first=1.0, k_last=1.0, r2=r2, n=int(t.size))
    fit.k_first = float(fit.k(t.min()))
    fit.k_last = float(fit.k(t.max()))
    return fit


def apply_gain(values, k, offset=0.0):
    """Undo a multiplicative gain change, with the pedestal handled correctly.

        corrected = (raw - offset) / k + offset

    The order is not cosmetic. The instrument adds the pedestal after the gain
    stage, so the pedestal was never scaled and must not be unscaled. Dividing
    the raw value including the pedestal inflates the background by a further
    offset * (1/k - 1) counts, which is a drift of the opposite sign
    manufactured by the correction itself.
    """
    v = np.asarray(values, dtype=float)
    return (v - float(offset)) / np.asarray(k, dtype=float) + float(offset)


def _split_change(times, values):
    """Percent difference between the late and early halves of a group."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if t.size < 6:
        return np.nan, 0, 0
    mid = np.median(t)
    early, late = v[t <= mid], v[t > mid]
    if early.size < 2 or late.size < 2 or early.mean() == 0:
        return np.nan, int(early.size), int(late.size)
    change = 100.0 * (late.mean() - early.mean()) / early.mean()
    return float(change), int(early.size), int(late.size)


def validate(df, raw_col, corrected_col, control_group="Control",
             group_col="group", time_col="t_min") -> NegativeControl:
    """Judge a correction on a group that received no treatment.

    `raw_col` and `corrected_col` must both be measurements of the same
    quantity, extracted the same way, one from the original images and one
    from the corrected images. If the corrected column was produced by
    rescaling `raw_col` rather than by rerunning the measurement, this test
    is vacuous and will report a reduction that means nothing.
    """
    sub = df[df[group_col] == control_group]
    if sub.empty:
        raise ValueError(f"no rows for control group {control_group!r}")

    raw_change, n_early, n_late = _split_change(sub[time_col], sub[raw_col])
    cor_change, _, _ = _split_change(sub[time_col], sub[corrected_col])

    def corr(col):
        v = sub[col].to_numpy(dtype=float)
        t = sub[time_col].to_numpy(dtype=float)
        ok = np.isfinite(v) & np.isfinite(t)
        if ok.sum() < 6 or np.ptp(v[ok]) == 0 or np.ptp(t[ok]) == 0:
            return np.nan, np.nan
        rho, p = stats.spearmanr(t[ok], v[ok])
        return float(rho), float(p)

    rho_raw, p_raw = corr(raw_col)
    rho_cor, p_cor = corr(corrected_col)

    return NegativeControl(
        group=str(control_group), value_col=raw_col,
        n_early=n_early, n_late=n_late,
        artefact_raw_pct=raw_change, artefact_corrected_pct=cor_change,
        rho_raw=rho_raw, rho_corrected=rho_cor,
        p_raw=p_raw, p_corrected=p_cor)


def scale_invariance(raw_values, corrected_values):
    """How much a multiplicative correction moved a measurement at all.

    Near zero means the endpoint is invariant to the gain by construction, and
    so needs no correction and gains nothing from one. Any measurement defined
    as a ratio within the same image behaves this way: an area at half of each
    object's own peak, a ratio between two channels, a count. Establishing this
    is worth more than correcting, because an invariant endpoint sidesteps the
    whole argument about whether the correction was right.
    """
    a = np.asarray(raw_values, dtype=float)
    b = np.asarray(corrected_values, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() == 0:
        return np.nan
    level = float(np.nanmedian(a[ok]))
    if level == 0:
        return np.nan
    return float(100.0 * (np.nanmedian(b[ok]) - level) / level)


def sweep_tau(df, level_col, time_col="t_min", offset=0.0,
              taus=(60, 100, 150, 200, 250, 400)):
    """Recompute the gain curve at fixed values of tau.

    Run when a correction has been rejected, to separate two explanations: the
    time constant was chosen badly, or the residual is not the kind of thing
    the correction removes. A residual that barely moves across a wide range of
    tau points at the second. That is not reassuring - it means the correction
    was never addressing what remains.
    """
    t = df[time_col].to_numpy(dtype=float)
    y = df[level_col].to_numpy(dtype=float) - float(offset)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]

    rows = []
    for tau in taus:
        basis = np.column_stack([np.ones_like(t), np.exp(-t / float(tau))])
        beta, *_ = np.linalg.lstsq(basis, y, rcond=None)
        a, b = float(beta[0]), float(beta[1])
        denom = a + b
        if denom == 0:
            continue
        k = (a + b * np.exp(-t / float(tau))) / denom
        resid = y - (a + b * np.exp(-t / float(tau)))
        tss = float(((y - y.mean()) ** 2).sum())
        rows.append({"tau_min": float(tau), "a": a, "b": b,
                     "k_last": float(k[np.argmax(t)]),
                     "r2": float(1.0 - (resid @ resid) / tss) if tss else np.nan})
    return pd.DataFrame(rows)
