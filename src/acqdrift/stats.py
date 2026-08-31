"""Statistical primitives shared by the schedule and drift audits.

Everything here is deliberately non-parametric. Image-level summary statistics
are bounded, skewed and occasionally contaminated by a bright artefact, so
rank-based association and a median-of-slopes estimator are safer defaults
than Pearson correlation and ordinary least squares.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def benjamini_hochberg(pvalues):
    """Return Benjamini-Hochberg adjusted p-values (q-values).

    A drift audit tests several metrics across several groups, so the family of
    tests is large enough that uncorrected p-values would manufacture findings.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    # enforce monotonicity from the largest p-value downwards
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def min_detectable_rho(n, alpha=0.05, power=0.80):
    """Smallest |rho| a Spearman test can detect with n observations.

    Reported alongside every non-significant result. "No drift detected" with
    n = 8 is a statement about the sample size, not about the microscope, and
    the report should say so rather than let the reader assume otherwise.

    Uses the Fisher z approximation, which treats the Spearman coefficient as
    if it were a Pearson coefficient. That is mildly optimistic; the returned
    value is a floor on what is detectable, not a guarantee.
    """
    if n < 5:
        return np.nan
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    z = (z_alpha + z_beta) / np.sqrt(n - 3)
    return float(np.tanh(z))


def trend(time, values, alpha=0.05):
    """Monotone trend of `values` against `time`.

    Returns Spearman rho with its p-value for the significance question, and a
    Theil-Sen slope with a confidence interval for the magnitude question.
    These answer different things and the report needs both: rho says whether
    there is a trend, the slope says how big it is.
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    n = t.size

    out = {
        "n": int(n),
        "rho": np.nan,
        "p": np.nan,
        "slope": np.nan,
        "slope_lo": np.nan,
        "slope_hi": np.nan,
        "span": np.nan,
        "ties_frac": np.nan,
        "min_detectable_rho": min_detectable_rho(n, alpha=alpha),
    }
    if n < 5 or np.ptp(t) == 0 or np.ptp(v) == 0:
        return out

    rho, p = stats.spearmanr(t, v)
    slope, _intercept, lo, hi = stats.theilslopes(v, t, alpha=1 - alpha)
    out.update(rho=float(rho), p=float(p), slope=float(slope),
               slope_lo=float(lo), slope_hi=float(hi), span=float(np.ptp(t)),
               ties_frac=float(1.0 - np.unique(v).size / v.size))
    return out


def as_percent_change(slope, level, window):
    """Express a slope as a percent change over a stated time window.

    `level` is the median of the metric, used as the denominator so that a
    single outlying image cannot inflate or deflate the reported percentage.
    """
    if not np.isfinite(slope) or not np.isfinite(level) or level == 0:
        return np.nan
    return float(100.0 * slope * window / level)


def eta_squared(values, labels):
    """Fraction of variance in `values` explained by the categorical `labels`.

    Applied to acquisition timestamps and group labels this is the confounding
    statistic: eta^2 near 1 means knowing the group tells you almost exactly
    when the image was taken, so no analysis can separate the two.

    Unlike a rank correlation between time and group index, this does not
    depend on how the groups happen to be ordered.
    """
    v = np.asarray(values, dtype=float)
    lab = np.asarray(labels)
    ok = np.isfinite(v)
    v, lab = v[ok], lab[ok]
    if v.size < 2:
        return np.nan
    grand = v.mean()
    ss_total = float(((v - grand) ** 2).sum())
    if ss_total == 0:
        return np.nan
    ss_between = 0.0
    for g in np.unique(lab):
        sub = v[lab == g]
        ss_between += sub.size * (sub.mean() - grand) ** 2
    return float(ss_between / ss_total)


def eta_squared_null(n, k):
    """Expected eta^2 when the labels carry no information about the values.

    eta^2 is biased upwards by the number of groups: assigning n observations
    at random to k groups already explains (k-1)/(n-1) of the variance, before
    any real structure. With five groups over eighty images that floor is 0.05
    and can be ignored; with fifty groups over two hundred it is 0.25 and the
    fixed interpretive bands stop meaning what they say.

    Reported next to eta^2 so a raw value can be read against its own floor.
    """
    if n is None or k is None or n < 2 or k < 2 or k > n:
        return np.nan
    return float((k - 1) / (n - 1))


def epsilon_squared(values, labels):
    """Bias-corrected eta^2, subtracting the variance explained by chance.

    Zero when the labels explain no more than a random partition of the same
    sizes would. Negative values are returned as given rather than clipped,
    because a value below zero is itself informative: the groups are more
    evenly spread through the session than chance would put them.
    """
    v = np.asarray(values, dtype=float)
    lab = np.asarray(labels)
    ok = np.isfinite(v)
    v, lab = v[ok], lab[ok]
    n, k = v.size, np.unique(lab).size
    eta = eta_squared(v, lab)
    if not np.isfinite(eta) or n <= k:
        return np.nan
    return float(eta - (k - 1) / (n - k) * (1.0 - eta))


def eta_squared_permutation_p(values, labels, n_perm=10000, seed=0):
    """Permutation p-value for eta^2, shuffling the group labels.

    Answers: how often would a random assignment of these same images to these
    same group sizes produce confounding this severe?
    """
    rng = np.random.default_rng(seed)
    observed = eta_squared(values, labels)
    if not np.isfinite(observed):
        return np.nan, observed
    lab = np.asarray(labels).copy()
    count = 0
    for _ in range(n_perm):
        rng.shuffle(lab)
        if eta_squared(values, lab) >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1)), observed
