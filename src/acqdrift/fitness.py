"""The combined verdict, and the list of things it does not cover.

Quality control reports fail in a predictable way: a green light gets read as
"the data are good" when all it ever meant was "the checks that were run did
not fire". The distinction is not pedantic. A confounded design plus measured
drift is a proof of unfitness. The absence of a finding is not a proof of
fitness, and cannot be turned into one by any amount of additional testing of
the same kind.

So this module returns one of three things and never a pass:

  UNFIT                       the files cannot support a comparison between
                              groups, and no reanalysis will change that
  INCONCLUSIVE                the audit could not establish either way
  NO EVIDENCE OF UNFITNESS    the checks that were run did not fire

Every verdict is printed alongside what was not examined.
"""
from __future__ import annotations

from dataclasses import dataclass, field

UNFIT = "UNFIT"
INCONCLUSIVE = "INCONCLUSIVE"
NO_EVIDENCE = "NO EVIDENCE OF UNFITNESS"

# Failure modes outside the reach of these two audits. Printed with every
# verdict, because the reader needs them to size what the result is worth.
NOT_EXAMINED = [
    "Images discarded before the files reached this audit. Selecting brighter "
    "or cleaner fields during acquisition biases the set, and the discarded "
    "images are not here to reveal it.",
    "Segmentation and thresholding. The audit stops before any object is "
    "detected, so bias introduced downstream is untouched.",
    "Drift that is not monotone in time. A step change after a realignment, a "
    "periodic fluctuation, or an effect that reverses mid-session will be "
    "underestimated by a rank correlation.",
    "Biological confounds that track acquisition order: animal age, time off "
    "food, plate of origin, position in the mounting queue.",
    "Observer effects in any scoring done by eye.",
]


@dataclass
class Fitness:
    verdict: str
    reasons: list = field(default_factory=list)
    remedy: str = ""

    @property
    def unfit(self) -> bool:
        return self.verdict == UNFIT


def assess(schedule_report, drift_report=None, sentinel_check=None,
           selection_report=None, conditioning_report=None) -> Fitness:
    """Combine every check that was run into one verdict.

    The order is not arbitrary. Settings that changed mid-session invalidate
    everything downstream, so they are checked first. Threshold selection comes
    next, because it is the one failure that a later correction provably cannot
    undo: the objects that fell below the line left no record. Only then do the
    schedule and drift verdicts get combined.
    """
    reasons = []

    if schedule_report.settings_varying:
        return Fitness(
            UNFIT,
            [f"Instrument settings changed during the session: "
             f"{', '.join(k.replace('set_', '') for k in schedule_report.settings_varying)}."],
            "Images acquired under different settings are not comparable. "
            "Split the session by setting, or reacquire.")

    if (selection_report is not None
            and selection_report.verdict == "THRESHOLD SELECTING"):
        detail = []
        if not selection_report.coupled.empty:
            groups = ", ".join(selection_report.coupled["group"].astype(str))
            detail.append(
                f"The number of detected objects tracks image brightness "
                f"within {groups}, where the treatment is constant and the "
                f"true count cannot be changing.")
        if selection_report.over_ceiling:
            detail.append(
                f"{selection_report.over_ceiling} of "
                f"{selection_report.n_images} images report more objects than "
                f"the preparation can contain.")
        detail.append(
            "A fixed threshold admitted a different sample from the bright "
            "images than from the dim ones. Objects that fell below the line "
            "left no record, so no rescaling applied afterwards can recover "
            "them: this is not a bias in the measured value, it is a change in "
            "what was measured.")
        detail.append(
            "The scope is every endpoint derived from that detector. Endpoints "
            "that never passed through it, including counts made by eye and "
            "behavioural assays, are unaffected.")
        return Fitness(UNFIT, detail, selection_report.remedy)

    confounded = schedule_report.verdict == "CONFOUNDED"
    at_risk = schedule_report.verdict == "AT RISK"

    if (conditioning_report is not None
            and conditioning_report.verdict == "ILL-CONDITIONED"):
        reasons.append(
            f"Integrated intensity above background amplifies the error in the "
            f"background by {conditioning_report.amplification:.0f}x, so the "
            f"scatter already present in the background puts "
            f"{conditioning_report.induced_integral_cv_pct:.0f}% of noise into "
            f"the integral. That endpoint is unusable here independently of "
            f"anything below; peak height and within-object ratios are not "
            f"affected.")

    if sentinel_check is not None and sentinel_check.verdict == "NOT TESTABLE":
        return Fitness(
            INCONCLUSIVE,
            ["The background sentinel could not be validated: "
             + "; ".join(sentinel_check.notes) + ".",
             "Without that check, a drift result from the sentinel cannot be "
             "distinguished from the biology moving it.",
             "The schedule audit above is unaffected and still stands."],
            "Record a treatment-independent channel, or use a signal proxy "
            "further into the bright tail.")

    if sentinel_check is not None and sentinel_check.verdict == "SENTINEL INVALID":
        groups = ", ".join(sentinel_check.failing["group"].astype(str))
        return Fitness(
            INCONCLUSIVE,
            [f"The background sentinel tracks the amount of signal in the "
             f"frame ({groups}), so it is not independent of the biology and "
             f"cannot be used as a control here.",
             "The schedule audit above is unaffected and still stands."],
            "Use a sentinel the treatment cannot move: a transmitted-light "
            "channel, a bead field, or a region of the frame with no specimen.")

    drift_found = (drift_report is not None
                   and drift_report.verdict == "DRIFT DETECTED")
    underpowered = (drift_report is not None
                    and drift_report.verdict == "UNDERPOWERED")

    if confounded and drift_found:
        worst = drift_report.significant.iloc[0]
        return Fitness(
            UNFIT,
            [f"Groups occupy separate blocks of the session "
             f"(eta^2 = {schedule_report.eta_squared:.3f}), so group and "
             f"acquisition time are collinear.",
             f"Signal drifts within groups, where the treatment is constant "
             f"({worst['metric']}, rho = {worst['rho']:+.3f}).",
             "Adjusting for time would remove the treatment effect along with "
             "the drift, so the confound cannot be undone after the fact."],
            "Reacquire with groups interleaved. `interleaved_order` proposes a "
            "schedule; a randomised order makes the same drift harmless.")

    if confounded:
        return Fitness(
            INCONCLUSIVE,
            [f"Groups occupy separate blocks of the session "
             f"(eta^2 = {schedule_report.eta_squared:.3f}).",
             "No drift reached significance, but the design offers no "
             "protection if any exists, so the comparison rests entirely on a "
             "negative result."
             + (" With these group sizes that negative result is weak."
                if underpowered else "")],
            "Interleave the next session rather than relying on this one being "
            "drift-free.")

    if drift_found:
        worst = drift_report.significant.iloc[0]
        reasons.append(
            f"Signal drifts within groups ({worst['metric']} in "
            f"{worst['group']}, rho = {worst['rho']:+.3f}, "
            f"{worst['pct_over_span']:+.1f}% across its window).")
        reasons.append(
            "Groups are interleaved well enough that time is not collinear "
            "with group, so the drift adds variance rather than bias.")
        return Fitness(INCONCLUSIVE, reasons,
                       "Drift is present but not confounded. Include "
                       "acquisition time as a covariate, and check that it "
                       "does not absorb the effect of interest.")

    if underpowered:
        floor = drift_report.table["min_detectable_rho"].min()
        return Fitness(
            INCONCLUSIVE,
            [f"No drift reached significance, but the smallest detectable "
             f"|rho| at these group sizes is {floor:.2f}.",
             "Nothing here distinguishes a clean session from a session too "
             "small to show its drift."],
            "More images per group, or a sentinel with less measurement noise.")

    if at_risk:
        reasons.append("Groups are partly separated in time; effects may be "
                       "attenuated or inflated by any undetected drift.")

    reasons.append("Groups are interleaved and no drift was detected in the "
                   "raw statistics at adequate power.")
    return Fitness(NO_EVIDENCE, reasons,
                   "Proceed, treating this as the absence of one class of "
                   "problem rather than a clean bill of health.")
