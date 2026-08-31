"""Rendering. Turns the two audits into something a person reads and acts on.

The wording matters as much as the numbers. A quality-control tool that says
"PASS" invites the reader to stop thinking, and a tool that flags a confounded
design without saying that the confound is irreversible invites the reader to
believe a covariate adjustment will fix it. Neither is true, so neither is
written here.
"""
from __future__ import annotations

import numpy as np

RULE = "=" * 74


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def render_schedule(report) -> str:
    out = [RULE, "LEVEL 1  ACQUISITION SCHEDULE", RULE, ""]
    out.append(f"{report.n_images} images, {report.n_groups} groups, "
               f"session {report.session_minutes:.0f} min")
    out.append("")
    out.append("Acquisition window per group")
    for group, row in report.blocks.iterrows():
        out.append(f"  {group:<18} n={int(row['n']):3d}   "
                   f"{row['start']:7.1f} - {row['end']:7.1f} min")
    out.append("")

    out.append(f"Variance in acquisition time explained by group: "
               f"eta^2 = {_fmt(report.eta_squared)}  (permutation P = {_fmt(report.eta_p, 4)})")
    out.append("  eta^2 is not a correlation. It is the fraction of the spread")
    out.append("  in acquisition times that knowing the group already accounts")
    out.append("  for. At 1.0 the group tells you exactly when the image was")
    out.append("  taken, and the two cannot be separated by any analysis.")
    overlapping = int((report.overlap["overlap_min"] > 0).sum())
    total_pairs = len(report.overlap)
    out.append(f"Group pairs whose windows overlap in time: {overlapping} of {total_pairs}")
    out.append(f"Longest same-group streak: {report.max_run} "
               f"(expected under random order: {report.max_run_expected:.1f})")
    out.append("")

    if report.settings_varying:
        out.append("Instrument settings that CHANGED during the session:")
        for key, values in report.settings_varying.items():
            out.append(f"  {key.replace('set_', ''):<22} {values}")
        out.append("  Resolve these before interpreting anything below.")
    elif report.settings_constant:
        out.append(f"Instrument settings constant across all files "
                   f"({len(report.settings_constant)} checked).")
    else:
        out.append("No instrument settings were supplied, so none were checked.")
    out.append("")

    out.append(f"VERDICT: {report.verdict}")
    if report.verdict == "CONFOUNDED":
        out += [
            "",
            "  Group and acquisition time are not separable in this design.",
            "  Any difference between groups is also a difference between",
            "  times of day, and no statistical adjustment can tell the two",
            "  apart: regressing out time removes the treatment effect with",
            "  it. If Level 2 finds drift, the comparison between groups",
            "  cannot be rescued from these files.",
        ]
    elif report.verdict == "AT RISK":
        out += [
            "",
            "  Groups are partly separated in time. Effects will be",
            "  attenuated or inflated depending on the direction of any",
            "  drift; check Level 2 before drawing conclusions.",
        ]
    return "\n".join(out)


def render_drift(report, top=12) -> str:
    out = [RULE, "LEVEL 2  DRIFT IN RAW IMAGE STATISTICS", RULE, ""]
    out.append("Trend of each metric against acquisition time, within group.")
    out.append("Treatment is constant inside a group, so a trend here is not")
    out.append("the treatment effect.")
    out.append("")

    if report.table.empty:
        return "\n".join(out + ["No testable metrics."])

    header = (f"{'group':<16}{'metric':<22}{'n':>4}{'span':>7}"
              f"{'rho':>8}{'q':>9}{'%/span':>9}")
    out += [header, "-" * len(header)]
    for _, row in report.table.head(top).iterrows():
        marks = ""
        if np.isfinite(row["q"]) and row["q"] < report.alpha:
            marks += " *"
        if (row.get("ties_frac", 0) > 0.5 and abs(row.get("pct_over_span", 0) or 0) < 1e-9
                and np.isfinite(row["rho"]) and abs(row["rho"]) > 0.3):
            marks += " t"
        out.append(f"{str(row['group']):<16}{str(row['metric']):<22}"
                   f"{int(row['n']):>4}{row['span_min']:>7.0f}"
                   f"{_fmt(row['rho']):>8}"
                   f"{_fmt(row['q'], 4):>9}{_fmt(row['pct_over_span'], 1):>9}{marks}")
    out += ["", f"  q = Benjamini-Hochberg adjusted p over all "
            f"{len(report.table)} group x metric tests. It is not a p-value:",
            "      read it as the share of flagged rows expected to be false.",
            "  * = q < %.2f" % report.alpha,
            "  t = metric heavily tied; the trend is real but its magnitude is",
            "      below the quantisation of the measurement",
            "  %/span is measured across the window each group was actually",
            "  acquired in, and is never extrapolated beyond it. A group",
            "  spanning 29 min is not reported as a rate over the 428 min",
            "  session, because the slope was never observed over that range.",
            ""]

    out.append(f"VERDICT: {report.verdict}")
    if report.verdict == "UNDERPOWERED":
        floor = report.table["min_detectable_rho"].min()
        out += ["",
                f"  No trend reached significance, but with these group sizes the",
                f"  smallest detectable |rho| is {_fmt(floor)}. Absence of a finding",
                f"  here is weak evidence of absence of drift."]
    elif report.verdict == "DRIFT DETECTED":
        worst = report.significant.iloc[0]
        out += ["",
                f"  Strongest: {worst['metric']} in {worst['group']}, "
                f"rho = {_fmt(worst['rho'])}, "
                f"{_fmt(worst['pct_over_span'], 1)}% across that group's "
                f"{worst['span_min']:.0f} min window.",
                "  Detecting drift is not correcting it. Read the Level 1",
                "  verdict before deciding whether these files can support a",
                "  comparison between groups."]
    return "\n".join(out)


def render_power(drift_report) -> str:
    """How many images each metric would need to detect the drift it shows.

    The point of this table is that the cheapest metrics are often the most
    sensitive, so the sentinel used for quality control need not be, and
    usually should not be, the biological endpoint.
    """
    table = drift_report.table.dropna(subset=["rho"]).copy()
    if table.empty:
        return ""
    z = np.arctanh(table["rho"].abs().clip(upper=0.999))
    table["n_required"] = np.ceil((1.96 + 0.84) ** 2 / z ** 2 + 3)
    table = table.sort_values("n_required")

    out = [RULE, "SENSITIVITY  images needed to detect the observed drift", RULE, ""]
    out.append(f"{'group':<16}{'metric':<22}{'|rho|':>8}{'n required':>12}")
    out.append("-" * 58)
    for _, row in table.head(12).iterrows():
        out.append(f"{str(row['group']):<16}{str(row['metric']):<22}"
                   f"{abs(row['rho']):>8.3f}{int(row['n_required']):>12}")
    return "\n".join(out)


def render_sentinel(check) -> str:
    out = [RULE, f"SENTINEL VALIDITY  {check.channel}", RULE, "",
           "Does the background sentinel move with the amount of signal in the",
           "frame, once any shared trend with time is removed? If it does, it",
           "is not independent of the biology and cannot serve as a control.",
           ""]
    if check.table.empty:
        return "\n".join(out + [f"Not testable. {'; '.join(check.notes)}"])

    header = (f"{'group':<16}{'n':>4}{'separation':>12}{'rho_partial':>13}"
              f"{'q':>9}{'range %':>10}")
    out += [header, "-" * len(header)]
    for _, row in check.table.iterrows():
        flag = " !" if (np.isfinite(row["q"]) and row["q"] < check.alpha
                        and row["rho_partial"] > 0) else ""
        if row.get("untestable_because"):
            flag = " ?"
        out.append(f"{str(row['group']):<16}{int(row['n']):>4}"
                   f"{_fmt(row.get('separation'), 2):>12}"
                   f"{_fmt(row['rho_partial']):>13}{_fmt(row['q'], 4):>9}"
                   f"{_fmt(row['sentinel_range_pct'], 1):>10}{flag}")
    out += ["", "  separation = signal proxy over sentinel; below 1.5 the proxy",
            "               is not distinguishable from the background",
            "  ! = sentinel tracks signal content; Level 2 is not usable here",
            "  ? = not testable in this group; see notes"]
    for note in check.notes:
        out.append(f"  note: {note}")
    out += ["", f"VERDICT: {check.verdict}"]
    return "\n".join(out)


def render_fitness(fitness) -> str:
    out = [RULE, f"FITNESS: {fitness.verdict}", RULE, ""]
    for reason in fitness.reasons:
        out += _wrap(reason, bullet="  - ")
    if fitness.remedy:
        out += [""] + _wrap(fitness.remedy, bullet="  Remedy: ")
    out += ["", "  Not examined by this audit:"]
    from .fitness import NOT_EXAMINED
    for item in NOT_EXAMINED:
        out += _wrap(item, bullet="    * ", width=68)
    if fitness.verdict == "NO EVIDENCE OF UNFITNESS":
        out += ["",
                "  This is not a pass. It records that the checks which were",
                "  run did not fire, over the failure modes they cover."]
    return "\n".join(out)


def render_offset(report) -> str:
    out = [RULE, "DETECTOR PEDESTAL", RULE, "",
           f"Offset of {report.offset:.0f} counts, from {report.source}.",
           ""]
    out.append(f"{report.column} over {report.span_min:.0f} min: "
               f"{report.change_counts:+.2f} counts "
               f"(rho = {_fmt(report.rho)}, P = {_fmt(report.p, 4)})")
    out.append("")
    out.append(f"  as a share of the raw level ({report.raw_level:.1f}) "
               f"{_fmt(report.pct_raw, 1)} %")
    out.append(f"  as a share of the net level ({report.net_level:.1f}) "
               f"{_fmt(report.pct_net, 1)} %")
    out.append("")
    if report.matters:
        out += _wrap(
            f"Leaving the pedestal in the denominator understates the drift by "
            f"a factor of {_fmt(report.inflation, 1)}. The net figure is the "
            f"one to quote: the pedestal is a constant the detector adds and "
            f"was never light, so it belongs in neither the numerator nor the "
            f"denominator of a change.", bullet="  ")
    else:
        out.append("  The pedestal is small enough here that removing it barely")
        out.append("  changes the reported drift.")
    return "\n".join(out)


def render_conditioning(report) -> str:
    out = [RULE, "CONDITIONING  integrated intensity above background", RULE, ""]
    if not np.isfinite(report.amplification):
        return "\n".join(out + ["Not computable from the columns supplied."])

    out.append(f"Background accounts for {100 * report.background_fraction:.1f} % "
               f"of the raw sum; signal for {100 * report.signal_fraction:.1f} %.")
    out.append(f"Error amplification of the subtraction: "
               f"{report.amplification:.1f}x")
    out.append("")
    out.append(f"  a 1 % error in the background becomes "
               f"{report.amplification:.0f} % in the integral")
    out.append(f"  the measured image-to-image scatter of the background is "
               f"{_fmt(report.background_cv_pct, 1)} %,")
    out.append(f"  which alone puts {_fmt(report.induced_integral_cv_pct, 0)} % "
               f"of noise into the integral")
    out.append(f"  for an integral good to 5 %, the background must be known "
               f"to {_fmt(report.tolerable_background_error_pct(5.0), 2)} %")
    out += ["", f"VERDICT: {report.verdict}"]
    if report.verdict == "ILL-CONDITIONED":
        out += [""] + _wrap(
            "Integrated intensity is not a usable endpoint at this signal "
            "fraction, before any question of drift or scheduling arises. "
            "Prefer a measurement that does not subtract a large background: "
            "peak height, or a ratio taken within each object.", bullet="  ")
    return "\n".join(out)


def render_mounts(report) -> str:
    out = [RULE, "BATCH STRUCTURE  where the drift lives", RULE, ""]
    out.append(f"{report.n_mounts} batches recovered from gaps in the "
               f"timestamps, sizes {report.sizes}.")
    out.append(f"Median interval {report.median_interval_min:.1f} min; "
               f"a gap above {report.gap_threshold_min:.1f} min starts a new "
               f"batch; batches average {report.mean_mount_minutes:.0f} min.")
    out.append("")
    out.append(f"Between batches   rho = {_fmt(report.between_rho)}  "
               f"P = {_fmt(report.between_p, 4)}  "
               f"slope = {_fmt(report.between_slope, 4)} per min")
    out.append(f"Within batches    rho = {_fmt(report.within_rho)}  "
               f"P = {_fmt(report.within_p, 4)}  "
               f"slope = {_fmt(report.within_slope, 4)} per min")
    out.append(f"                  95 % CI [{_fmt(report.within_slope_lo, 4)}, "
               f"{_fmt(report.within_slope_hi, 4)}]")
    out.append(f"                  F = {_fmt(report.within_f, 2)}  "
               f"P = {_fmt(report.within_f_p, 3)}  "
               f"R^2 {_fmt(report.r2_time, 4)} -> {_fmt(report.r2_time_plus_within, 4)}")
    out += ["", f"VERDICT: {report.locus}"] + [""] + _wrap(report.remedy, bullet="  ")
    if report.locus in ("BETWEEN MOUNTS", "NEITHER RESOLVED"):
        out += [""] + _wrap(
            "The within-batch test is the weaker of the two by construction, "
            "since a batch is short and the session is long. Read its "
            "confidence interval before treating a flat result as an absence.",
            bullet="  ")
    return "\n".join(out)


def render_selection(report) -> str:
    out = [RULE, "THRESHOLD SELECTION", RULE, "",
           "Does the number of detected objects track how bright the image was?",
           "Within a group the true count cannot be changing, so a positive",
           "correlation means the threshold is admitting a different sample",
           "rather than the detector measuring a different value.", ""]
    if report.table.empty:
        return "\n".join(out + [f"Not testable. {'; '.join(report.notes)}"])

    header = f"{'group':<16}{'n':>4}{'count':>8}{'rho':>9}{'q':>9}"
    out += [header, "-" * len(header)]
    for _, row in report.table.iterrows():
        flag = " !" if (np.isfinite(row["q"]) and row["q"] < report.alpha
                        and row["rho"] > 0) else ""
        if row.get("untestable_because"):
            flag = " ?"
        out.append(f"{str(row['group']):<16}{int(row['n']):>4}"
                   f"{_fmt(row['count_median'], 1):>8}{_fmt(row['rho']):>9}"
                   f"{_fmt(row['q'], 4):>9}{flag}")
    out.append("")
    if np.isfinite(report.ceiling):
        out.append(f"Images reporting more than {report.ceiling:.0f} objects: "
                   f"{report.over_ceiling} of {report.n_images} "
                   f"({100 * report.over_ceiling_frac:.1f} %)")
        out.append("  Counts above a stated biological ceiling are false")
        out.append("  positives by construction, with no ground truth needed.")
    if np.isfinite(report.headroom_sigma):
        out.append(f"Threshold sits {report.headroom_sigma:.1f} background "
                   f"standard deviations above the background level, while the "
                   f"background itself moves {report.headroom_shift_sigma:.1f} "
                   f"across the session.")
    for note in report.notes:
        out.append(f"  note: {note}")
    out += ["", f"VERDICT: {report.verdict}"]
    if report.remedy:
        out += [""] + _wrap(report.remedy, bullet="  ")
    return "\n".join(out)


def render_negative_control(control) -> str:
    out = [RULE, f"NEGATIVE CONTROL  {control.value_col} within {control.group}",
           RULE, "",
           "A group that received no treatment, split against itself by",
           "acquisition time. Anything separating its early images from its",
           "late ones is artefact, so a correction that works shrinks it.", ""]
    out.append(f"  n = {control.n_early} early against {control.n_late} late")
    out.append(f"  before correction  {_fmt(control.artefact_raw_pct, 1):>8} %   "
               f"rho = {_fmt(control.rho_raw)}  P = {_fmt(control.p_raw, 3)}")
    out.append(f"  after correction   {_fmt(control.artefact_corrected_pct, 1):>8} %   "
               f"rho = {_fmt(control.rho_corrected)}  P = {_fmt(control.p_corrected, 3)}")
    out.append(f"  artefact removed   {_fmt(control.reduction_pct, 1):>8} %")
    out += ["", f"VERDICT: {control.verdict}", ""] + _wrap(control.note, bullet="  ")
    out += [""] + _wrap(
        "This verdict is only worth what the corrected measurement is worth. "
        "It requires the measurement to have been recomputed from the "
        "corrected images with the same detection settings. Rescaling numbers "
        "already extracted from the raw images tests nothing.", bullet="  ")
    return "\n".join(out)


def render_records(check) -> str:
    out = [RULE, "ACQUISITION RECORD", RULE, ""]
    out.append(check.summary())
    return "\n".join(out)


def _wrap(text, bullet="  - ", width=70):
    import textwrap
    pad = " " * len(bullet)
    lines = textwrap.wrap(text, width=width)
    return [bullet + lines[0]] + [pad + line for line in lines[1:]]
