"""Every v0.2 check, run from the committed record. No raw files needed.

`run.py` reads the original .czi files and needs the 81 stacks. This script
reads `acquisition_record.csv` and `correction_check.csv`, which are in the
repository, so anyone who clones it can reproduce the worked example in a
second and see what the checks say on a session that failed all of them.

    python examples/celegans_day5/run_audits.py

`correction_check.csv` is the part worth looking at twice. It holds the same
measurement extracted two ways: once from the original volumes and once from
volumes with the fitted gain divided back out, with the detector rerun from
scratch on both. That is what makes the negative control mean anything, and it
is not something the package can produce on its own.
"""
from pathlib import Path

import pandas as pd

import acqdrift as ad

HERE = Path(__file__).parent
OFFSET = 100.0             # digital offset, constant across all 81 files
N_VOXELS = 512 * 512 * 10  # 512x512, ten planes
CEILING = 6                # dopaminergic neurons in the head of C. elegans
THRESHOLD = 250.0          # the absolute detection threshold that was used


def main():
    record, check = ad.from_csv(HERE / "acquisition_record.csv")
    correction = pd.read_csv(HERE / "correction_check.csv")
    merged = correction.merge(record[["file", "GFP.background"]], on="file")

    print(ad.render_records(check), "\n")

    settings = [c for c in record.columns if c.startswith("set_")]
    schedule = ad.audit_schedule(record, settings_cols=settings)
    print(ad.render_schedule(schedule), "\n")

    print(ad.render_offset(
        ad.audit_offset(record, "GFP.background", offset=OFFSET)), "\n")

    print(ad.render_conditioning(
        ad.assess_integral(record["GFP.total"], record["GFP.background"],
                           N_VOXELS)), "\n")

    print(ad.render_mounts(ad.decompose(record, "GFP.background")), "\n")

    selection = ad.audit_threshold(merged, count_col="n_objects_raw",
                                   level_col="GFP.background",
                                   threshold=THRESHOLD, ceiling=CEILING)
    print(ad.render_selection(selection), "\n")

    # The correction: fitted on the background, judged on an untreated group.
    fit = ad.fit_gain(record["t_min"], record["GFP.background"], offset=OFFSET)
    print(f"GAIN FIT  A = {fit.a:.2f}  B = {fit.b:.2f}  "
          f"tau = {fit.tau_min:.1f} min  R^2 = {fit.r2:.3f}")
    print(f"          k runs from {fit.k_first:.3f} to {fit.k_last:.3f}, "
          f"an attenuation of {abs(fit.span_pct):.1f} %\n")

    print(ad.render_negative_control(
        ad.validate_correction(correction, "peak_raw", "peak_corrected",
                               control_group="Control")), "\n")

    area = ad.scale_invariance(correction["area_raw"],
                               correction["area_corrected"])
    peak = ad.scale_invariance(correction["peak_raw"],
                               correction["peak_corrected"])
    print(f"Under the same correction the peak moves {peak:+.1f} % and the "
          f"area {area:+.2f} %.")
    print("The area is a ratio taken within each object, so the gain cannot "
          "touch it. It never needed correcting.\n")

    print(ad.render_fitness(
        ad.assess(schedule, selection_report=selection)))


if __name__ == "__main__":
    main()
