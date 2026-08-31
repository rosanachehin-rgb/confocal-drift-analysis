# Output reference

Everything `acqdrift` produces, field by field. For the six values whose
obvious reading is the wrong one, see *Reading the numbers* in the
[README](README.md); this file is the exhaustive list.

A run produces three things: an optional per-image table, six printed
sections, and an exit code.

---

## 1. The per-image table (`--csv`)

One row per image. In the worked example, 81 rows by 31 columns. The
statistical core consumes exactly this, so a session from any instrument can
be audited by building this table yourself.

### Identification

| Column | Meaning |
|---|---|
| `file` | filename as found on disk |
| `timestamp` | `AcquisitionDateAndTime` from the file metadata, ISO 8601 |
| `t_min` | minutes since the first image of the session |
| `group` | condition, from the filename parser you supply |
| `n_channels` | channels detected in the file |

### Geometry, read from the file rather than assumed

| Column | Meaning |
|---|---|
| `px_um` | pixel size in micrometres, from the metadata scaling block |
| `z_um` | z step in micrometres |

Reading these rather than hard-coding them is not pedantry. In this project a
z step recorded as 3.0 µm had been carried in code as 4.0 µm from an earlier
session.

### Instrument settings (`set_*`)

Ten tags are looked for: `PinholeSizeAiry`, `Voltage`, `DigitalGain`, `Offset`,
`Power`, `ExcitationWavelength`, `LaserScanPixelTime`, `BitsPerPixel`,
`Attenuation`, `Zoom`.

A tag absent from the file is left empty and reported as unchecked rather than
assumed constant. The schedule audit counts how many it could actually verify.
If any of these varies across a session, the fitness verdict short-circuits to
`UNFIT`: images taken under different settings are not comparable, and nothing
further needs testing.

### Image statistics, per channel

Computed on the raw voxels. No segmentation, no threshold, no object
detection. Column names are prefixed with the channel name, so a session with
`ZsGr1` and `ESID` yields fourteen of these.

| Column | Meaning |
|---|---|
| `<ch>.background` | median of every voxel. The sentinel |
| `<ch>.p01` | 1st percentile. Tracks the floor of the detector |
| `<ch>.p99` | 99th percentile |
| `<ch>.p999` | 99.9th percentile. Default signal proxy for the sentinel check |
| `<ch>.total` | sum of all voxels |
| `<ch>.mad` | median absolute deviation. Robust spread of the background |
| `<ch>.sat_frac` | fraction of voxels at the top of the bit depth |

`background` is the median because in a sparsely labelled preparation the
objects of interest occupy a small fraction of the field, so the median tracks
the background and cannot be moved by how many cells happen to be in frame.
That independence is the whole basis for using it as a control, and it is
tested rather than assumed — see section 5.

---

## 2. Level 1, acquisition schedule

Metadata only. No pixels are read, so this section runs in about a second on a
full session.

| Value | Meaning |
|---|---|
| acquisition window per group | first and last minute each group occupies |
| `eta_squared` | fraction of the variance in acquisition time explained by group |
| `eta_null` | what a random partition into the same group sizes already explains, (k−1)/(n−1) |
| `epsilon_squared` | η² with that floor removed; zero means no more than chance |
| `eta_p` | permutation p-value, from shuffled group labels |
| overlapping pairs | how many group pairs share any part of the session, out of all pairs |
| `max_run` | longest streak of consecutive images from one group |
| `max_run_expected` | mean longest streak under random ordering, for comparison |
| `min_group_n` | images in the smallest group |
| `bands_reliable` | whether the η² cuts below apply to a design of this shape |
| settings checked | how many `set_*` tags were present and constant |
| `verdict` | `OK` (η² < 0.60), `AT RISK` (0.60–0.90), `CONFOUNDED` (≥ 0.90), `INCONCLUSIVE` |
| `verdict_reason` | why the bands were withheld, when they were |

Zero overlapping pairs together with a `max_run` far above expectation is the
signature of a design where each group was acquired as one uninterrupted block.

**η² has a floor that rises with the number of groups.** Assigning n images to k
groups at random already explains (k−1)/(n−1) of the variance in acquisition
time. With five groups over eighty images that floor is 0.05 and can be ignored.
With fifty groups over two hundred it is 0.25, and a raw η² read against the
cuts above would call a well-interleaved session confounded. The bands assume a
handful of groups over many images, so the audit returns `INCONCLUSIVE` rather
than a verdict when the smallest group holds fewer than 5 images or when
`eta_null` exceeds 0.15. Both cuts are conventions chosen to be legible.

`INCONCLUSIVE` here is not a pass. It means the schedule was measured and could
not be judged, usually because the column supplied as `group` was a replicate or
acquisition identifier rather than the comparison the study makes.

---

## 3. Level 2, drift in raw statistics

One row per group per metric. Sorted by q, so the strongest findings are at the
top. In the worked example this is 5 groups × 14 metrics = 70 rows, of which
the report prints the first twelve.

| Column | Meaning |
|---|---|
| `group`, `metric` | which test this row is |
| `n` | images in the group |
| `span_min` | minutes the group spans |
| `level` | median of the metric, used as the denominator for percentages |
| `rho` | Spearman correlation against acquisition time, within the group |
| `p` | uncorrected p-value for that correlation |
| `q` | Benjamini–Hochberg adjusted across all rows |
| `pct_over_span` | Theil–Sen slope as a percent of `level`, across `span_min` |
| `pct_per_hour` | the same slope per hour |
| `extrapolation` | 60 / `span_min`. Above 2, `pct_per_hour` reaches well past the data |
| `slope_lo`, `slope_hi` | confidence interval on the Theil–Sen slope |
| `ties_frac` | fraction of repeated values in the metric |
| `min_detectable_rho` | smallest \|ρ\| this group size could find at 80% power |
| `verdict` | `DRIFT DETECTED`, `NO DRIFT DETECTED`, `UNDERPOWERED` |

Printed markers: `*` for q below alpha, `t` for a metric so heavily tied that
the slope reads zero while the rank correlation is strong. The second means the
trend is real but its size falls below the quantisation of the measurement.

Association and magnitude come from different estimators on purpose. Spearman
answers whether there is a trend; Theil–Sen answers how big it is, and being a
median of pairwise slopes it survives a single contaminated image.

---

## 4. Sensitivity

For every row of section 3, the number of images that would have been needed to
detect that association at 80% power and α = 0.05. Sorted ascending.

Its purpose is comparative. The cheapest metric is often the most sensitive, so
the sentinel used for quality control need not be, and usually should not be,
the biological endpoint. In the worked example the raw background needs 11
images and the segmented cell area needs 49.

---

## 5. Sentinel validity

Tests the assumption Level 2 rests on: that the sentinel is independent of the
biology. One row per group.

| Column | Meaning |
|---|---|
| `n` | images in the group |
| `separation` | median of the signal proxy over median of the sentinel |
| `rho_partial` | correlation of sentinel with signal proxy, time removed from both |
| `p`, `q` | significance, adjusted across groups |
| `sentinel_range_pct` | total travel of the sentinel, as a percent of its median |
| `untestable_because` | why a group could not be tested, when it could not |
| `verdict` | `SENTINEL VALID`, `SENTINEL VALID WHERE TESTED`, `SENTINEL INVALID`, `NOT TESTABLE` |

A positive, significant `rho_partial` means the sentinel moves with the amount
of bright material in the frame and therefore cannot serve as a control. Time
is removed from both variables first, because the sentinel and the proxy may
share a drift, and a raw correlation between them would be that shared drift
rather than any dependence of one on the other.

Four reasons a group is reported untestable, each named explicitly rather than
folded into a single silent failure: fewer than six images; a signal proxy that
does not separate from the sentinel; a sentinel that is a perfect monotone
function of time, leaving no residual to test; and a constant metric.

Printed markers: `!` for an invalid group, `?` for an untestable one.

---

## 6. Channel contrast

Present when both `--signal-channel` and `--reference-channel` are given. One
row per group per shared metric.

| Column | Meaning |
|---|---|
| `<signal>_rho`, `<signal>_q` | drift in the signal channel |
| `<reference>_rho`, `<reference>_q` | drift in the reference channel |
| `interpretation` | see below |

| `interpretation` | Reading |
|---|---|
| `fluorescence path` | signal drifts, reference does not: excitation, detection, or photophysics |
| `specimen or mount` | both drift: the preparation was changing |
| `reference only - check focus or stage` | unusual, and worth investigating before anything else |
| `no drift in either` | neither channel moved |

The report prints only rows that are not `no drift in either`.

This narrows the search. It does not identify a cause.

---

## 7. Fitness verdict

| Verdict | Meaning |
|---|---|
| `UNFIT` | the files cannot support a comparison between groups, and no reanalysis changes that |
| `INCONCLUSIVE` | the audit could not establish either way |
| `NO EVIDENCE OF UNFITNESS` | the checks that were run did not fire |

Printed with numbered reasons, a concrete remedy, and the list of failure modes
the audit does not cover: images discarded before it saw them, segmentation
bias, drift that is not monotone in time, biological confounds that track
acquisition order, and observer effects.

`NO EVIDENCE OF UNFITNESS` carries an explicit note that it is not a pass.

---

## 8. Exit code

`1` when the verdict is `UNFIT`, `0` otherwise, so a run can gate a pipeline:

```bash
acqdrift session/ --schedule-only || echo "redesign before analysing"
```

Note that `INCONCLUSIVE` exits `0`. The exit code marks proven unfitness, not
the absence of doubt, and a pipeline that needs to stop on doubt should read
the verdict string instead.

## 9. Detector pedestal

Printed when a pedestal is available, from `set_Offset` or from `--offset`.

| field | meaning |
|---|---|
| `offset` | the pedestal in counts, and where it came from |
| `change_counts` | Theil–Sen slope times the span, in counts |
| `pct_raw` | that change over the raw median level |
| `pct_net` | that change over the level with the pedestal removed |
| `inflation` | `pct_net / pct_raw` |

`pct_net` is the figure to quote. The pedestal is a constant the detector adds
and was never light, so it belongs in neither the numerator nor the denominator
of a change. `inflation` at or above 1.5 prints the warning; below that the
pedestal is small enough not to matter.

The audit refuses to proceed if the pedestal moved during the session. An
average pedestal describes none of the images.

## 10. Conditioning

Printed when `--raw-sum-col` and `--n-voxels` are supplied.

| field | meaning |
|---|---|
| `background_fraction` | share of the raw sum that is background, median over images |
| `amplification` | `f / (1 - f)`, the error multiplier of the subtraction |
| `background_cv_pct` | measured image-to-image scatter of the background |
| `induced_integral_cv_pct` | that scatter after amplification |
| `tolerable_background_error_pct(x)` | how well the background must be known for an integral good to x % |

Verdicts: `ILL-CONDITIONED` at 25 % induced noise or more, `FRAGILE` from 10 %,
`WELL-CONDITIONED` below. The cuts are conventions chosen to be legible, not
values derived from anything.

This is a property of the estimator, not a finding about the microscope. It is
computable before any comparison is run and it is reported whether or not drift
was detected.

## 11. Batch structure

| field | meaning |
|---|---|
| `n_mounts`, `sizes` | batches recovered from gaps in the timestamps |
| `gap_threshold_min` | interval above which a new batch starts |
| `between_rho`, `between_p`, `between_slope` | trend of the batch means across the session |
| `within_rho`, `within_p` | trend of the residuals against elapsed time inside a batch |
| `within_slope`, `within_slope_lo/hi` | the within-batch coefficient and its 95 % interval |
| `within_f`, `within_f_p` | F test for adding the within-batch term |
| `r2_time`, `r2_time_plus_within` | R² before and after adding it |

`locus` is `BETWEEN MOUNTS`, `WITHIN MOUNTS`, `BOTH` or `NEITHER RESOLVED`, and
each carries its own remedy. Between-batch drift tracks the clock and is
defeated by randomising the order; within-batch drift resets with each mount
and is not.

Read `within_slope_lo/hi` before treating a flat within-batch result as an
absence. Batches are short and sessions are long, so that test always has less
power than the between-batch one.

## 12. Threshold selection

Printed when `--count-col` is supplied.

| field | meaning |
|---|---|
| `rho`, `q` | per group, correlation of object count with image brightness |
| `over_ceiling`, `over_ceiling_frac` | images reporting more objects than `--ceiling` allows |
| `headroom_sigma` | distance from background to threshold, in background standard deviations |
| `headroom_shift_sigma` | how far the background moves across the session, in the same units |

Within a group the treatment is constant and the true count cannot change, so a
positive `rho` means the threshold is admitting a different sample rather than
the detector measuring a different value. Counts above a stated biological
ceiling are false positives by construction, with no ground truth needed.

`THRESHOLD SELECTING` makes the session `UNFIT` for every endpoint derived from
that detector, and only those. Counts made by eye and assays that never passed
through it are unaffected.

## 13. Negative control on a correction

Returned by `sensitivity.validate`, not printed by the command line.

| field | meaning |
|---|---|
| `artefact_raw_pct` | late-half against early-half change within the untreated group, before correction |
| `artefact_corrected_pct` | the same after correction |
| `reduction_pct` | how much of the artefact was removed; negative if it grew |
| `rho_raw`, `rho_corrected` | the same contrast as a trend against time |

Verdicts: `CORRECTION REJECTED` when the artefact did not shrink,
`CORRECTION INSUFFICIENT` below 50 % removed, `CORRECTION REDUCES ARTEFACT`
above it. There is no verdict meaning the correction is right.

The result is worth nothing unless `corrected_col` was recomputed from the
corrected images with the same detection settings. Rescaling numbers already
extracted from the raw images tests nothing, because the selection that did the
damage happened before the rescaling.

## 14. Acquisition record

`records.validate` reports what the table can support: `can_audit_schedule`,
`can_audit_drift`, `can_correct_offset`, the channels it found, the `set_*`
columns it will check for constancy, and a list of warnings. Print
`check.summary()` before trusting any verdict. `acqdrift --schema` prints the
field layout.
