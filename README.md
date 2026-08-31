# acqdrift

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22206711.svg)](https://doi.org/10.5281/zenodo.22206711)

Pre-flight quality control for grouped imaging sessions.

Two checks that run before any measurement is interpreted, and that answer one
question: can a difference between groups in this dataset still be attributed
to the treatment?

**Level 1 — acquisition schedule.** Reads timestamps and group labels. No pixel
data. Reports how much of the variance in acquisition time is explained by
group, whether any two groups overlap in time, and whether instrument settings
changed mid-session. Runs in about a second.

**Level 2 — drift in raw statistics.** Reads pixels, but performs no
segmentation and applies no threshold. Tests background level, percentiles and
saturation against acquisition time *within* each group. Treatment is constant
inside a group, so a trend there is not the treatment effect.

Level 2 rests on an assumption, and the package tests it rather than making it.
See *Is the sentinel valid here?* below.

Around those two sit four checks on whether the metrics going into Level 2 mean
what they appear to mean — the **detector pedestal**, the **conditioning** of a
background-subtracted integral, the **batch structure** recovered from the
timestamps, and **threshold selection** — and one module that fits a correction
and then tries to disprove it against an untreated group. Each has its own
section below.

## Why the endpoint is the wrong sentinel

The metric a study is designed around is usually a poor detector of the drift
that threatens it. In the worked example below, the same session drift is
visible in the raw background at ρ = −0.76 and in the segmented cell area at
ρ = +0.39. Detecting it through the background needs about 11 images. Detecting
it through the area needs about 49, and in that dataset the area never survived
correction for multiple testing while the background did.

The background is also the cheaper measurement, and it cannot be moved by the
biology under test, which is what makes it usable as a control.

## The result is asymmetric, and the report says so

A confounded design together with measured drift is a proof of unfitness. The
absence of a finding is not a proof of fitness. The audit therefore returns one
of three things and never a pass:

| verdict | meaning |
|---|---|
| `UNFIT` | the files cannot support a comparison between groups, and no reanalysis changes that |
| `INCONCLUSIVE` | the audit could not establish either way |
| `NO EVIDENCE OF UNFITNESS` | the checks that were run did not fire |

Every verdict prints the failure modes it did not examine: images discarded
before the audit saw them, segmentation bias, drift that is not monotone in
time, biological confounds that track acquisition order, and observer effects.

## Is the sentinel valid here?

Using the median voxel value as a background sentinel only works where the
labelling is sparse enough that the median sits in the background population,
so the biology under test cannot move it. In a densely labelled preparation the
median is signal, it responds to the treatment, and Level 2 becomes circular.

`check_sentinel` tests this on the actual data. Within each group it asks
whether the sentinel moves with the amount of bright material in the frame,
after removing any shared trend with acquisition time. A positive dependence
means the sentinel is not a control, and the Level 2 verdict is withheld rather
than reported.

Two things the check refuses to guess at. The signal proxy must sit far enough
into the bright tail to be signal at all; where labelling is very sparse even a
p99 is still background, and the test is declared untestable instead of
correlating background against background. And a sentinel that is a perfect
monotone function of time leaves no residual variation to test, which is
reported as a limit rather than as a clean result.

On the worked example the proxy sits 5.3 to 6.5 times above the sentinel in
every group and no group shows a dependence, so the assumption holds there.

## Reading the numbers

Most of what the audit prints is self-explanatory and documented field by field
in [OUTPUT.md](OUTPUT.md). Six values are not, in the sense that reading them
the obvious way gives the wrong answer.

**η² is not a correlation.** It is the fraction of the spread in acquisition
times that knowing the group already accounts for. At 0 the groups are spread
evenly through the session. At 1 the group tells you exactly when an image was
taken. Unlike a correlation between time and a group index it does not depend
on how the groups are ordered, which matters because that ordering is arbitrary.
Above 0.90 the audit calls the design confounded.

**eta^2 has a floor, and the floor rises with the number of groups.** Assigning n
images to k groups at random already explains (k-1)/(n-1) of the variance in
acquisition time. Five groups over eighty images puts that at 0.05, which is
ignorable. Fifty groups over two hundred puts it at 0.25, and reading a raw eta^2
against the cuts above would then call a well-interleaved session confounded.
The audit prints the floor next to eta^2, and withholds the verdict entirely —
`INCONCLUSIVE`, with the reason — when the smallest group holds fewer than five
images or the floor exceeds 0.15. This came out of running Level 1 against
public depositions where the column supplied as `group` turned out to be a
replicate identifier with fifty levels.

**q is not p.** Every drift table applies Benjamini–Hochberg across the whole
family of group × metric tests, which in the worked example is 70 of them. Read
q as the share of flagged rows you should expect to be false, not as the
probability that a particular row is a fluke. A row with p = 0.03 and q = 0.41
is not a finding.

**%/span is not a rate.** It is the change across the window each group was
actually acquired in, and it is never extrapolated past it. A group spanning
29 minutes is not reported as a percentage over the 428-minute session, because
the slope was never observed over that range. `pct_per_hour` is available in
the DataFrame for comparison across groups, with an `extrapolation` column
recording how far it reaches beyond the data.

**separation below 1.5 invalidates the sentinel test.** It is the ratio of the
signal proxy to the sentinel. Where labelling is very sparse even a p99 sits in
the background, and the check would be correlating background against
background. The audit declares this untestable rather than passing it.

**min_detectable_rho is what makes a null result mean anything.** It is the
smallest association the group's sample size could have found. A
non-significant row next to a floor of 0.79 says the session was too small, not
that the instrument was steady. Roughly: 0.79 at n = 10, 0.67 at n = 15, 0.49
at n = 30.

**There are three verdicts and none of them is a pass.** `UNFIT` is a positive
finding of unfitness. `INCONCLUSIVE` means the audit could not establish
either way. `NO EVIDENCE OF UNFITNESS` records that the checks which were run
did not fire, over the failure modes they cover, and nothing more.

## Install

```bash
pip install acqdrift          # core
pip install "acqdrift[czi]"   # plus the Zeiss CZI reader
```

## Use

```bash
acqdrift /path/to/session --schedule-only          # metadata, ~1 s
acqdrift /path/to/session \
    --signal-channel ZsGr1 --reference-channel ESID \
    --offset 100 --count-col n_objects --ceiling 6

acqdrift --schema                                  # the record layout
acqdrift --from-csv my_session.csv                 # any other microscope
```

`--offset` is the one flag worth not forgetting. Without it the drift
percentages use the raw level as the denominator and are understated, usually
several-fold.

Or from Python:

```python
from acqdrift import read_session, audit_schedule, analyse, render_schedule

table = read_session("session/", group_from_name=my_parser)
print(render_schedule(audit_schedule(table)))
```

Exit status is 0 when the design is separable and no drift was found, 1
otherwise, so it can gate a pipeline.

## A reference channel makes the result interpretable

If a treatment-independent channel is recorded alongside the signal —
transmitted light, a second fluorophore that should not respond, a bead field —
`compare_channels` contrasts the two. Both see the same specimen through the
same mount at the same moment, so:

| signal | reference | reading |
|---|---|---|
| drifts | flat | the fluorescence path: excitation, detection, photophysics |
| drifts | drifts | the specimen or the mount |
| flat | drifts | unusual; suspect focus or stage |

This narrows the search. It does not identify a cause.

## Four checks on the numbers going into Level 2

Level 2 compares metrics against time. Four things decide whether those metrics
mean what they appear to mean, and each fails independently of the others.

### The detector pedestal

Every detector adds a constant to the digitised value so that noise below zero
is not clipped. It is harmless in a difference and poison in a ratio, and a
drift expressed as a percentage is a ratio. In the worked example the
background falls 8.5 counts over the session:

```
as a share of the raw level (142.0)    -6.0 %
as a share of the net level  (42.0)   -20.1 %
```

Same photons, a factor of 3.4 between the two readings. The first gets a
session waved through. `audit_offset` reports both and names the source of the
pedestal, refusing to guess when the metadata says it changed mid-session.

### Conditioning of the integral

Integrated intensity above background is the difference between two large,
similar numbers. Where the background is a fraction *f* of the raw sum, a
relative error in the background arrives in the integral multiplied by
*f*/(1−*f*). In the worked example the background is 94.9 % of the raw sum, so
the amplification is 18.7×: a background known to 1 % gives an integral known
to 19 %. The image-to-image scatter of the background is 2.1 %, which alone
puts 40 % of noise into the integral. `assess_integral` calls that
`ILL-CONDITIONED` and says what the background would have to be known to.

Peak height is conditioned entirely differently — the background is 2 to 3 % of
the peak — and the two are routinely confused.

### Batch structure

Sessions come in batches: mount, image a few fields, remount. The batches are
recoverable from gaps in the timestamps without anyone having recorded them,
and the recovery separates two kinds of drift that need different remedies.
Drift *between* batches tracks the clock and is defeated by randomising the
acquisition order. Drift *within* batches resets with each mount, so
interleaving spreads it evenly over the groups instead of removing it.

In the worked example `decompose` recovers 9 batches of 8 to 11 stacks from a
median interval of 3.6 minutes, and puts the drift squarely between them:

```
between batches   rho = -0.917   P = 0.0005   slope = -0.0230 per min
within batches    rho = -0.009   F = 0.27     P = 0.60
                  R^2  0.7913 -> 0.7920       95% CI [-0.037, +0.021]
```

The within-batch interval is printed next to the null because batches are short
and sessions are long: that test always has less power, and the interval here
still admits an effect the size of the global drift.

### Threshold selection

A detector applied at a fixed number of counts asks the same question of every
image, which is not fairness when the images differ in brightness. Objects near
the threshold enter the sample in the bright images and fall out of it in the
dim ones. This is worse than a bias in the measured value and a different kind
of damage: the objects that fell below the line left no record, so no rescaling
applied afterwards can recover them.

`audit_threshold` looks for it in the per-image object counts, without opening
an image. In the worked example 53 of 81 images report more than six objects,
in a preparation that contains six dopaminergic neurons in the head. Counts
above a stated biological ceiling are false positives by construction, and
`assess` treats this as `UNFIT` for every endpoint derived from that detector —
endpoints that never passed through it are unaffected.

## Correcting drift, and the control that decides whether it worked

`sensitivity` fits a multiplicative gain curve and applies it as
`(raw − offset) / k + offset`. The order is not cosmetic: the instrument adds
the pedestal after the gain stage, so the pedestal was never scaled and must
not be unscaled.

Fitting is the easy part and it is not evidence. On the worked example the fit
is clean — A = 39.01, B = 10.68, τ = 171.9 min, k running from 1.000 to 0.803 —
and after correction the background's correlation with time collapses from
ρ = −0.861 to +0.036 while the difference between groups goes from P < 10⁻¹³ to
P = 0.111. Every printed number improves. None of it is evidence, because the
curve was fitted to the background and describing the background is what it
must do.

The test that is not circular is a group that received no treatment, split
against itself by acquisition time. Whatever separates its early images from
its late ones is artefact by construction:

```
before correction    +57.2 %    rho = +0.201
after correction     +62.1 %    rho = +0.316
VERDICT: CORRECTION REJECTED
```

The correction was arithmetically right, cleaned the background completely, and
made the artefact worse. It could not have helped: the artefact came from the
fixed threshold choosing different objects, not from the gain scaling the same
ones. The same correction manufactures detections where it amplifies most —
images with more than eight objects go from 18 to 21 of 81, and the group with
the most aggressive factor from 8.50 to 9.00 objects per image, in an animal
with six.

`scale_invariance` is worth running first. An endpoint defined as a ratio
within the same image — an area at half of each object's own peak, a ratio
between two channels, a count — moves by less than 0.5 % under the correction,
so it never needed one. Establishing that is worth more than correcting.

`validate` requires the corrected measurement to have been recomputed from the
corrected images with the same detection settings. Rescaling numbers already
extracted from the raw images tests nothing, and the report says so every time.

## Any microscope

Nothing in the audits knows what wrote the files. They consume a table with one
row per image:

```
reader  ->  acquisition record  ->  audits
```

The package ships a reader only for Zeiss CZI, the format it has been tested
against. A reader that has never been run against a real file from an
instrument is a guess about where a timestamp lives inside a proprietary
container, and a bad one.

For every other microscope, build the record yourself and load it:

```bash
acqdrift --schema                                   # the layout
acqdrift --from-csv my_session.csv --signal-channel GFP
```

`from_csv` validates the table and reports which audits it can support, so a
session with nothing but timestamps and group labels still gets Level 1 — the
cheapest check in the package and usually the one that decides the outcome. The
alternative is to write a reader: one function returning one dictionary per
file, keyed as in `acqdrift.records.SCHEMA`.

One field is both the hardest to obtain and the most important to have.
`timestamp` must be when the instrument acquired the image. File modification
time is not that — it survives a copy, a sync or an export, and it silently
reorders a session. Where only file times exist, declare
`time_source="filesystem"` and every schedule verdict is marked provisional.

## Worked example

81 confocal stacks of *C. elegans* dopaminergic neurons, five conditions, one
7-hour session on a Zeiss LSM 800. Included under `examples/celegans_day5/`,
runs from the raw files in 18 seconds.

Level 1, from timestamps alone:

```
Control       0.0 –  97.3 min      n=21
OliDA 1/5   145.8 – 223.8 min      n=19
OliDAD9     234.7 – 328.2 min      n=20
DA 140uM    351.1 – 386.8 min      n=11
DAD9 140uM  399.1 – 427.7 min      n=10

eta^2 = 0.962   overlapping group pairs: 0 of 10   longest streak: 21 (random: 3.5)
VERDICT: CONFOUNDED
```

Level 2, within the control group, where the treatment is constant:

```
ZsGr1.background   rho = -0.761   q = 0.002   *
ZsGr1.p01          rho = -0.758   q = 0.002   *
ESID.background    rho = -0.089   q = 0.948
```

The fluorescence channel drifts and the transmitted-light channel does not, in
the same animals on the same slide. Level 1 had already established that the
groups cannot be told apart from the time of day. Combined verdict: `UNFIT`.

For scale, the labelled material occupies about 10% of the field in this
preparation, well inside the range where the median is background.

This session had been analysed to completion before either check was run.

## What it cannot do

Detecting a confound is not removing one. When groups occupy separate blocks,
group and time are collinear, and adjusting for time removes the treatment
effect along with the drift. The audit reports that the files cannot support a
comparison; it does not repair them. The remedy is in the next session, through
`interleaved_order`.

A negative Level 2 result is bounded by sample size. Every non-significant
finding is reported next to the smallest effect that number of images could
have detected, and a session too small to detect anything is labelled
`UNDERPOWERED` rather than clean. As a guide, the smallest detectable |ρ| is
about 0.79 at n = 10 per group, 0.67 at n = 15, and 0.49 at n = 30.

Only Zeiss CZI has a reader so far. Every other format works by building the
acquisition record yourself and loading it with `--from-csv`; see *Any
microscope* above. The statistical core takes a plain DataFrame and has no
opinion about the instrument.

Correcting a drift is not the same as removing its consequences. `sensitivity`
will fit and apply a gain curve, and the negative control it insists on
rejected that correction on the only session where it has been tried. A
correction that passes the negative control has survived one failure mode and
no others.

It has been validated on synthetic data and on one real session. Whether the
thresholds transfer to other preparations and other microscopes is untested,
and sessions from other groups would settle it. The `ILL-CONDITIONED` cut at
25 % induced noise and the 1.5× pedestal-inflation cut are conventions chosen
to be legible, not values derived from anything.

The audit sees drift that is monotone in acquisition time. A step change, a
periodic fluctuation, or an effect that reverses within the session will be
underestimated by a rank correlation.

## Method notes

Association is Spearman's ρ; magnitude is a Theil–Sen slope with a confidence
interval, expressed as a percentage of the metric's median over the window each
group was actually acquired in and never extrapolated beyond it. p-values are
adjusted across the whole family of group × metric tests by Benjamini–Hochberg.
The confounding statistic is η² of acquisition time by group, with a
permutation p-value from shuffled labels; unlike a correlation between time and
a group index it does not depend on how the groups happen to be ordered.

`background` is the median of every voxel in the volume. Where labelling is
sparse the objects of interest occupy a small fraction of the field, so the
median tracks the background and is insensitive to how many cells are in frame.

Test coverage includes specificity on simulated clean sessions, recovery of an
injected slope, confirmation that a drift below the stated detection floor is
usually missed, and rejection of densely labelled fields by the sentinel check.
The occupancy estimator is tested for accuracy below half the field and for its
documented collapse above it, which is why it is descriptive only and never
decides a verdict.

## Citation

If this is useful in published work, please cite the repository. Authors and
affiliation are in [CITATION.cff](CITATION.cff); a DOI is minted by Zenodo for
each tagged release.

## License

MIT
