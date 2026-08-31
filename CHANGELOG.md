# Changelog

## 0.2.1

A guard on the shape of the design, found by running Level 1 against public
depositions in the Image Data Resource.

- eta^2 is bounded below by (k-1)/(n-1): a random partition into k groups already
  explains that much of the variance in acquisition time. Fifty groups over two
  hundred images puts the floor at 0.25, so a raw eta^2 read against the fixed cuts
  called a well-interleaved session confounded. `eta_squared_null` and
  `epsilon_squared` report and remove the floor, and both are printed next to it.
- `ScheduleReport` gains `eta_null`, `epsilon_squared`, `min_group_n`,
  `bands_reliable` and `verdict_reason`. The verdict is withheld as
  `INCONCLUSIVE` when the smallest group holds fewer than 5 images or the floor
  exceeds 0.15. Both cuts are conventions chosen to be legible.
- `separable` is now false on a design the bands cannot judge, so it can no
  longer be read as a pass.
- The fitness verdict folds this in rather than falling through to
  `NO EVIDENCE OF UNFITNESS`, which is what it did before.
- The worked example is unchanged: eta^2 = 0.962 against a floor of 0.050,
  epsilon^2 = 0.960, still `CONFOUNDED`.

## 0.2.0

Four checks on whether the numbers going into the drift audit mean what they
appear to mean, and one module that fits a correction and then tries to
disprove it.

- `offset`: the detector pedestal, read per channel from the metadata and
  removed from the denominator of every percentage. On the worked example a
  drift of 8.5 counts reads as -6.0 % against the raw level and -20.1 %
  against signal, a factor of 3.4.
- `conditioning`: error amplification of a background-subtracted integral,
  `f / (1 - f)`. At a background fraction of 94.9 % the amplification is 18.7x,
  which decides whether integrated intensity is a usable endpoint at all.
- `mounts`: batch structure recovered from gaps in the timestamps, splitting
  drift into a between-batch part that randomising the order defeats and a
  within-batch part that it does not.
- `selection`: a fixed detection threshold admitting a different sample from
  bright images than from dim ones. The one failure a later correction
  provably cannot undo, because the objects that fell below the line left no
  record.
- `sensitivity`: fits a multiplicative gain curve and judges it on an
  untreated group split against itself, not on the quantity used to fit it.
  On the only session where it has been tried the correction was rejected.
- `records`: a vendor-neutral acquisition record, so any microscope can be
  audited by supplying a CSV. `acqdrift --schema` prints the layout.

The fitness verdict now folds in threshold selection and conditioning, and
still returns three outcomes with no pass.

## 0.1.0

First release.

- Level 1, acquisition schedule audit from timestamps alone: η² of acquisition
  time by group with a permutation p-value, pairwise temporal overlap, longest
  same-group streak against its random expectation, and constancy of instrument
  settings.
- Level 2, drift in raw image statistics without segmentation: Spearman
  association and Theil–Sen magnitude against acquisition time within each
  group, adjusted across the whole family by Benjamini–Hochberg.
- Sentinel validity check, which tests rather than assumes that the background
  is independent of the amount of signal in the frame.
- Combined fitness verdict with three outcomes and no pass, printed alongside
  the failure modes the audit does not cover.
- `interleaved_order` proposes a randomised acquisition schedule.
- Zeiss CZI reader; the statistical core takes a plain DataFrame, so other
  formats work by building that table yourself.
