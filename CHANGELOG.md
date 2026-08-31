# Changelog

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
- Detector offset removed from the denominator of every percentage, read per
  channel from the file metadata.
- `interleaved_order` proposes a randomised acquisition schedule.
- Zeiss CZI reader; the statistical core takes a plain DataFrame, so other
  formats work by building that table yourself.
