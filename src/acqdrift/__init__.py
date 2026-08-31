"""acqdrift - pre-flight quality control for grouped imaging sessions.

A sequence of checks, run before any measurement is interpreted. Each one asks
the same question from a different angle: can a difference between groups in
this dataset still be attributed to the treatment, or has the way the session
was run made that attribution impossible?

  Level 1  the acquisition schedule, from timestamps alone
  Level 2  drift in raw image statistics, without segmentation

and, around them, the four checks that decide whether the numbers going into
Level 2 mean what they appear to mean:

  pedestal      a drift percentage computed over the detector offset is
                understated, often several-fold
  conditioning  an integral above a large background amplifies the error in
                that background, sometimes past the point of usability
  batches       drift that tracks the clock and drift that resets with each
                mount need different remedies; timestamps separate them
  selection     a fixed detection threshold changes which objects enter the
                sample when the images change brightness, and no rescaling
                afterwards can undo that

`sensitivity` fits and applies a multiplicative correction, and judges it on
an untreated group split against itself rather than on the quantity used to
fit it. It is the only module that changes any pixel value, and it reports
rejection as readily as success.

Nothing here returns a pass. The best available verdict is that the checks
which were run did not fire.
"""
from .conditioning import (ConditioningReport, amplification, assess_integral,
                           assess_peak)
from .drift import analyse, compare_channels, metric_columns
from .fitness import assess
from .io import czi_metadata, image_metrics, read_session
from .mounts import MountReport, decompose, find_mounts
from .offset import (OffsetReport, add_net_columns, audit_offset,
                     constant_offset, net)
from .records import (SCHEMA, RecordCheck, describe_schema, from_csv,
                      from_records, validate)
from .report import (render_conditioning, render_drift, render_fitness,
                     render_mounts, render_negative_control, render_offset,
                     render_power, render_records, render_schedule,
                     render_selection, render_sentinel)
from .schedule import audit_schedule, interleaved_order
from .selection import SelectionReport, audit_threshold, headroom
from .sensitivity import (GainFit, NegativeControl, apply_gain, fit_gain,
                          scale_invariance, sweep_tau)
from .sensitivity import validate as validate_correction
from .sentinel import check_sentinel, occupancy, partial_spearman
from .stats import (benjamini_hochberg, eta_squared, min_detectable_rho, trend)

__version__ = "0.2.0"
__all__ = [
    # level 1 and level 2
    "audit_schedule", "interleaved_order",
    "analyse", "compare_channels", "metric_columns",
    "check_sentinel", "occupancy", "partial_spearman",
    # the four checks around them
    "OffsetReport", "audit_offset", "constant_offset", "net", "add_net_columns",
    "ConditioningReport", "amplification", "assess_integral", "assess_peak",
    "MountReport", "decompose", "find_mounts",
    "SelectionReport", "audit_threshold", "headroom",
    # correction, and the control that judges it
    "GainFit", "NegativeControl", "fit_gain", "apply_gain",
    "validate_correction", "scale_invariance", "sweep_tau",
    # records and readers
    "SCHEMA", "RecordCheck", "describe_schema", "from_csv", "from_records",
    "validate", "czi_metadata", "image_metrics", "read_session",
    # verdict and rendering
    "assess",
    "render_schedule", "render_drift", "render_power", "render_sentinel",
    "render_offset", "render_conditioning", "render_mounts",
    "render_selection", "render_negative_control", "render_records",
    "render_fitness",
    # statistics
    "benjamini_hochberg", "eta_squared", "min_detectable_rho", "trend",
]
