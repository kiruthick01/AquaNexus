"""Data quality checks for observations, cross-sections and the merged dataset.

Every check returns :class:`Issue` records rather than raising. A dataset with
problems is still usable - the caller decides what is disqualifying - and a
validator that stops at the first fault is useless for surveying a whole reach.

Checks are grounded in defects actually encountered on this project, not in
generic assertions:

* A cross-section extracted through a constriction or a bridge pier reports a
  channel far narrower than its neighbours. On the Ayase test tile RS 200 came
  out 1.9 m wide against 10-13 m either side, and drove velocity to 2.46 m/s
  where neighbours ran at 0.5. Nothing downstream would have flagged that.
* HEC-RAS rejects bank stations absent from the station/elevation list, so those
  are checked before a run rather than after a failed one.
* Monitoring values below the detection limit are reported *as* the limit, so an
  unusually high share of censored values in a column means its numbers are
  mostly reporting artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("data.validator")

Severity = Literal["error", "warning", "info"]


@dataclass
class Issue:
    """One quality finding."""

    check: str
    severity: Severity
    message: str
    location: str = ""
    value: float | None = None

    def __str__(self) -> str:
        where = f" [{self.location}]" if self.location else ""
        return f"{self.severity.upper():7s} {self.check}{where}: {self.message}"


@dataclass
class Report:
    """Collected findings for one validation run."""

    issues: list[Issue] = field(default_factory=list)
    checked: int = 0

    def add(self, check: str, severity: Severity, message: str,
            location: str = "", value: float | None = None) -> None:
        self.issues.append(Issue(check, severity, message, location, value))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when nothing disqualifying was found. Warnings do not disqualify."""
        return not self.errors

    def summary(self) -> str:
        return (f"{self.checked} item(s) checked: "
                f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)")

    def __str__(self) -> str:
        return "\n".join([self.summary(), *(str(i) for i in self.issues)])


# ---------------------------------------------------------------------------
# Cross-sections
# ---------------------------------------------------------------------------


def validate_sections(
    sections,
    width_ratio: float = 0.35,
    min_points: int = 10,
    max_invert_jump: float = 2.0,
) -> Report:
    """Check extracted cross-sections before they become HEC-RAS geometry.

    ``width_ratio`` flags a section narrower than this fraction of the median
    width of its immediate neighbours - the constriction case above.
    ``max_invert_jump`` flags a bed elevation step larger than this between
    adjacent sections, which usually means a section was cut at a bad angle
    rather than that the bed really steps.
    """
    report = Report(checked=len(sections))
    if not sections:
        report.add("sections.empty", "error", "no cross-sections were extracted")
        return report

    ordered = sorted(sections, key=lambda s: s.river_station)

    for xs in ordered:
        rs = f"RS {xs.river_station:.0f}"

        if xs.is_empty:
            report.add("section.empty", "error", "no station/elevation data", rs)
            continue
        if len(xs.station) < min_points:
            report.add("section.sparse", "warning",
                       f"only {len(xs.station)} points; profile may be unreliable",
                       rs, float(len(xs.station)))
        if not np.all(np.diff(xs.station) > 0):
            report.add("section.station_order", "error",
                       "stations are not strictly increasing", rs)
        if not np.isfinite(xs.elevation).all():
            report.add("section.nonfinite", "error",
                       "elevation contains NaN or inf", rs)

        # Bank stations must exist in the data or HEC-RAS refuses the run.
        from aquanexus.hecras.geometry import bank_stations

        left, right = bank_stations(xs)
        if left not in xs.station or right not in xs.station:
            report.add("section.bank_not_in_data", "error",
                       "bank station absent from station/elevation list", rs)
        if left >= right:
            report.add("section.bank_order", "error",
                       f"left bank {left:.2f} not left of right bank {right:.2f}", rs)

    # Neighbour comparisons
    widths = np.array([xs.width for xs in ordered])
    inverts = np.array([xs.thalweg for xs in ordered])

    for i, xs in enumerate(ordered):
        rs = f"RS {xs.river_station:.0f}"
        neighbours = [widths[j] for j in (i - 1, i + 1) if 0 <= j < len(ordered)]
        if neighbours and widths[i] > 0:
            reference = float(np.median(neighbours))
            if reference > 0 and widths[i] < width_ratio * reference:
                report.add(
                    "section.constriction", "warning",
                    f"width {widths[i]:.1f} m is far below neighbours "
                    f"(~{reference:.1f} m); likely cut through a structure "
                    f"or constriction",
                    rs, float(widths[i]),
                )

        if i > 0:
            jump = abs(inverts[i] - inverts[i - 1])
            if np.isfinite(jump) and jump > max_invert_jump:
                report.add("section.invert_jump", "warning",
                           f"bed steps {jump:.2f} m from the previous section",
                           rs, float(jump))

    # A reach whose bed falls in the downstream direction is the expected sign.
    finite = np.isfinite(inverts)
    if finite.sum() >= 2:
        slope = np.polyfit(
            [s.river_station for s, f in zip(ordered, finite, strict=True) if f],
            inverts[finite], 1,
        )[0]
        if slope < 0:
            report.add("reach.slope_sign", "warning",
                       "bed rises downstream; check that river stations increase "
                       "in the upstream direction", value=float(slope))

    log.info("cross-sections: %s", report.summary())
    return report


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

#: Physically possible ranges. Deliberately wider than the Ayase record - these
#: catch unit errors and corrupt rows, not unusual weather.
PLAUSIBLE_RANGES = {
    "water_temp": (-2.0, 45.0),
    "air_temp": (-30.0, 45.0),
    "dissolved_oxygen": (0.0, 25.0),
    "ph": (3.0, 11.0),
    "discharge": (0.0, 5000.0),
    "bod": (0.0, 100.0),
    "cod": (0.0, 200.0),
    "suspended_solids": (0.0, 5000.0),
    "nitrogen_total": (0.0, 50.0),
    "phosphorus_total": (0.0, 10.0),
    "depth": (0.0, 30.0),
    "velocity": (0.0, 10.0),
}


def validate_observations(
    frame: pd.DataFrame,
    required: list[str] | None = None,
    max_missing: float = 0.5,
    max_censored: float = 0.5,
) -> Report:
    """Check a loaded monitoring frame."""
    report = Report(checked=len(frame))

    if frame.empty:
        report.add("observations.empty", "error", "frame has no rows")
        return report

    for column in required or []:
        if column not in frame.columns:
            report.add("observations.missing_column", "error",
                       f"required column '{column}' is absent")

    if "timestamp" in frame.columns:
        stamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        if stamps.isna().any():
            report.add("observations.bad_timestamp", "error",
                       f"{int(stamps.isna().sum())} row(s) have no usable timestamp")
        if frame["timestamp"].duplicated().all() and len(frame) > 1:
            report.add("observations.single_time", "warning",
                       "every row shares one timestamp")

    for column, (low, high) in PLAUSIBLE_RANGES.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        outside = values.notna() & ((values < low) | (values > high))
        if outside.any():
            report.add("observations.out_of_range", "error",
                       f"{int(outside.sum())} value(s) outside the plausible "
                       f"range {low}-{high}", column, float(values[outside].iloc[0]))

        present = values.notna().mean()
        if present < (1.0 - max_missing):
            report.add("observations.sparse", "warning",
                       f"only {present:.0%} of rows carry a value", column, present)

    # A column that is mostly detection limits is not really measured.
    for column in [c for c in frame.columns if c.endswith("_flag")]:
        base = column[:-5]
        if base not in frame.columns:
            continue
        censored = frame[column].astype(str).str.startswith("<").mean()
        if censored > max_censored:
            report.add("observations.mostly_censored", "warning",
                       f"{censored:.0%} of values are below the detection limit "
                       f"and report the limit rather than a measurement",
                       base, float(censored))

    log.info("observations: %s", report.summary())
    return report


# ---------------------------------------------------------------------------
# Merged dataset
# ---------------------------------------------------------------------------


def validate_dataset(
    frame: pd.DataFrame,
    features: list[str],
    target: str = "hsi",
    min_rows: int = 100,
    max_correlation: float = 0.999,
) -> Report:
    """Check the modelling dataset immediately before training."""
    report = Report(checked=len(frame))

    if len(frame) < min_rows:
        report.add("dataset.too_small", "error",
                   f"{len(frame)} rows is below the {min_rows} minimum")

    absent = [f for f in features if f not in frame.columns]
    if absent:
        report.add("dataset.missing_features", "error",
                   f"features absent: {', '.join(absent)}")

    if target not in frame.columns:
        report.add("dataset.missing_target", "error", f"target '{target}' is absent")
        return report

    labels = pd.to_numeric(frame[target], errors="coerce")
    if labels.isna().any():
        report.add("dataset.target_nan", "error",
                   f"{int(labels.isna().sum())} row(s) have no target value")
    if labels.notna().any():
        if labels.min() < 0.0 or labels.max() > 1.0:
            report.add("dataset.target_range", "error",
                       f"target outside [0, 1]: {labels.min():.3f}..{labels.max():.3f}")
        if labels.nunique() < 10:
            report.add("dataset.target_degenerate", "error",
                       f"target takes only {labels.nunique()} distinct value(s)")
        elif labels.std() < 0.05:
            report.add("dataset.target_flat", "warning",
                       f"target varies little (sd {labels.std():.3f}); "
                       "the model has almost nothing to learn", value=float(labels.std()))

    present = [f for f in features if f in frame.columns]
    numeric = frame[present].select_dtypes(include=[np.number])

    for column in numeric.columns:
        if numeric[column].nunique() <= 1:
            report.add("dataset.constant_feature", "warning",
                       "feature is constant and carries no information", column)

    # A feature almost perfectly correlated with the target is usually leakage -
    # here, a component of the label function fed back in as an input.
    if target in frame.columns and not numeric.empty and labels.nunique() > 1:
        for column in numeric.columns:
            # A constant column has zero variance, so correlation is undefined;
            # it is already reported as constant above.
            if numeric[column].nunique() <= 1:
                continue
            correlation = numeric[column].corr(labels)
            if pd.notna(correlation) and abs(correlation) > max_correlation:
                report.add("dataset.target_leakage", "error",
                           f"correlates {correlation:.4f} with the target; "
                           "likely a component of the label function",
                           column, float(correlation))

    log.info("dataset: %s", report.summary())
    return report
