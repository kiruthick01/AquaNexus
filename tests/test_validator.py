"""Validator tests.

The headline case is the RS 200 constriction found on the real Ayase tile: a
section extracted 1.9 m wide between neighbours of 10-13 m, which drove velocity
to 2.46 m/s where neighbours ran at 0.5. Nothing else in the pipeline noticed.
"""

import numpy as np
import pandas as pd
import pytest

from aquanexus.data.validator import (
    Report,
    validate_dataset,
    validate_observations,
    validate_sections,
)
from aquanexus.hecras.geometry import CrossSection


def section(river_station, width=40.0, depth=3.0, points=41, invert=5.0):
    """A V-shaped section of a given top width."""
    station = np.linspace(0.0, width, points)
    centre = width / 2.0
    elevation = invert + depth * (np.abs(station - centre) / centre) ** 2
    return CrossSection(river_station, station, elevation, (0.0, 0.0), (1.0, 0.0))


def reach(n=5, width=40.0):
    # Bed falls downstream: invert decreases as river station decreases.
    return [section(100.0 * (i + 1), width=width, invert=5.0 + 0.1 * i)
            for i in range(n)]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_ok_ignores_warnings():
    r = Report()
    r.add("c", "warning", "m")
    assert r.ok
    r.add("c", "error", "m")
    assert not r.ok


def test_issue_renders_location():
    r = Report()
    r.add("check", "error", "broke", "RS 200")
    assert "[RS 200]" in str(r.issues[0])
    assert "ERROR" in str(r.issues[0])


# ---------------------------------------------------------------------------
# Cross-sections
# ---------------------------------------------------------------------------


def test_clean_reach_passes():
    report = validate_sections(reach())
    assert report.ok
    assert not any(i.check == "section.constriction" for i in report.issues)


def test_constriction_is_flagged():
    """The real RS 200 defect: far narrower than both neighbours."""
    sections = reach(5)
    sections[2] = section(300.0, width=2.0, invert=5.2)
    report = validate_sections(sections)
    flagged = [i for i in report.issues if i.check == "section.constriction"]
    assert len(flagged) == 1
    assert "RS 300" in flagged[0].location


def test_uniform_narrow_reach_is_not_flagged():
    """A genuinely narrow river must not trip the neighbour comparison."""
    report = validate_sections(reach(5, width=3.0))
    assert not any(i.check == "section.constriction" for i in report.issues)


def test_empty_section_is_an_error():
    sections = reach(3)
    sections.append(CrossSection(400.0, np.empty(0), np.empty(0), (0, 0), (1, 0)))
    report = validate_sections(sections)
    assert any(i.check == "section.empty" and i.severity == "error"
               for i in report.issues)


def test_no_sections_at_all():
    report = validate_sections([])
    assert not report.ok
    assert report.issues[0].check == "sections.empty"


def test_sparse_section_warns():
    sections = reach(3)
    sections[1] = section(200.0, points=5)
    assert any(i.check == "section.sparse" for i in validate_sections(sections).issues)


def test_unsorted_stations_are_an_error():
    xs = section(100.0)
    xs.station = xs.station[::-1].copy()
    assert any(i.check == "section.station_order"
               for i in validate_sections([xs]).issues)


def test_nonfinite_elevation_is_an_error():
    xs = section(100.0)
    xs.elevation[3] = np.nan
    assert any(i.check == "section.nonfinite" for i in validate_sections([xs]).issues)


def test_invert_jump_warns():
    sections = [section(100.0, invert=5.0), section(200.0, invert=12.0)]
    assert any(i.check == "section.invert_jump"
               for i in validate_sections(sections).issues)


def test_bed_rising_downstream_warns():
    """Stationing convention mistakes show up as a reversed bed slope."""
    sections = [section(100.0 * (i + 1), invert=8.0 - 0.5 * i) for i in range(4)]
    assert any(i.check == "reach.slope_sign"
               for i in validate_sections(sections).issues)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@pytest.fixture
def observations():
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-04-01", periods=12, freq="30D"),
        "water_temp": np.linspace(8, 31, 12),
        "dissolved_oxygen": np.linspace(11, 3.5, 12),
        "ph": np.full(12, 7.4),
        "discharge": np.linspace(2, 60, 12),
        "phosphorus_total": np.full(12, 0.22),
        "phosphorus_total_flag": [np.nan] * 12,
    })


def test_clean_observations_pass(observations):
    assert validate_observations(observations).ok


def test_empty_frame_is_an_error():
    assert not validate_observations(pd.DataFrame()).ok


def test_required_column_absent(observations):
    report = validate_observations(observations, required=["velocity"])
    assert any(i.check == "observations.missing_column" for i in report.issues)


def test_impossible_value_is_an_error(observations):
    observations.loc[3, "water_temp"] = 250.0  # a unit error, not weather
    report = validate_observations(observations)
    assert any(i.check == "observations.out_of_range" and i.location == "water_temp"
               for i in report.issues)


def test_negative_oxygen_is_an_error(observations):
    observations.loc[2, "dissolved_oxygen"] = -1.0
    assert not validate_observations(observations).ok


def test_sparse_column_warns(observations):
    observations.loc[2:, "discharge"] = np.nan
    assert any(i.check == "observations.sparse"
               for i in validate_observations(observations).issues)


def test_mostly_censored_column_warns(observations):
    """A column reporting detection limits is not really measured."""
    observations["phosphorus_total_flag"] = ["<"] * 10 + [np.nan] * 2
    report = validate_observations(observations)
    assert any(i.check == "observations.mostly_censored" for i in report.issues)


def test_unparseable_timestamp_is_an_error(observations):
    observations["timestamp"] = observations["timestamp"].astype(object)
    observations.loc[0, "timestamp"] = "not a date"
    assert not validate_observations(observations).ok


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({
        "water_temp": rng.uniform(5, 32, n),
        "dissolved_oxygen": rng.uniform(3, 12, n),
        "depth": rng.uniform(0.2, 3.0, n),
        "velocity": rng.uniform(0.1, 1.5, n),
        "hsi": rng.uniform(0.05, 0.95, n),
    })


FEATURES = ["water_temp", "dissolved_oxygen", "depth", "velocity"]


def test_clean_dataset_passes(dataset):
    assert validate_dataset(dataset, FEATURES).ok


def test_too_few_rows(dataset):
    assert not validate_dataset(dataset.head(10), FEATURES).ok


def test_missing_feature_and_target(dataset):
    assert any(i.check == "dataset.missing_features"
               for i in validate_dataset(dataset, [*FEATURES, "absent"]).issues)
    assert not validate_dataset(dataset.drop(columns="hsi"), FEATURES).ok


def test_target_outside_unit_interval(dataset):
    dataset.loc[0, "hsi"] = 1.4
    assert any(i.check == "dataset.target_range"
               for i in validate_dataset(dataset, FEATURES).issues)


def test_degenerate_target(dataset):
    dataset["hsi"] = 0.5
    report = validate_dataset(dataset, FEATURES)
    assert any(i.check == "dataset.target_degenerate" for i in report.issues)


def test_constant_feature_warns(dataset):
    dataset["depth"] = 1.0
    assert any(i.check == "dataset.constant_feature"
               for i in validate_dataset(dataset, FEATURES).issues)


def test_target_leakage_is_an_error(dataset):
    """A feature that is the label in disguise inflates every metric."""
    dataset["sneaky"] = dataset["hsi"] * 1.0001
    report = validate_dataset(dataset, [*FEATURES, "sneaky"])
    assert any(i.check == "dataset.target_leakage" for i in report.issues)
    assert not report.ok
