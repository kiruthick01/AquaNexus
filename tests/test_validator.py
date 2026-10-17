"""Validator tests for observations and the merged dataset."""

import numpy as np
import pandas as pd
import pytest

from aquanexus.data.validator import (
    Report,
    validate_dataset,
    validate_observations,
)

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
