"""Phase 2c validation tests.

Persistence is the baseline that matters. A model that cannot beat "same as last
sample" has not earned its complexity, and it is the baseline most easily left
out of a comparison table.
"""

import numpy as np
import pandas as pd
import pytest

from aquanexus.ml.models import ModelConfig
from aquanexus.ml.validator import (
    ModelValidator,
    hydraulic_only_features,
    persistence_baseline,
)


@pytest.fixture
def frame():
    """Four stations, monthly samples, a genuine temperature-driven signal."""
    rng = np.random.default_rng(3)
    rows = []
    for station in ("A", "B", "C", "D"):
        offset = {"A": 0.0, "B": 0.7, "C": -0.5, "D": 0.2}[station]
        for i in range(24):
            temp = 8 + 12 * (1 + np.sin(2 * np.pi * i / 12))
            discharge = float(rng.uniform(0.5, 50))
            rows.append({
                "station": station,
                "timestamp": pd.Timestamp("2022-04-01") + pd.DateOffset(months=i),
                "water_temp": temp,
                "discharge": discharge,
                "reach_depth": 0.4 * np.log1p(discharge),
                "reach_velocity": 0.1 * np.log1p(discharge),
                "dissolved_oxygen": 13.5 - 0.22 * temp + offset + rng.normal(0, 0.4),
            })
    return pd.DataFrame(rows)


FEATURES = ["water_temp", "discharge", "reach_depth", "reach_velocity"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persistence_uses_the_previous_sample_at_the_same_station(frame):
    lagged = persistence_baseline(frame, "dissolved_oxygen")
    ordered = frame.sort_values(["station", "timestamp"])
    first_a = ordered[ordered.station == "A"].index[0]
    second_a = ordered[ordered.station == "A"].index[1]
    assert np.isnan(lagged[first_a])
    assert lagged[second_a] == frame.loc[first_a, "dissolved_oxygen"]


def test_persistence_does_not_cross_stations(frame):
    """Each station's first sample has no predecessor and must stay NaN."""
    lagged = persistence_baseline(frame, "dissolved_oxygen")
    assert int(lagged.isna().sum()) == frame.station.nunique()


def test_persistence_preserves_the_original_index(frame):
    lagged = persistence_baseline(frame.sample(frac=1, random_state=0),
                                  "dissolved_oxygen")
    assert set(lagged.index) == set(frame.index)


# ---------------------------------------------------------------------------
# Feature subsets
# ---------------------------------------------------------------------------


def test_hydraulic_subset_excludes_chemistry():
    subset = hydraulic_only_features(
        ["water_temp", "discharge", "reach_depth", "dissolved_oxygen", "bod"]
    )
    assert "discharge" in subset and "reach_depth" in subset
    assert "water_temp" not in subset and "bod" not in subset


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_cross_val_predict_covers_every_row(frame):
    validator = ModelValidator(frame, FEATURES, "dissolved_oxygen")
    predictions = validator.cross_val_predict(
        ModelConfig(model_type="linear", output_range=None)
    )
    assert len(predictions) == len(frame)
    assert np.isfinite(predictions).all()


def test_cross_validation_holds_whole_stations_out(frame, monkeypatch):
    """Every fold's test rows must come from stations absent in training."""
    from sklearn.model_selection import GroupKFold

    seen = []
    splitter = GroupKFold(n_splits=frame.station.nunique())
    for train_idx, test_idx in splitter.split(frame[FEATURES],
                                              frame["dissolved_oxygen"],
                                              frame["station"]):
        train_stations = set(frame.iloc[train_idx]["station"])
        test_stations = set(frame.iloc[test_idx]["station"])
        seen.append(train_stations & test_stations)
    assert all(not overlap for overlap in seen)


def test_single_group_cannot_be_cross_validated(frame):
    one = frame[frame.station == "A"]
    validator = ModelValidator(one, FEATURES, "dissolved_oxygen")
    with pytest.raises(ValueError, match="at least two groups"):
        validator.cross_val_predict(ModelConfig(model_type="linear"))


def test_absent_features_are_skipped_with_a_warning(frame):
    validator = ModelValidator(frame, [*FEATURES, "nonexistent"], "dissolved_oxygen")
    assert "nonexistent" not in validator.features


def test_report_includes_all_three_specified_baselines(frame):
    report = ModelValidator(frame, FEATURES, "dissolved_oxygen").run(
        model_types=("linear",), output_range=None
    )
    models = set(report.table()["model"])
    assert {"linear", "persistence", "mean", "hydraulic-only"} <= models


def test_best_excludes_the_baselines(frame):
    report = ModelValidator(frame, FEATURES, "dissolved_oxygen").run(
        model_types=("linear",), output_range=None
    )
    assert report.best().model not in {"mean", "persistence"}


def test_a_real_signal_beats_the_mean(frame):
    report = ModelValidator(frame, FEATURES, "dissolved_oxygen").run(
        model_types=("linear",), output_range=None
    )
    table = report.table().set_index("model")
    assert table.loc["linear", "rmse"] < table.loc["mean", "rmse"]


def test_hydraulic_only_underperforms_when_the_driver_is_thermal(frame):
    """DO here is temperature-driven, so hydraulics alone should do poorly."""
    report = ModelValidator(frame, FEATURES, "dissolved_oxygen").run(
        model_types=("linear",), output_range=None
    )
    table = report.table().set_index("model")
    assert table.loc["hydraulic-only", "rmse"] > table.loc["linear", "rmse"]


# ---------------------------------------------------------------------------
# Generalisation breakdowns
# ---------------------------------------------------------------------------


def test_by_group_scores_each_station(frame):
    validator = ModelValidator(frame, FEATURES, "dissolved_oxygen")
    table = validator.by_group(ModelConfig(model_type="linear", output_range=None))
    assert len(table) == frame.station.nunique()
    assert {"rmse", "bias", "n"} <= set(table.columns)
    assert table["rmse"].is_monotonic_decreasing  # sorted worst-first


def test_by_condition_splits_into_three_bands(frame):
    validator = ModelValidator(frame, FEATURES, "dissolved_oxygen")
    table = validator.by_condition(
        ModelConfig(model_type="linear", output_range=None), "discharge"
    )
    assert list(table["band"]) == ["low tail", "middle", "high tail"]
    assert table["n"].sum() == len(frame)


def test_by_condition_bands_are_ordered_by_the_driver(frame):
    validator = ModelValidator(frame, FEATURES, "dissolved_oxygen")
    table = validator.by_condition(
        ModelConfig(model_type="linear", output_range=None), "discharge"
    )
    means = table["discharge_mean"].tolist()
    assert means == sorted(means)


def test_report_renders(frame):
    report = ModelValidator(frame, FEATURES, "dissolved_oxygen").run(
        model_types=("linear",), output_range=None
    )
    text = str(report)
    assert "dissolved_oxygen" in text and "persistence" in text
