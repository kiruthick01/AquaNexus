"""Tests for splits, models, evaluation and the state-vector builder.

The split tests matter most. The dataset has group structure - 53 rows share one
observation's chemistry - so an ungrouped split reports memorisation. These
assert that the grouped splits actually keep groups whole.
"""

import numpy as np
import pandas as pd
import pytest

from aquanexus.data.dataset import (
    build_state_vectors,
    feature_columns,
    interpolate_hydraulics,
)
from aquanexus.ml.evaluator import compare, evaluate, residual_summary
from aquanexus.ml.models import HabitatPredictor, MeanPredictor, ModelConfig, ModelFactory
from aquanexus.ml.splits import (
    flow_split,
    grouped_split,
    random_split,
    spatial_split,
    temporal_split,
)


@pytest.fixture
def sweep():
    """Three sections across a log-spaced discharge sweep."""
    rows = []
    for q in (0.2, 1.0, 5.0, 25.0, 100.0):
        for i, rs in enumerate((0.0, 500.0, 1000.0)):
            rows.append({
                "discharge_bc": q, "river_station": rs,
                "invert": 5.0 + 0.1 * i,
                "wse": 5.0 + 0.1 * i + 0.4 * np.log1p(q),
                "depth": 0.4 * np.log1p(q) + 0.05 * i,
                "velocity": 0.15 * np.log1p(q),
                "flow_area": 5.0 * q**0.5,
                "top_width": 10.0 + 3.0 * np.log1p(q),
                "energy_slope": 0.001,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def observations():
    n = 20
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-15", periods=n, freq="30D"),
        "station": ["A"] * n,
        "discharge": rng.uniform(0.3, 60.0, n),
        "water_temp": rng.uniform(6, 31, n),
        "dissolved_oxygen": rng.uniform(3.5, 11.0, n),
        "suspended_solids": rng.uniform(5, 40, n),
    })


@pytest.fixture
def dataset(observations, sweep):
    return build_state_vectors(observations, sweep)


# ---------------------------------------------------------------------------
# State vectors
# ---------------------------------------------------------------------------


def test_interpolation_hits_swept_values_exactly(sweep):
    got = interpolate_hydraulics(sweep, 5.0)
    expected = sweep[sweep.discharge_bc == 5.0].sort_values("river_station")
    assert np.allclose(got.sort_values("river_station").depth.to_numpy(),
                       expected.depth.to_numpy())


def test_interpolation_is_monotonic_between_points(sweep):
    depths = [interpolate_hydraulics(sweep, q).depth.mean() for q in (0.5, 2, 10, 50)]
    assert depths == sorted(depths)


def test_interpolation_clamps_rather_than_extrapolating(sweep):
    """Extrapolating hydraulics beyond the swept range invents physics."""
    below = interpolate_hydraulics(sweep, 0.001)
    at_min = interpolate_hydraulics(sweep, 0.2)
    assert np.allclose(below.depth.to_numpy(), at_min.depth.to_numpy())

    above = interpolate_hydraulics(sweep, 10_000.0)
    at_max = interpolate_hydraulics(sweep, 100.0)
    assert np.allclose(above.depth.to_numpy(), at_max.depth.to_numpy())


def test_state_vectors_shape_and_grouping(dataset, observations, sweep):
    assert len(dataset) == len(observations) * sweep.river_station.nunique()
    assert dataset["group"].nunique() == len(observations)
    # Chemistry is constant within a group; hydraulics are not.
    first = dataset[dataset.group == 0]
    assert first["water_temp"].nunique() == 1
    assert first["depth"].nunique() > 1


def test_state_vectors_carry_label_and_features(dataset):
    assert "hsi" in dataset.columns
    assert dataset["hsi"].between(0, 1).all()
    for column in ("do_deficit", "froude_number", "shear_stress"):
        assert column in dataset.columns


def test_observations_without_discharge_are_skipped(observations, sweep):
    observations.loc[0, "discharge"] = np.nan
    built = build_state_vectors(observations, sweep)
    assert built["group"].nunique() == len(observations) - 1


def test_missing_discharge_column_raises(observations, sweep):
    with pytest.raises(ValueError, match="discharge"):
        build_state_vectors(observations.drop(columns="discharge"), sweep)


def test_feature_columns_exclude_label_components(dataset):
    features = feature_columns(dataset)
    for leaked in ("hsi", "group", "river_station", "thermal_stress_index",
                   "oxygen_stress_index", "combined_stress_index"):
        assert leaked not in features
    assert "depth" in features and "dissolved_oxygen" in features


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_grouped_split_keeps_groups_whole(dataset):
    split = grouped_split(dataset)
    train = set(dataset.loc[split.train, "group"])
    test = set(dataset.loc[split.test, "group"])
    validation = set(dataset.loc[split.validation, "group"])
    assert not (train & test) and not (train & validation) and not (test & validation)


def test_grouped_split_covers_every_row(dataset):
    split = grouped_split(dataset)
    total = split.train.sum() + split.validation.sum() + split.test.sum()
    assert total == len(dataset)


def test_spatial_split_holds_out_whole_sections(dataset):
    split = spatial_split(dataset, test_fraction=0.34, validation_fraction=0.0)
    train = set(dataset.loc[split.train, "river_station"])
    test = set(dataset.loc[split.test, "river_station"])
    assert not (train & test)
    assert len(test) >= 1


def test_flow_split_tests_the_extremes(dataset):
    split = flow_split(dataset)
    train_q = dataset.loc[split.train, "discharge"]
    test_q = dataset.loc[split.test, "discharge"]
    # Test discharges lie outside the training interior on at least one side.
    assert test_q.min() <= train_q.min() or test_q.max() >= train_q.max()


def test_temporal_split_puts_later_dates_in_test(dataset):
    split = temporal_split(dataset)
    train_t = pd.to_datetime(dataset.loc[split.train, "timestamp"])
    test_t = pd.to_datetime(dataset.loc[split.test, "timestamp"])
    assert train_t.max() < test_t.min()


def test_random_split_leaks_groups(dataset):
    """The leak this split exists to demonstrate."""
    split = random_split(dataset)
    train = set(dataset.loc[split.train, "group"])
    test = set(dataset.loc[split.test, "group"])
    assert train & test, "an ungrouped split should straddle groups"


def test_split_apply_returns_matching_shapes(dataset):
    features = feature_columns(dataset)
    split = grouped_split(dataset)
    (Xtr, ytr), (Xv, yv), (Xte, yte) = split.apply(dataset, features)
    assert Xtr.shape[1] == len(features)
    assert len(Xtr) == len(ytr) and len(Xte) == len(yte)


def test_leaking_split_is_rejected(dataset):
    from aquanexus.ml.splits import Split, _check_disjoint

    everything = pd.Series(True, index=dataset.index)
    bad = Split("bad", everything, ~everything, everything)
    with pytest.raises(ValueError, match="leaking"):
        _check_disjoint(dataset, bad, "group")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_factory_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown model type"):
        ModelFactory.create(ModelConfig(model_type="magic"))


def test_mean_predictor_is_the_floor():
    model = MeanPredictor().fit(np.zeros((5, 2)), [0.2, 0.4, 0.6, 0.8, 1.0])
    assert model.predict(np.zeros((3, 2))).tolist() == [0.6, 0.6, 0.6]


@pytest.mark.parametrize("model_type", ["xgboost", "random_forest", "linear", "mean"])
def test_models_fit_and_predict_in_range(dataset, model_type):
    features = feature_columns(dataset)
    split = grouped_split(dataset)
    (Xtr, ytr), _, (Xte, _) = split.apply(dataset, features)
    predictor = HabitatPredictor(ModelConfig(model_type=model_type)).fit(Xtr, ytr)
    predictions = predictor.predict(Xte)
    assert len(predictions) == len(Xte)
    assert ((predictions >= 0.0) & (predictions <= 1.0)).all()


def test_linear_model_tolerates_missing_values(dataset):
    """Ridge rejects NaN natively; the pipeline must impute inside the fit."""
    features = feature_columns(dataset)
    corrupted = dataset.copy()
    corrupted.loc[corrupted.index[:50], features[0]] = np.nan
    split = grouped_split(corrupted)
    (Xtr, ytr), _, (Xte, _) = split.apply(corrupted, features)
    predictor = HabitatPredictor(ModelConfig(model_type="linear")).fit(Xtr, ytr)
    assert np.isfinite(predictor.predict(Xte)).all()


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        HabitatPredictor().predict(pd.DataFrame({"a": [1.0]}))


def test_importances_available_for_every_model(dataset):
    features = feature_columns(dataset)
    split = grouped_split(dataset)
    (Xtr, ytr), _, _ = split.apply(dataset, features)
    for model_type in ("xgboost", "random_forest", "linear"):
        predictor = HabitatPredictor(ModelConfig(model_type=model_type)).fit(Xtr, ytr)
        importances = predictor.importances()
        assert len(importances) == len(features), model_type


def test_uncertainty_is_nan_where_unsupported(dataset):
    features = feature_columns(dataset)
    split = grouped_split(dataset)
    (Xtr, ytr), _, (Xte, _) = split.apply(dataset, features)

    forest = HabitatPredictor(ModelConfig(model_type="random_forest")).fit(Xtr, ytr)
    _, spread = forest.predict_with_uncertainty(Xte)
    assert np.isfinite(spread).all() and (spread >= 0).all()

    linear = HabitatPredictor(ModelConfig(model_type="linear")).fit(Xtr, ytr)
    _, none = linear.predict_with_uncertainty(Xte)
    assert np.isnan(none).all()


def test_save_and_load_roundtrip(dataset, tmp_path):
    features = feature_columns(dataset)
    split = grouped_split(dataset)
    (Xtr, ytr), _, (Xte, _) = split.apply(dataset, features)
    predictor = HabitatPredictor(ModelConfig(model_type="random_forest")).fit(Xtr, ytr)
    before = predictor.predict(Xte)

    path = predictor.save(tmp_path / "m.joblib")
    reloaded = HabitatPredictor.load(path)
    assert reloaded.feature_names == predictor.feature_names
    assert np.allclose(reloaded.predict(Xte), before)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_perfect_prediction():
    y = np.linspace(0, 1, 50)
    m = evaluate(y, y, "perfect", "s", baseline=float(y.mean()))
    assert m.rmse == pytest.approx(0.0)
    assert m.r2 == pytest.approx(1.0)
    assert m.skill == pytest.approx(1.0)


def test_mean_prediction_scores_zero_skill():
    y = np.linspace(0, 1, 50)
    mean = float(y.mean())
    m = evaluate(y, np.full_like(y, mean), "mean", "s", baseline=mean)
    assert m.skill == pytest.approx(0.0)
    assert m.r2 == pytest.approx(0.0, abs=1e-9)


def test_bias_is_signed():
    y = np.full(10, 0.5)
    assert evaluate(y, y + 0.1).bias == pytest.approx(0.1)
    assert evaluate(y, y - 0.1).bias == pytest.approx(-0.1)


def test_evaluate_ignores_non_finite_pairs():
    y = np.array([0.1, 0.2, np.nan, 0.4])
    p = np.array([0.1, 0.2, 0.3, np.inf])
    assert evaluate(y, p).n == 2


def test_evaluate_requires_finite_data():
    with pytest.raises(ValueError, match="no finite pairs"):
        evaluate([np.nan], [np.nan])


def test_compare_orders_by_rmse():
    rows = [evaluate([0.1, 0.9], [0.5, 0.5], "bad", "s"),
            evaluate([0.1, 0.9], [0.1, 0.9], "good", "s")]
    assert compare(rows).iloc[0]["model"] == "good"


def test_residual_summary_exposes_extremes():
    y = np.linspace(0, 1, 200)
    pred = np.clip(y, 0.2, 0.8)  # flattened at both ends
    summary = residual_summary(y, pred)
    assert summary.iloc[0]["bias"] > 0    # over-predicts the low tail
    assert summary.iloc[-1]["bias"] < 0   # under-predicts the high tail
