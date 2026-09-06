"""Explainer tests.

Two things matter most here, and both are about *not* reporting confident
nonsense:

* The explainer must dispatch on model type. ML_STRATEGY §7.2 assumes XGBoost and
  specifies TreeExplainer; the model actually shipped for the observed dissolved
  oxygen target is a Ridge pipeline, which TreeExplainer cannot handle.
* Collinear features share SHAP credit arbitrarily, and a two-way interaction
  between two collinear features has empty cells. Returning a bare NaN there
  invites reading it as "no interaction", so it must be flagged.
"""

import numpy as np
import pandas as pd
import pytest

from aquanexus.ml.explainer import HabitatExplainer
from aquanexus.ml.models import HabitatPredictor, ModelConfig


@pytest.fixture
def data():
    """Two independent drivers, one collinear copy, one irrelevant column."""
    rng = np.random.default_rng(0)
    n = 120
    temp = rng.uniform(5, 32, n)
    flow = rng.uniform(0.5, 60, n)
    frame = pd.DataFrame({
        "temp": temp,
        "flow": flow,
        "flow_copy": flow * 2.0 + rng.normal(0, 0.01, n),  # collinear with flow
        "noise": rng.normal(0, 1, n),
    })
    frame["target"] = 12.0 - 0.22 * temp - 0.03 * flow + rng.normal(0, 0.3, n)
    return frame


FEATURES = ["temp", "flow", "flow_copy", "noise"]


def fit(data, model_type="random_forest"):
    predictor = HabitatPredictor(
        ModelConfig(model_type=model_type, n_estimators=40, max_depth=4,
                    output_range=None)
    ).fit(data[FEATURES], data["target"])
    return predictor


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def test_unfitted_model_is_rejected():
    with pytest.raises(RuntimeError, match="must be fitted"):
        HabitatExplainer(HabitatPredictor())


def test_compute_before_fit_explainer_raises(data):
    explainer = HabitatExplainer(fit(data))
    with pytest.raises(RuntimeError, match="fit_explainer"):
        explainer.compute(data[FEATURES])


def test_importance_before_compute_raises(data):
    explainer = HabitatExplainer(fit(data)).fit_explainer(data[FEATURES])
    with pytest.raises(RuntimeError, match="compute"):
        explainer.feature_importance()


@pytest.mark.parametrize("model_type", ["random_forest", "xgboost"])
def test_tree_models_use_tree_explainer(data, model_type):
    import shap

    explainer = HabitatExplainer(fit(data, model_type)).fit_explainer(data[FEATURES])
    assert isinstance(explainer.explainer, shap.TreeExplainer)


def test_linear_pipeline_does_not_use_tree_explainer(data):
    """The regression this guards: TreeExplainer cannot explain a Ridge pipeline."""
    import shap

    explainer = HabitatExplainer(fit(data, "linear")).fit_explainer(data[FEATURES])
    assert not isinstance(explainer.explainer, shap.TreeExplainer)


# ---------------------------------------------------------------------------
# Values and importance
# ---------------------------------------------------------------------------


def test_shap_values_shape(data):
    explainer = HabitatExplainer(fit(data)).fit_explainer(data[FEATURES])
    values = explainer.compute(data[FEATURES])
    assert values.shape == (len(data), len(FEATURES))


def test_importance_ranks_real_drivers_above_noise(data):
    explainer = HabitatExplainer(fit(data)).fit_explainer(data[FEATURES])
    explainer.compute(data[FEATURES])
    importance = explainer.feature_importance().set_index("feature")["mean_abs_shap"]
    assert importance["temp"] > importance["noise"]


def test_importance_reports_signed_and_absolute(data):
    explainer = HabitatExplainer(fit(data)).fit_explainer(data[FEATURES])
    explainer.compute(data[FEATURES])
    table = explainer.feature_importance()
    assert set(table.columns) == {"feature", "mean_abs_shap", "mean_shap"}
    assert (table["mean_abs_shap"] >= table["mean_shap"].abs() - 1e-9).all()


def test_collinearity_finds_the_duplicated_feature(data):
    explainer = HabitatExplainer(fit(data)).fit_explainer(data[FEATURES])
    explainer.compute(data[FEATURES])
    pairs = explainer.collinearity()
    found = {frozenset((row.feature_a, row.feature_b)) for row in pairs.itertuples()}
    assert frozenset(("flow", "flow_copy")) in found


def test_collinearity_threshold_is_respected(data):
    """Only pairs at or beyond the threshold are reported."""
    explainer = HabitatExplainer(fit(data)).fit_explainer(data[FEATURES])
    explainer.compute(data[FEATURES])
    assert not explainer.collinearity(threshold=0.9).empty
    # flow/flow_copy correlate ~1.0, so nothing clears a threshold above it.
    assert explainer.collinearity(threshold=1.01).empty


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------


def test_explain_prediction_matches_the_model(data):
    predictor = fit(data)
    explainer = HabitatExplainer(predictor).fit_explainer(data[FEATURES])
    explainer.compute(data[FEATURES])
    row = data[FEATURES].iloc[[3]]
    explanation = explainer.explain_prediction(row)
    assert explanation.prediction == pytest.approx(predictor.predict(row)[0])
    assert len(explanation.top_features) == min(5, len(FEATURES))
    # Ordered by absolute contribution.
    magnitudes = [abs(v) for _, v in explanation.top_features]
    assert magnitudes == sorted(magnitudes, reverse=True)


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------


def test_interaction_between_independent_features_is_identifiable(data):
    explainer = HabitatExplainer(fit(data)).fit_explainer(data[FEATURES])
    explainer.compute(data[FEATURES])
    result = explainer.interaction_analysis("temp", "flow")
    assert result["identifiable"]
    assert all(count > 0 for count in result["counts"].values())
    assert np.isfinite(result["interaction"])


def test_collinear_pair_is_flagged_not_silently_nan(data):
    """Empty corners mean the design cannot separate the two effects."""
    explainer = HabitatExplainer(fit(data)).fit_explainer(data[FEATURES])
    explainer.compute(data[FEATURES])
    result = explainer.interaction_analysis("flow", "flow_copy")
    assert not result["identifiable"]
    assert "reason" in result and "unidentifiable" in result["reason"]
    assert np.isnan(result["interaction"])


def test_interaction_reports_correlation_and_counts(data):
    explainer = HabitatExplainer(fit(data)).fit_explainer(data[FEATURES])
    explainer.compute(data[FEATURES])
    result = explainer.interaction_analysis("temp", "flow")
    assert sum(result["counts"].values()) == len(data)
    assert -1.0 <= result["correlation"] <= 1.0


def test_unknown_feature_raises(data):
    explainer = HabitatExplainer(fit(data)).fit_explainer(data[FEATURES])
    explainer.compute(data[FEATURES])
    with pytest.raises(KeyError):
        explainer.interaction_analysis("temp", "absent")


# ---------------------------------------------------------------------------
# Threshold discovery
# ---------------------------------------------------------------------------


def test_threshold_sweep_shape_and_range(data):
    explainer = HabitatExplainer(fit(data))
    result = explainer.threshold_discovery(data, "temp", steps=25)
    assert len(result["grid"]) == 25 == len(result["prediction"])
    low, high = data["temp"].quantile([0.01, 0.99])
    assert result["grid"][0] == pytest.approx(low)
    assert result["grid"][-1] == pytest.approx(high)


def test_influential_feature_moves_the_prediction(data):
    explainer = HabitatExplainer(fit(data))
    driver = explainer.threshold_discovery(data, "temp")
    irrelevant = explainer.threshold_discovery(data, "noise")
    assert driver["response_span"] > irrelevant["response_span"]
    assert driver["is_influential"]


def test_threshold_rejects_unknown_feature(data):
    explainer = HabitatExplainer(fit(data))
    with pytest.raises(KeyError, match="not a model feature"):
        explainer.threshold_discovery(data, "absent")


def test_threshold_rejects_constant_feature(data):
    constant = data.copy()
    constant["noise"] = 1.0
    explainer = HabitatExplainer(fit(constant))
    with pytest.raises(ValueError, match="no usable range"):
        explainer.threshold_discovery(constant, "noise")


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------


def test_shap_values_are_saved(data, tmp_path):
    explainer = HabitatExplainer(fit(data)).fit_explainer(data[FEATURES])
    explainer.compute(data[FEATURES])
    path = explainer.save_values(tmp_path / "shap.npy")
    assert np.load(path).shape == (len(data), len(FEATURES))


def test_summary_plot_is_written(data, tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    explainer = HabitatExplainer(fit(data)).fit_explainer(data[FEATURES])
    explainer.compute(data[FEATURES])
    explainer.plot_summary(save_path=tmp_path / "summary.png")
    assert (tmp_path / "summary.png").stat().st_size > 0
