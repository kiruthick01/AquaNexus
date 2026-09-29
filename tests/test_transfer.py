"""Tests for the Phase 6 transfer-learning methods.

Each method is checked against what it specifically claims: domain alignment
changes only the input standardisation, fine-tuning degenerates to the
source model as shrinkage grows, and the frozen-head adapter genuinely
leaves earlier layers untouched. None of these run against real Ayase/Naka
data here - that is the experiment script's job
(`scripts/phase6_transfer_learning_experiment.py`); these confirm the
mechanism is correct on small synthetic data first.
"""

import numpy as np
import pandas as pd
import pytest

from aquanexus.ml.models import HabitatPredictor, ModelConfig
from aquanexus.ml.transfer import domain_aligned_predict, fine_tune_ridge, predict_fine_tuned

torch = pytest.importorskip("torch")


@pytest.fixture
def source_model():
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({
        "x1": rng.uniform(0, 10, n),
        "x2": rng.uniform(-5, 5, n),
    })
    y = 2.0 * X["x1"] - 0.5 * X["x2"] + 3.0 + rng.normal(0, 0.3, n)
    return HabitatPredictor(ModelConfig(model_type="linear", output_range=None)).fit(X, y)


# ---------------------------------------------------------------------------
# Domain-aligned prediction
# ---------------------------------------------------------------------------


def test_domain_aligned_predict_uses_target_statistics_not_source(source_model):
    rng = np.random.default_rng(1)
    n = 40
    # Target features shifted well outside the source's training range.
    X_target = pd.DataFrame({
        "x1": rng.uniform(100, 110, n),
        "x2": rng.uniform(45, 55, n),
    })
    aligned = domain_aligned_predict(source_model, X_target)
    unaligned = source_model.predict(X_target)
    # Re-centring on the target's own mean must change the prediction
    # relative to using the source's (very different) mean/std.
    assert not np.allclose(aligned, unaligned)
    assert np.all(np.isfinite(aligned))


def test_domain_aligned_predict_matches_source_when_distributions_match():
    rng = np.random.default_rng(2)
    n = 200
    X = pd.DataFrame({"x1": rng.normal(5, 2, n)})
    y = 3.0 * X["x1"] + 1.0
    model = HabitatPredictor(ModelConfig(model_type="linear", output_range=None)).fit(X, y)
    # Predicting on data drawn from the same distribution the model was
    # trained on: target mean/std should be close to the source's own, so
    # domain alignment should barely change anything.
    X_same = pd.DataFrame({"x1": rng.normal(5, 2, n)})
    aligned = domain_aligned_predict(model, X_same)
    unaligned = model.predict(X_same)
    # Not element-wise equal - two different samples of the same distribution
    # have slightly different sample mean/std - but should track closely: high
    # correlation, and a mean difference small relative to the prediction range.
    correlation = np.corrcoef(aligned, unaligned)[0, 1]
    assert correlation > 0.99
    assert np.mean(np.abs(aligned - unaligned)) < 0.05 * (unaligned.max() - unaligned.min())


# ---------------------------------------------------------------------------
# Fine-tuned Ridge (L2-SP-style shrinkage toward the source coefficients)
# ---------------------------------------------------------------------------


def test_fine_tune_ridge_approaches_source_coefficients_as_shrinkage_grows(source_model):
    rng = np.random.default_rng(3)
    n = 20
    X_adapt = pd.DataFrame({"x1": rng.uniform(0, 10, n), "x2": rng.uniform(-5, 5, n)})
    y_adapt = rng.uniform(0, 20, n)  # unrelated target - an extreme adaptation case

    source_coef = source_model.model.named_steps["ridge"].coef_
    coefficients, _ = fine_tune_ridge(source_model, X_adapt, y_adapt, shrinkage=1e8)
    np.testing.assert_allclose(coefficients, source_coef, rtol=1e-3)


def test_fine_tune_ridge_moves_away_from_source_at_low_shrinkage(source_model):
    rng = np.random.default_rng(4)
    n = 50
    X_adapt = pd.DataFrame({"x1": rng.uniform(0, 10, n), "x2": rng.uniform(-5, 5, n)})
    y_adapt = -5.0 * X_adapt["x1"] + rng.normal(0, 0.1, n)  # opposite-signed relationship

    source_coef = source_model.model.named_steps["ridge"].coef_
    coefficients, _ = fine_tune_ridge(source_model, X_adapt, y_adapt, shrinkage=1e-6)
    assert not np.allclose(coefficients, source_coef, rtol=0.1)


def test_predict_fine_tuned_returns_finite_predictions(source_model):
    rng = np.random.default_rng(5)
    n = 15
    X_adapt = pd.DataFrame({"x1": rng.uniform(0, 10, n), "x2": rng.uniform(-5, 5, n)})
    y_adapt = 2.0 * X_adapt["x1"] - 0.5 * X_adapt["x2"] + 3.0
    coefficients, intercept = fine_tune_ridge(source_model, X_adapt, y_adapt, shrinkage=5.0)

    X_test = pd.DataFrame({"x1": rng.uniform(0, 10, 10), "x2": rng.uniform(-5, 5, 10)})
    predictions = predict_fine_tuned(source_model, X_test, coefficients, intercept)
    assert predictions.shape == (10,)
    assert np.all(np.isfinite(predictions))


# ---------------------------------------------------------------------------
# Frozen-feature-extractor + target head
# ---------------------------------------------------------------------------


def test_frozen_head_adapter_leaves_body_weights_unchanged():
    from aquanexus.ml.deep import MLPModel
    from aquanexus.ml.transfer import FrozenHeadAdapter

    rng = np.random.default_rng(6)
    n = 80
    X = pd.DataFrame({"x1": rng.uniform(0, 10, n), "x2": rng.uniform(-5, 5, n)})
    y = 2.0 * X["x1"] - X["x2"] + rng.normal(0, 0.2, n)
    source_mlp = MLPModel(hidden_sizes=(8, 4), max_epochs=50, patience=10, seed=0).fit(X, y)

    body_weights_before = [p.clone() for p in list(source_mlp.torch_module.children())[:-1]
                           [0].parameters()]

    X_adapt = pd.DataFrame({"x1": rng.uniform(0, 10, 15), "x2": rng.uniform(-5, 5, 15)})
    y_adapt = rng.uniform(-20, 20, 15)
    adapter = FrozenHeadAdapter(source_mlp, max_epochs=30, patience=10, seed=0).fit(
        X_adapt, y_adapt
    )

    body_weights_after = list(source_mlp.torch_module.children())[:-1][0].parameters()
    for before, after in zip(body_weights_before, body_weights_after, strict=True):
        torch.testing.assert_close(before, after)

    predictions = adapter.predict(X_adapt)
    assert predictions.shape == (15,)
    assert np.all(np.isfinite(predictions))


def test_frozen_head_adapter_head_weights_do_change():
    from aquanexus.ml.deep import MLPModel
    from aquanexus.ml.transfer import FrozenHeadAdapter

    rng = np.random.default_rng(7)
    n = 80
    X = pd.DataFrame({"x1": rng.uniform(0, 10, n)})
    y = 2.0 * X["x1"] + rng.normal(0, 0.2, n)
    source_mlp = MLPModel(hidden_sizes=(4,), max_epochs=50, patience=10, seed=0).fit(X, y)

    X_adapt = pd.DataFrame({"x1": rng.uniform(0, 10, 20)})
    y_adapt = -3.0 * X_adapt["x1"] + 50.0  # a very different relationship
    adapter = FrozenHeadAdapter(source_mlp, max_epochs=100, patience=30, seed=0).fit(
        X_adapt, y_adapt
    )
    assert adapter.curve.train_loss[-1] < adapter.curve.train_loss[0]


def test_frozen_head_adapter_save_load_round_trip(tmp_path):
    from aquanexus.ml.deep import MLPModel
    from aquanexus.ml.transfer import FrozenHeadAdapter

    rng = np.random.default_rng(8)
    n = 80
    X = pd.DataFrame({"x1": rng.uniform(0, 10, n)})
    y = 2.0 * X["x1"] + rng.normal(0, 0.2, n)
    source_mlp = MLPModel(hidden_sizes=(4,), max_epochs=50, patience=10, seed=0).fit(X, y)

    X_adapt = pd.DataFrame({"x1": rng.uniform(0, 10, 20)})
    y_adapt = -3.0 * X_adapt["x1"] + 50.0
    adapter = FrozenHeadAdapter(source_mlp, max_epochs=50, patience=20, seed=0).fit(
        X_adapt, y_adapt
    )
    before = adapter.predict(X_adapt)

    path = adapter.save(tmp_path / "frozen_head.joblib")
    restored = FrozenHeadAdapter.load(path)
    after = restored.predict(X_adapt)

    np.testing.assert_allclose(before, after)
