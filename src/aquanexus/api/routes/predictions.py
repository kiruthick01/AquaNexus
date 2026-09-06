"""Prediction endpoints."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from aquanexus.api.registry import registry
from aquanexus.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from aquanexus.logger import get_logger

log = get_logger("api.predictions")
router = APIRouter(tags=["predictions"])


def interpret(target: str, value: float) -> str:
    """Plain-language reading of a prediction.

    The habitat bands follow ML_STRATEGY §1; the oxygen bands follow the
    thresholds used in the label function - below 2 mg/L is acutely lethal for
    most freshwater fish, and below 5 is where hypoxic stress begins.
    """
    if target == "hsi":
        for limit, label in ((0.2, "unsuitable"), (0.4, "poor"), (0.6, "moderate"),
                             (0.8, "good"), (1.01, "optimal")):
            if value < limit:
                return f"Habitat suitability is {label} ({value:.2f})"
        return f"Habitat suitability is optimal ({value:.2f})"

    if value < 2.0:
        return f"{value:.2f} mg/L - acutely lethal for most freshwater fish"
    if value < 5.0:
        return f"{value:.2f} mg/L - hypoxic stress likely"
    if value < 8.0:
        return f"{value:.2f} mg/L - adequate for most species"
    return f"{value:.2f} mg/L - well oxygenated"


def _predict_one(state: dict, target: str, explain: bool) -> PredictionResponse:
    try:
        model = registry.get(target)
    except KeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    features = registry.build_features(state, model)
    try:
        value = float(model.predictor.predict(features)[0])
    except Exception as exc:  # noqa: BLE001 - surface as a 422, not a 500
        raise HTTPException(
            status_code=422,
            detail=f"could not predict from the supplied state: {exc}",
        ) from exc

    uncertainty = None
    try:
        _, spread = model.predictor.predict_with_uncertainty(features)
        if np.isfinite(spread[0]):
            uncertainty = float(spread[0])
    except Exception:  # noqa: BLE001 - uncertainty is optional, never fatal
        uncertainty = None

    contributions = None
    if explain:
        contributions = _contributions(model, features)

    return PredictionResponse(
        target=target,
        prediction=value,
        unit=model.unit,
        interpretation=interpret(target, value),
        labels=model.labels,
        uncertainty=uncertainty,
        out_of_range=registry.check_ranges(state, model),
        contributions=contributions,
        caveats=model.caveats,
    )


def _contributions(model, features) -> dict[str, float] | None:
    """Per-feature SHAP contributions, or None if they cannot be produced.

    An explanation is a convenience, so a failure here must not cost the caller
    their prediction.
    """
    try:
        from aquanexus.ml.explainer import HabitatExplainer

        explainer = HabitatExplainer(model.predictor).fit_explainer(features)
        explanation = explainer.explain_prediction(features)
        return {k: float(v) for k, v in explanation.contributions.items()}
    except Exception as exc:  # noqa: BLE001
        log.warning("explanation unavailable: %s", exc)
        return None


@router.post("/predict", response_model=PredictionResponse, summary="Predict one state")
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict a single environmental state.

    The response carries the model's caveats and flags any input outside the
    training range. Out-of-range inputs are answered rather than refused - the
    caller is told, and decides.
    """
    state = request.state.model_dump(exclude_none=True)
    if not state:
        raise HTTPException(status_code=422, detail="state is empty")
    return _predict_one(state, request.target, request.explain)


@router.post("/batch_predict", response_model=BatchPredictionResponse,
             summary="Predict many states")
def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict a batch of states.

    Rows are predicted individually so one malformed entry cannot void the batch;
    a row that cannot be predicted returns NaN in its position.
    """
    try:
        model = registry.get(request.target)
    except KeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    predictions: list[float] = []
    out_of_range = 0
    for state_model in request.states:
        state = state_model.model_dump(exclude_none=True)
        try:
            features = registry.build_features(state, model)
            predictions.append(float(model.predictor.predict(features)[0]))
        except Exception as exc:  # noqa: BLE001
            log.warning("row failed: %s", exc)
            predictions.append(float("nan"))
        if registry.check_ranges(state, model):
            out_of_range += 1

    return BatchPredictionResponse(
        target=request.target, unit=model.unit, labels=model.labels,
        n=len(predictions), predictions=predictions,
        out_of_range_count=out_of_range, caveats=model.caveats,
    )
