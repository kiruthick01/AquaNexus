"""SHAP explanation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from aquanexus.api.registry import registry
from aquanexus.api.schemas import (
    ExplanationResponse,
    FeatureContribution,
    PredictionRequest,
)
from aquanexus.logger import get_logger

log = get_logger("api.explanations")
router = APIRouter(tags=["explainability"])


@router.post("/explain", response_model=ExplanationResponse,
             summary="Explain a prediction with SHAP")
def explain(request: PredictionRequest) -> ExplanationResponse:
    """Per-feature contributions for one prediction.

    Contributions sum to the difference between the prediction and the model's
    expected value.

    The response also lists feature pairs that are collinear in training. SHAP
    divides credit between such features arbitrarily, so their contributions
    should be read as one combined effect rather than as a ranking - on this
    model the hydraulic features are all derived from discharge and move
    together.
    """
    try:
        model = registry.get(request.target)
    except KeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    state = request.state.model_dump(exclude_none=True)
    if not state:
        raise HTTPException(status_code=422, detail="state is empty")

    features = registry.build_features(state, model)
    try:
        from aquanexus.ml.explainer import HabitatExplainer

        explainer = HabitatExplainer(model.predictor).fit_explainer(features)
        explanation = explainer.explain_prediction(features, top=len(model.features))
    except Exception as exc:  # noqa: BLE001 - explanation may be unavailable
        raise HTTPException(status_code=503,
                            detail=f"explanation unavailable: {exc}") from exc

    ordered = explanation.contributions.reindex(
        explanation.contributions.abs().sort_values(ascending=False).index
    )
    contributions = []
    for name, value in ordered.items():
        present = name in features.columns and bool(features[name].notna().iloc[0])
        contributions.append(FeatureContribution(
            feature=name,
            contribution=float(value),
            value=float(features[name].iloc[0]) if present else None,
        ))

    return ExplanationResponse(
        target=request.target,
        prediction=explanation.prediction,
        baseline=explanation.expected_value,
        contributions=contributions,
        collinear_pairs=[list(pair) for pair in getattr(model, "collinear_pairs", [])],
        caveats=model.caveats,
    )
