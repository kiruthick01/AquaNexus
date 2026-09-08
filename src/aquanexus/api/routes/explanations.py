"""SHAP explanation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from aquanexus.api.registry import registry
from aquanexus.api.schemas import (
    ExplanationResponse,
    FeatureContribution,
    PredictionRequest,
)
from aquanexus.api.throttle import BoundedCache, TokenBucket
from aquanexus.config import settings
from aquanexus.logger import get_logger

log = get_logger("api.explanations")
router = APIRouter(tags=["explainability"])

#: Explanations are a deterministic function of the feature row, so the same
#: state asked twice is answered from memory.
explanation_cache = BoundedCache(settings.EXPLAIN_CACHE_SIZE)

#: Spent only on a cache miss - the limit protects CPU, and a hit costs none.
explain_limiter = TokenBucket(settings.EXPLAIN_RATE_LIMIT)


def cache_key(target: str, features) -> tuple:
    """A hashable identity for one explanation request.

    Rounded to six decimals: a slider that emits 24.500000000000004 should hit
    the same entry as one that emits 24.5.
    """
    row = features.iloc[0]
    return (target, *((name, round(float(value), 6) if value == value else None)
                      for name, value in row.items()))


@router.post("/explain", response_model=ExplanationResponse,
             summary="Explain a prediction with SHAP")
def explain(request: PredictionRequest,
            http_request: Request) -> ExplanationResponse:
    """Per-feature contributions for one prediction.

    Contributions sum to the difference between the prediction and the model's
    expected value.

    Repeated states are served from an in-process cache; a state not seen
    before spends a token from a per-client rate limit, because computing one
    costs about 200 ms of SHAP. Exceeding it returns 429 with `Retry-After`.

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
    if model.background is None:
        raise HTTPException(
            status_code=503,
            detail=f"no background sample for {request.target}; SHAP needs a "
                   "reference distribution. Re-run scripts/train_models.py.",
        )

    key = cache_key(request.target, features)
    explanation = explanation_cache.get(key)

    if explanation is None:
        client = http_request.client.host if http_request.client else "unknown"
        if not explain_limiter.take(client):
            wait = explain_limiter.retry_after(client)
            log.warning("rate limited %s on /explain", client)
            raise HTTPException(
                status_code=429,
                detail=f"too many explanations; each one costs about 200 ms of "
                       f"SHAP. Retry in {wait}s, or repeat a state already "
                       f"explained - those are served from cache.",
                headers={"Retry-After": str(wait)},
            )
        try:
            # The background is the *training* sample, never the row being
            # explained: explaining a request against itself makes every
            # contribution exactly zero, which looks like an answer and is not
            # one.
            explainer = registry.explainer_for(model)
            explanation = explainer.explain_prediction(features,
                                                       top=len(model.features))
        except Exception as exc:  # noqa: BLE001 - explanation may be unavailable
            raise HTTPException(status_code=503,
                                detail=f"explanation unavailable: {exc}") from exc
        explanation_cache.put(key, explanation)

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
        collinear_pairs=[list(pair) for pair in model.collinear_pairs],
        caveats=model.caveats,
    )
