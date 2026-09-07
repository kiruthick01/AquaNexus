"""Health and model-information endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from aquanexus.api.registry import registry
from aquanexus.api.schemas import HealthResponse, ModelInfo
from aquanexus.config import settings

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and readiness")
def health() -> HealthResponse:
    """Report whether models are actually loaded.

    Returns "degraded" rather than a 503 when no model is available: the process
    is alive and able to say why, which is more useful to an operator than a bare
    failure.
    """
    return HealthResponse(
        status="ok" if registry.ready else "degraded",
        version=settings.VERSION,
        river=f"{settings.RIVER_NAME} ({settings.RIVER_NAME_JA})",
        models_loaded=sorted(registry.models),
        detail=registry.error,
    )


@router.get("/models", response_model=list[ModelInfo], summary="Describe the models")
def list_models() -> list[ModelInfo]:
    """Every served model, with its metrics and caveats.

    Worth reading before the prediction endpoints: one of the two models is
    trained on synthetic labels, and says so here.
    """
    if not registry.ready:
        raise HTTPException(status_code=503, detail=registry.error or "no models loaded")
    return [
        ModelInfo(
            target=name, unit=m.unit, model_type=m.model_type, labels=m.labels,
            n_train=m.n_train, features=m.features,
            metrics=dict(m.metrics), caveats=m.caveats,
            training_ranges={k: list(v) for k, v in m.training_ranges.items()},
            target_range=list(m.target_range),
        )
        for name, m in sorted(registry.models.items())
    ]
