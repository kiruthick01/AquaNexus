"""Health and model-information endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from aquanexus.api.registry import MODE_SUMMARY, registry
from aquanexus.api.schemas import HealthResponse, ModelInfo
from aquanexus.config import settings

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and readiness")
def health() -> HealthResponse:
    """Report whether models are loaded; "degraded" rather than a 503 when none are."""
    stations = 0 if registry.stations is None else len(registry.stations)
    return HealthResponse(
        status="ok" if registry.ready else "degraded",
        version=settings.VERSION,
        models_loaded=sorted(registry.bundles),
        stations=stations,
        detail=registry.error,
    )


@router.get("/models", response_model=list[ModelInfo], summary="Describe the models")
def list_models() -> list[ModelInfo]:
    """Every served model with the held-out accuracy it is reported with."""
    if not registry.ready:
        raise HTTPException(status_code=503, detail=registry.error or "no models loaded")
    out = []
    for mode, bundle in sorted(registry.bundles.items()):
        meta = bundle.metadata
        out.append(
            ModelInfo(
                mode=mode,
                summary=MODE_SUMMARY.get(mode, ""),
                features=bundle.features,
                low_do_weight=meta.get("low_do_weight", 1.0),
                trained_on=meta.get("trained_on", {}),
                held_out_strict=meta.get("held_out_strict", {}),
                held_out_classifier=meta.get("held_out_strict_classifier", {}),
                held_out_note=meta.get("held_out_note", ""),
                importance=meta.get("importance", {}),
                interval_half_widths=bundle.bin_widths,
                interval_bin_edges=bundle.bin_edges,
                built=meta.get("built", ""),
                commit=meta.get("commit", ""),
            )
        )
    return out
