"""Prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from aquanexus.api.registry import registry
from aquanexus.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter(tags=["predictions"])

CAVEATS = [
    "Trained on monthly river samples; predicts DO at sampling dates, not continuously.",
    "Intervals cover about 90% of cases overall but far fewer when DO is below 5 mg/L.",
    "Missing inputs are treated as unknown, which lowers accuracy; see missing_features.",
]


def interpret(value: float) -> str:
    """Plain-language reading of a DO value; 5 mg/L is the usual stress threshold."""
    if value < 2.0:
        return f"{value:.2f} mg/L - severe hypoxia, acutely lethal for most fish"
    if value < 5.0:
        return f"{value:.2f} mg/L - below the 5 mg/L stress threshold"
    if value < 8.0:
        return f"{value:.2f} mg/L - adequate for most species"
    return f"{value:.2f} mg/L - well oxygenated"


def _predict(request: PredictionRequest) -> PredictionResponse:
    try:
        bundle = registry.get(request.mode)
        built = registry.build_input(bundle, request.as_inputs())
    except KeyError as exc:
        text = str(exc).strip("'\"")
        raise HTTPException(status_code=503 if "not loaded" in text else 404, detail=text) from exc

    reason = registry.missing_signal(bundle, built)
    if reason:
        detail = " ".join([reason + ".", *built.notes])
        raise HTTPException(status_code=422, detail=detail)

    out = bundle.predict(built.row).iloc[0]
    prediction = float(out["prediction"])
    caveats = list(CAVEATS)
    if request.mode == "full_alert":
        caveats.append(
            "full_alert over-predicts low DO on purpose: expect about 3 false alarms in 4."
        )
    return PredictionResponse(
        mode=request.mode,
        prediction=prediction,
        probability_below_5=float(out["p_low"]) if "p_low" in out else None,
        lower=float(out["lower"]),
        upper=float(out["upper"]),
        level=bundle.level,
        interpretation=interpret(prediction),
        missing_features=built.missing,
        out_of_range=built.out_of_range,
        contributions=registry.contributions(bundle, built.row) if request.explain else None,
        notes=built.notes,
        caveats=caveats,
    )


@router.post("/predict", response_model=PredictionResponse, summary="Predict dissolved oxygen")
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict DO (mg/L) with a 90% interval for one place and date.

    Inputs outside the training range are answered but flagged in ``out_of_range``.
    """
    return _predict(request)


@router.post("/batch_predict", response_model=BatchPredictionResponse, summary="Predict many")
def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    predictions = [_predict(r) for r in request.requests]
    return BatchPredictionResponse(n=len(predictions), predictions=predictions)
