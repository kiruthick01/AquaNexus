"""What-if scenario endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from aquanexus.api.registry import registry
from aquanexus.api.routes.predictions import interpret
from aquanexus.api.schemas import ScenarioRequest, ScenarioResponse
from aquanexus.logger import get_logger

log = get_logger("api.scenarios")
router = APIRouter(tags=["scenarios"])


@router.post("/scenario_run", response_model=ScenarioResponse,
             summary="Compare a modified state against its baseline")
def scenario_run(request: ScenarioRequest) -> ScenarioResponse:
    """Apply changes to a baseline state and report the difference.

    Modifications are fractional - a discharge entry of -0.2 means 20% less -
    while absolute_modifications set values directly. Absolute wins where both
    name the same field.

    This is a **model sensitivity, not a simulation**. Changing discharge here
    moves the model's input; it does not re-run HEC-RAS. Where a driver is
    correlated with others in training, changing it alone describes a state that
    may not physically occur, so large excursions are indicative at best.
    """
    try:
        model = registry.get(request.target)
    except KeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    baseline = request.baseline.model_dump(exclude_none=True)
    if not baseline:
        raise HTTPException(status_code=422, detail="baseline state is empty")

    unknown = (set(request.modifications) | set(request.absolute_modifications)) - set(baseline)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"cannot modify field(s) absent from the baseline: {sorted(unknown)}",
        )

    scenario = dict(baseline)
    applied: dict[str, float] = {}
    for name, fraction in request.modifications.items():
        scenario[name] = baseline[name] * (1.0 + fraction)
        applied[name] = float(scenario[name])
    for name, value in request.absolute_modifications.items():
        scenario[name] = value
        applied[name] = float(value)

    try:
        before = float(model.predictor.predict(
            registry.build_features(baseline, model))[0])
        after = float(model.predictor.predict(
            registry.build_features(scenario, model))[0])
    except Exception as exc:  # noqa: BLE001 - surface as 422 rather than 500
        raise HTTPException(status_code=422,
                            detail=f"scenario could not be evaluated: {exc}") from exc

    change = after - before
    percent = (100.0 * change / before) if before else None
    direction = "improves" if change > 0 else ("worsens" if change < 0 else "does not change")

    return ScenarioResponse(
        scenario_name=request.scenario_name, target=request.target, unit=model.unit,
        labels=model.labels, baseline_prediction=before, scenario_prediction=after,
        change=change, percent_change=percent, applied=applied,
        interpretation=(f"{request.scenario_name} {direction} the outcome by "
                        f"{abs(change):.3f} {model.unit}. "
                        f"Scenario: {interpret(request.target, after)}"),
        out_of_range=registry.check_ranges(scenario, model),
        caveats=[*model.caveats,
                 "Scenario moves the model's inputs; it does not re-run the "
                 "hydraulic simulation."],
    )
