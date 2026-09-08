"""What-if scenario endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from aquanexus.api.registry import registry
from aquanexus.api.routes.predictions import interpret
from aquanexus.api.schemas import ScenarioRequest, ScenarioResponse
from aquanexus.logger import get_logger

log = get_logger("api.scenarios")
router = APIRouter(tags=["scenarios"])


def _carry_hydraulics(baseline: dict, scenario: dict,
                      applied: dict[str, float]) -> dict[str, float]:
    """Re-interpolate the hydraulics when the discharge has moved.

    A caller's depth and velocity describe the baseline. Leaving them untouched
    while discharge changes hands the model a state the river cannot be in - a
    drought at the flood's depth - so they are replaced from the sweep, in the
    scenario only.

    Anything the caller set explicitly is left alone: they have said what they
    want, and second-guessing it would be worse than the inconsistency.
    """
    if "discharge" not in applied:
        return {}
    new_discharge = scenario.get("discharge")
    if new_discharge is None or new_discharge == baseline.get("discharge"):
        return {}

    reach = registry.reach_hydraulics(float(new_discharge))
    if not reach:
        return {}

    derived: dict[str, float] = {}
    for field, alias in (("depth", "reach_depth"), ("velocity", "reach_velocity")):
        if field in applied:
            continue  # the caller asked for this value; leave it
        if alias in reach:
            scenario[field] = reach[alias]
            derived[field] = round(reach[alias], 4)
    if "reach_top_width" in reach:
        derived["top_width"] = round(reach["reach_top_width"], 4)
    return derived


def _method_caveat(derived: dict[str, float]) -> str:
    if derived:
        return ("Depth, velocity and width were re-interpolated from the "
                "HEC-RAS sweep at the scenario discharge. The simulation itself "
                "is not re-run, so this is a model sensitivity, not a forecast.")
    return ("Scenario moves the model's inputs; it does not re-run the "
            "hydraulic simulation.")


@router.post("/scenario_run", response_model=ScenarioResponse,
             summary="Compare a modified state against its baseline")
def scenario_run(request: ScenarioRequest) -> ScenarioResponse:
    """Apply changes to a baseline state and report the difference.

    Modifications are fractional - a discharge entry of -0.2 means 20% less -
    while absolute_modifications set values directly. Absolute wins where both
    name the same field.

    Changing discharge carries the hydraulics with it: depth, velocity and width
    are re-interpolated from the HEC-RAS flow sweep at the new discharge, which
    is how the training rows were built. The simulation itself is not re-run -
    the sweep is a precomputed set of steady-flow profiles - so this remains a
    **model sensitivity rather than a forecast**, and any driver changed alone
    still describes a state that may not physically occur.

    A caller who sets depth or velocity explicitly is taken at their word and
    nothing is re-derived for them.
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

    derived = _carry_hydraulics(baseline, scenario, applied)

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
        change=change, percent_change=percent, applied=applied, derived=derived,
        interpretation=(f"{request.scenario_name} {direction} the outcome by "
                        f"{abs(change):.3f} {model.unit}. "
                        f"Scenario: {interpret(request.target, after)}"),
        out_of_range=registry.check_ranges(scenario, model),
        caveats=[*model.caveats, _method_caveat(derived)],
    )
