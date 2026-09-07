"""Pydantic request and response models for the API.

Every prediction response carries the model's caveats. That is deliberate: the
habitat model is trained on labels this repository generated, and the oxygen
model is unreliable at low flow. A consumer reading only the number would not
know either, so the API states it rather than leaving it in a document nobody
opens.

Input ranges are validated against what the model has actually seen. A request
outside them is answered, but flagged - refusing it would be unhelpful, and
answering silently would imply a confidence the model has not earned.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Target = Literal["dissolved_oxygen", "hsi"]


class EnvironmentalState(BaseModel):
    """One environmental observation to predict from.

    Only the fields a given model needs are required; the rest are optional and
    derived where possible. Bounds are physical limits, not the training range -
    training coverage is reported separately in the response.
    """

    water_temp: float | None = Field(None, ge=-2, le=45, description="Water temperature (°C)")
    air_temp: float | None = Field(None, ge=-30, le=45, description="Air temperature (°C)")
    dissolved_oxygen: float | None = Field(None, ge=0, le=25, description="DO (mg/L)")
    discharge: float | None = Field(None, ge=0, le=5000, description="Discharge (m³/s)")
    depth: float | None = Field(None, ge=0, le=30, description="Depth (m)")
    velocity: float | None = Field(None, ge=0, le=10, description="Velocity (m/s)")
    suspended_solids: float | None = Field(None, ge=0, le=5000, description="SS (mg/L)")
    ph: float | None = Field(None, ge=3, le=11)
    nitrogen_total: float | None = Field(None, ge=0, le=50, description="Total N (mg/L)")
    phosphorus_total: float | None = Field(None, ge=0, le=10, description="Total P (mg/L)")
    month: int | None = Field(None, ge=1, le=12, description="Month, for seasonal terms")

    model_config = {
        "json_schema_extra": {
            "example": {
                "water_temp": 24.5, "dissolved_oxygen": 7.2, "discharge": 12.0,
                "depth": 1.8, "velocity": 0.45, "suspended_solids": 15.0, "month": 7,
            }
        }
    }


class PredictionRequest(BaseModel):
    state: EnvironmentalState
    target: Target = "dissolved_oxygen"
    explain: bool = Field(False, description="Include per-feature SHAP contributions")


class RangeWarning(BaseModel):
    """A feature whose value lies outside the model's training range."""

    feature: str
    value: float
    training_min: float
    training_max: float


class PredictionResponse(BaseModel):
    target: Target
    prediction: float
    unit: str
    interpretation: str
    labels: Literal["observed", "SYNTHETIC"]
    uncertainty: float | None = Field(
        None, description="Model spread where available; null when the model "
                          "cannot express one, rather than a fabricated value"
    )
    out_of_range: list[RangeWarning] = Field(default_factory=list)
    contributions: dict[str, float] | None = None
    caveats: list[str] = Field(default_factory=list)


class BatchPredictionRequest(BaseModel):
    states: list[EnvironmentalState] = Field(..., min_length=1, max_length=1000)
    target: Target = "dissolved_oxygen"

    @field_validator("states")
    @classmethod
    def _not_empty(cls, value):
        if not value:
            raise ValueError("at least one state is required")
        return value


class BatchPredictionResponse(BaseModel):
    target: Target
    unit: str
    labels: Literal["observed", "SYNTHETIC"]
    n: int
    predictions: list[float]
    out_of_range_count: int = 0
    caveats: list[str] = Field(default_factory=list)


class ScenarioRequest(BaseModel):
    """A what-if: change some drivers relative to a baseline state."""

    scenario_name: str = Field(..., min_length=1, max_length=120)
    baseline: EnvironmentalState
    modifications: dict[str, float] = Field(
        ...,
        description="Fractional changes per field, e.g. {'discharge': -0.2} for "
                    "20% lower discharge. Use absolute_modifications for "
                    "direct values.",
    )
    absolute_modifications: dict[str, float] = Field(default_factory=dict)
    target: Target = "dissolved_oxygen"

    @field_validator("modifications")
    @classmethod
    def _bounded(cls, value):
        for name, change in value.items():
            if change < -1.0:
                raise ValueError(
                    f"{name}: a fractional change below -1.0 would make the "
                    "value negative"
                )
        return value


class ScenarioResponse(BaseModel):
    scenario_name: str
    target: Target
    unit: str
    labels: Literal["observed", "SYNTHETIC"]
    baseline_prediction: float
    scenario_prediction: float
    change: float
    percent_change: float | None
    applied: dict[str, float]
    interpretation: str
    out_of_range: list[RangeWarning] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    value: float | None = None


class ExplanationResponse(BaseModel):
    target: Target
    prediction: float
    baseline: float = Field(..., description="Model's expected value")
    contributions: list[FeatureContribution]
    collinear_pairs: list[list[str]] = Field(
        default_factory=list,
        description="Feature pairs correlated above 0.9 in training. SHAP "
                    "divides credit between such features arbitrarily, so their "
                    "individual contributions should be read as one combined "
                    "effect.",
    )
    caveats: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    target: Target
    unit: str
    model_type: str
    labels: Literal["observed", "SYNTHETIC"]
    n_train: int
    features: list[str]
    metrics: dict[str, float | str]
    caveats: list[str]
    training_ranges: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Observed [min, max] per feature. A client that cannot see "
                    "these cannot show where the model's evidence ends, and "
                    "would have to hard-code them to try.",
    )
    target_range: list[float] = Field(
        default_factory=list,
        description="Observed [min, max] of the predicted quantity itself, so a "
                    "client can show a reading against the range actually "
                    "measured on this river.",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    river: str
    models_loaded: list[str]
    detail: str | None = None
