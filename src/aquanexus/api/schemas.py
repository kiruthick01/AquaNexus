"""Request and response models for the dissolved-oxygen API.

A prediction is for one place on one date. The caller supplies weather and, for the
richer modes, what neighbouring stations or the station itself read earlier; any
input left out is passed to the model as missing and listed in the response, so a
degraded prediction says so instead of looking complete.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Mode = Literal["geo", "nowcast", "full", "full_alert"]


class Location(BaseModel):
    latitude: float | None = Field(None, ge=20, le=46, description="Degrees north (Japan)")
    longitude: float | None = Field(None, ge=122, le=154, description="Degrees east (Japan)")
    elevation: float | None = Field(None, ge=-10, le=3800, description="Metres above sea level")
    relief_10km: float | None = Field(
        None, ge=0, le=3800, description="Elevation range within 10 km"
    )
    rel_elevation: float | None = Field(
        None, description="Height relative to stations within 10 km"
    )


class Weather(BaseModel):
    air_temp: float | None = Field(
        None, ge=-30, le=45, description="Daily mean air temperature, °C"
    )
    air_temp_3d: float | None = Field(None, ge=-30, le=45, description="Mean of the 3 days before")
    air_temp_7d: float | None = Field(None, ge=-30, le=45)
    air_temp_30d: float | None = Field(None, ge=-30, le=45)
    rain_1d: float | None = Field(None, ge=0, le=1000, description="Rain the day before, mm")
    rain_3d: float | None = Field(None, ge=0, le=2000)
    rain_7d: float | None = Field(None, ge=0, le=3000)
    rain_14d: float | None = Field(None, ge=0, le=5000)
    rain_30d: float | None = Field(None, ge=0, le=8000)
    days_since_rain: float | None = Field(None, ge=0, le=400)
    solar: float | None = Field(None, ge=0, le=45, description="Surface solar, MJ/m²/day")
    wind: float | None = Field(None, ge=0, le=60, description="10 m wind, m/s")


class History(BaseModel):
    prev_do: float | None = Field(None, ge=0, le=25, description="Station's previous DO, mg/L")
    prev_do_gap_days: float | None = Field(None, ge=0, description="Days since that reading")
    station_do_mean: float | None = Field(
        None, ge=0, le=25, description="Station's long-run mean DO"
    )
    station_do_n: float | None = Field(None, ge=0, description="Number of earlier visits")


class Neighbors(BaseModel):
    nb_do_prior: float | None = Field(None, ge=0, le=25)
    nb_do_prior_n: float | None = Field(None, ge=0)
    nb_do_now: float | None = Field(
        None, ge=0, le=25, description="Neighbours' mean DO within 10 days"
    )
    nb_do_now_n: float | None = Field(None, ge=0)
    nb_count: float | None = Field(None, ge=0)
    nb_dist_mean: float | None = Field(None, ge=0)


class PredictionRequest(BaseModel):
    date: dt.date
    mode: Mode = "geo"
    station_id: int | None = Field(None, description="A known station; fills location and history")
    location: Location = Field(default_factory=Location)
    weather: Weather = Field(default_factory=Weather)
    history: History = Field(default_factory=History)
    neighbors: Neighbors = Field(default_factory=Neighbors)
    explain: bool = Field(False, description="Include exact per-feature contributions")

    @model_validator(mode="after")
    def _needs_a_place(self) -> PredictionRequest:
        loc = self.location
        if self.station_id is None and (loc.latitude is None or loc.longitude is None):
            raise ValueError("give a station_id, or location.latitude and location.longitude")
        return self

    def as_inputs(self) -> dict:
        return {
            "date": self.date,
            "station_id": self.station_id,
            "location": self.location.model_dump(exclude_none=True),
            "weather": self.weather.model_dump(exclude_none=True),
            "history": self.history.model_dump(exclude_none=True),
            "neighbors": self.neighbors.model_dump(exclude_none=True),
        }


class RangeWarning(BaseModel):
    feature: str
    value: float
    training_min: float
    training_max: float


class PredictionResponse(BaseModel):
    mode: Mode
    prediction: float = Field(description="Dissolved oxygen, mg/L")
    lower: float
    upper: float
    level: float = Field(description="Nominal coverage of [lower, upper]")
    probability_below_5: float | None = Field(
        None, description="Calibrated probability that DO is below 5 mg/L, when the model has one"
    )
    interpretation: str
    missing_features: list[str]
    out_of_range: list[RangeWarning] = Field(default_factory=list)
    contributions: dict[str, float] | None = None
    notes: list[str] = Field(default_factory=list)
    caveats: list[str]


class BatchPredictionRequest(BaseModel):
    requests: list[PredictionRequest] = Field(min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    n: int
    predictions: list[PredictionResponse]


class ModelInfo(BaseModel):
    mode: Mode
    summary: str
    features: list[str]
    low_do_weight: float
    trained_on: dict
    held_out_strict: dict = Field(description="Scored on unseen stations in later years")
    held_out_classifier: dict = Field(
        default_factory=dict, description="Ranking and calibration of the below-5 probability"
    )
    held_out_note: str
    importance: dict[str, float] = Field(description="Mean |contribution|, mg/L")
    interval_half_widths: list[float]
    interval_bin_edges: list[float]
    built: str
    commit: str


class StationInfo(BaseModel):
    station_id: int
    name: str | None
    water_body: str | None
    pref_code: int
    latitude: float
    longitude: float
    elevation: float | None
    last_date: dt.date
    last_do: float
    station_do_mean: float
    station_do_n: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    models_loaded: list[str]
    stations: int
    detail: str | None = None
