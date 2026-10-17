"""Station lookup, so a caller can predict for a known monitoring point."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from aquanexus.api.registry import registry
from aquanexus.api.schemas import StationInfo

router = APIRouter(tags=["stations"])

_FIELDS = list(StationInfo.model_fields)


def _info(row) -> StationInfo:
    data = {k: row[k] for k in _FIELDS}
    data["last_date"] = row["last_date"].date()
    for key in ("name", "water_body", "elevation"):
        if data[key] != data[key]:  # NaN
            data[key] = None
    return StationInfo(**data)


@router.get("/stations", response_model=list[StationInfo], summary="Search stations")
def search_stations(
    q: str | None = Query(None, description="Part of a station or river name"),
    prefecture: int | None = Query(None, ge=1, le=47),
    limit: int = Query(25, ge=1, le=200),
) -> list[StationInfo]:
    if registry.stations is None:
        raise HTTPException(status_code=503, detail="station table not loaded")
    found = registry.search_stations(q, prefecture, limit)
    return [_info(r) for _, r in found.iterrows()]


@router.get("/stations/{station_id}", response_model=StationInfo, summary="One station")
def get_station(station_id: int) -> StationInfo:
    row = registry.station(station_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown station {station_id}")
    return _info(row)
