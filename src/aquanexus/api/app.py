"""FastAPI application serving the nationwide dissolved-oxygen models.

Run with::

    uvicorn aquanexus.api.app:app --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aquanexus.api.registry import registry
from aquanexus.api.routes import health, predictions, stations
from aquanexus.config import settings
from aquanexus.logger import get_logger

log = get_logger("api.app")

DESCRIPTION = """
Dissolved-oxygen prediction for Japanese rivers, trained on about 565,000 real
monitoring samples from 5,883 stations (Ministry of the Environment, FY2013-22).

Four models are served, from `geo` (works anywhere) to `full` (needs the station's
own history). Every model reports the accuracy it scored on stations it had never
seen, in later years than it trained on; `GET /models` lists those numbers.
Predictions carry a 90% interval, which under-covers when oxygen is below 5 mg/L.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup.

    A failure to load is logged and surfaced through /health rather than
    preventing start: an API that can explain why it is degraded is more useful
    than one that will not boot.
    """
    registry.load()
    if registry.ready:
        log.info("ready with model(s): %s", sorted(registry.models))
    else:
        log.error("started with no models: %s", registry.error)
    yield
    registry.models.clear()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_and_errors(request: Request, call_next):
    """Time every request and convert unhandled errors into JSON.

    Without this an unexpected exception returns an HTML traceback page, which
    is both unhelpful to an API client and a small information leak.
    """
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001 - deliberate catch-all at the boundary
        elapsed = (time.perf_counter() - started) * 1000
        log.exception("unhandled error on %s %s after %.0f ms",
                      request.method, request.url.path, elapsed)
        return JSONResponse(
            status_code=500,
            content={"detail": "internal error", "path": request.url.path},
        )

    elapsed = (time.perf_counter() - started) * 1000
    response.headers["X-Response-Time-ms"] = f"{elapsed:.1f}"
    log.debug("%s %s -> %d in %.1f ms", request.method, request.url.path,
              response.status_code, elapsed)
    return response


app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(stations.router)


@app.get("/", tags=["meta"], summary="Service description")
def root() -> dict:
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "models": sorted(registry.models),
        "note": "Read GET /models for the held-out accuracy of each model before use.",
    }
