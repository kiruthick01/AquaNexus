# Backend image. HEC-RAS itself is Windows-only and is NOT installed here —
# simulations are run on the host and their outputs mounted into data/hecras.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp is required by xgboost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --upgrade pip && pip install ".[ml,api]"

COPY scripts/ ./scripts/

RUN useradd --create-home --uid 1000 aquanexus \
    && mkdir -p /app/data \
    && chown -R aquanexus:aquanexus /app
USER aquanexus

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"

CMD ["uvicorn", "aquanexus.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
