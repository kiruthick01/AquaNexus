# Deployment

Two images: a FastAPI backend and an nginx-served React bundle. Neither has been
built — Docker is not installed on the development machine — so read the
[Verification status](#verification-status) section before trusting any of this.

---

## Running locally, without containers

```bash
# Backend
pip install -e ".[ml,api]"
python scripts/train_models.py                 # writes data/models/
uvicorn aquanexus.api.app:app --reload         # http://localhost:8000

# Frontend, in a second shell
cd frontend
npm install
npm run dev                                    # http://localhost:3000
```

Open the app at **`http://localhost:3000`**. `http://127.0.0.1:3000` also works,
but only because both spellings are in `CORS_ORIGINS` — a browser treats them as
different origins, and an unlisted one fails as an unexplained "cannot reach the
API".

Check it end to end:

```bash
python scripts/verify_deployment.py --url http://localhost:8000
```

---

## Containers

```bash
docker compose up --build
# API      http://localhost:8000  (docs at /docs)
# Frontend http://localhost:3000
```

### What the compose file does, and why

**Models are not in the image.** They are build outputs, not source, so `./data`
is mounted. Without that mount the API starts *degraded*: it stays up, `/health`
returns `degraded` with the reason, and the prediction endpoints answer 503.
That is deliberate — a service that can say why it is useless beats one that
will not boot.

**`DATA_DIR=/app/data` is pinned in the Dockerfile.** The settings derive their
paths from the source file location, which in an installed layout is
site-packages rather than `/app`. Without the pin the mount would be ignored and
the API would start degraded with a correct-looking volume attached.

**`PYTHONPATH=/app/src` is set for the API service.** The image installs the
package non-editably, so mounting `./src` over `/app/src` would otherwise change
nothing: imports would still resolve to the copy inside the image, while
`--reload` restarted on edits that could not take effect.

**The frontend's API URL is a build argument, not an environment variable.**
Vite substitutes `import.meta.env` when the bundle is built, so a running
container cannot be repointed:

```bash
docker build --build-arg VITE_API_BASE_URL=https://api.example.com ./frontend
```

It must be an address the **browser** can reach. `http://api:8000` resolves
inside the compose network and fails in the browser.

---

## Deploying somewhere real

Nothing here is production-hardened. At minimum, before exposing it:

| Concern | Current state | What to do |
|---|---|---|
| Authentication | None. Every endpoint is open. | Put it behind a gateway or add an API key. |
| CORS | Allows the two local dev origins. | Set `CORS_ORIGINS` to the deployed frontend. |
| TLS | None; plain HTTP. | Terminate TLS at a proxy. |
| Rate limiting | None. `/explain` costs ~200 ms of SHAP per call. | Limit it, or cache by state. |
| Model artefacts | Mounted from the host. | Bake them into a release image, or fetch from object storage at startup. |
| Workers | One uvicorn worker. | `--workers N`; each loads its own copy of the models (~35 MB). |
| Logs | stdout, human-readable. | `LOG_JSON=true` for structured output. |

### Configuration

Everything in `aquanexus.config.Settings` reads from the environment. The ones
that matter for a deployment:

| Variable | Default | Notes |
|---|---|---|
| `DATA_DIR` | `<repo>/data` | Setting it moves `raw`, `processed`, `models`, `hecras`, `cache` with it. |
| `MODELS_DIR` | `$DATA_DIR/models` | Set explicitly to override that. |
| `CORS_ORIGINS` | the four local dev origins | Must include the deployed frontend's origin. |
| `LOG_LEVEL` | `INFO` | |
| `LOG_JSON` | `false` | |
| `API_DEBUG` | `false` | |

`HECRAS_EXE` is irrelevant to a deployment: HEC-RAS is Windows-only and is used
to *generate* hydraulics, not to serve them. The container never runs it.

---

## Verification status

Honest accounting of what has actually been checked.

**Verified without Docker:**

- `tests/test_deployment.py` (20 tests) — every path a `COPY` reads exists, the
  extras installed are declared in `pyproject.toml`, `.dockerignore` keeps the
  README the build needs, the `CMD` import path resolves, the `HEALTHCHECK`
  probes a route the app serves, compose mounts resolve, the frontend's API URL
  is a build argument, and the SPA fallback is configured.
- The API image's install step, by installing `.[ml,api]` into a clean virtual
  environment and serving from the installed copy with no source on the path —
  degraded without `DATA_DIR`, and 14/14 smoke checks with it.
- The frontend build (`npm run build`) and its 22 tests, on every change.

**Not verified:** `docker build`, either image. The base images, Linux wheels,
`libgomp`, the non-root user against a bind-mounted volume, the nginx config as
nginx reads it, and the healthcheck loops are all unchecked. The `containers`
job in `.github/workflows/ci.yml` builds both images and starts the API, so the
first CI run on a machine with a daemon is where this gets settled.
