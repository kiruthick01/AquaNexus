#!/usr/bin/env bash
# Run both images and check what only a running container can show.
#
# tests/test_deployment.py checks everything about the images that can be
# checked without a daemon, and CI builds both. Neither answers the questions
# below, and until this script existed they were answered by inference:
#
#   - Does the API container, with artefacts mounted, actually *load* them? The
#     DATA_DIR trap in the Dockerfile is silent when it bites: the mount looks
#     right and the service starts degraded.
#   - Does nginx serve the single-page app - a client route on reload, a
#     never-cached /config.js, immutable assets, a 404 for a missing asset
#     rather than an HTML page with a JavaScript MIME type?
#   - Does one image really serve any backend, or was the entrypoint only ever
#     checked by reading it?
#   - Do the HEALTHCHECK loops converge, or is the interval/retry arithmetic
#     wrong in a way nothing notices until an orchestrator restarts a container?
#
# No Docker on the development machine, so this runs in CI. It takes the images
# as arguments so that a developer with a daemon runs exactly what CI runs:
#
#     docker build -t aquanexus-api:dev .
#     docker build -t aquanexus-frontend:dev ./frontend
#     scripts/check_containers.sh aquanexus-api:dev aquanexus-frontend:dev
#
# Exit status is 0 only if every check passed.
set -uo pipefail

API_IMAGE="${1:-aquanexus-api:ci}"
WEB_IMAGE="${2:-aquanexus-frontend:ci}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-8080}"
# The interpreter that runs the smoke suite; it needs httpx.
PYTHON="${PYTHON:-python}"
API_NAME="aquanexus-api-check"
WEB_NAME="aquanexus-web-check"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
API_URL="http://localhost:${API_PORT}"
WEB_URL="http://localhost:${WEB_PORT}"

failures=0

pass() { printf '  [PASS] %s\n' "$*"; }
fail() { printf '  [FAIL] %s\n' "$*"; failures=$((failures + 1)); }

cleanup() {
  # Logs before removal: a container that never became healthy has said why.
  if [ "$failures" -ne 0 ]; then
    for name in "$API_NAME" "$WEB_NAME"; do
      printf '\n--- docker logs %s ---\n' "$name"
      docker logs "$name" 2>&1 | tail -50 || true
    done
  fi
  docker rm -f "$API_NAME" "$WEB_NAME" > /dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

wait_for_health() {  # wait_for_health <container> <seconds>
  local name="$1" limit="$2" waited=0 status
  while [ "$waited" -lt "$limit" ]; do
    status="$(docker inspect --format '{{.State.Health.Status}}' "$name" 2>/dev/null || echo missing)"
    case "$status" in
      healthy) printf '  [PASS] %s HEALTHCHECK reports healthy after %ds\n' "$name" "$waited"; return 0 ;;
      unhealthy)
        printf '  [FAIL] %s HEALTHCHECK reports unhealthy\n' "$name"
        health_log "$name"
        failures=$((failures + 1))
        return 1 ;;
    esac
    sleep 2
    waited=$((waited + 2))
  done
  printf '  [FAIL] %s never became healthy (%s after %ds)\n' "$name" "$status" "$limit"
  health_log "$name"
  failures=$((failures + 1))
  return 1
}

health_log() {  # health_log <container>
  # What the probe itself said. Without this the failure is the word
  # "unhealthy", which does not distinguish a service that is down from a probe
  # that is wrong - and the first time this script ran, it was the probe.
  docker inspect --format '{{range .State.Health.Log}}    exit {{.ExitCode}}: {{.Output}}{{end}}' \
    "$1" 2>/dev/null | head -10
}

body() { curl -sS "$@"; }
status_of() { curl -sS -o /dev/null -w '%{http_code}' "$1"; }
headers_of() { curl -sS -o /dev/null -D - "$1"; }

# ---------------------------------------------------------------------------
# The API container, holding artefacts
# ---------------------------------------------------------------------------

printf '\nAPI container: %s\n' "$API_IMAGE"

# World-writable because the image runs as its own unprivileged user, whose uid
# does not match the one that owns the checkout.
mkdir -p "$WORK/data/processed" "$WORK/data/models"
cp "$REPO/data/processed/holdout_naka.json" "$WORK/data/processed/"
chmod -R 777 "$WORK/data"

# Fabricated artefacts, made *by the image* - which is itself the check that
# scripts/ is in it and that DATA_DIR resolves to the mount rather than into
# site-packages. See scripts/make_stand_in_artifacts.py for what they are not.
if docker run --rm -v "$WORK/data:/app/data" "$API_IMAGE" \
     python scripts/make_stand_in_artifacts.py > "$WORK/stand-in.log" 2>&1; then
  pass "stand-in artefacts written by the image into the mount"
else
  fail "the image could not write stand-in artefacts"
  tail -20 "$WORK/stand-in.log"
fi

docker run -d --name "$API_NAME" -p "${API_PORT}:8000" \
  -v "$WORK/data:/app/data" "$API_IMAGE" > /dev/null
wait_for_health "$API_NAME" 120

health="$(body "$API_URL/health")"
case "$health" in
  *'"status":"ok"'*) pass "API loaded the mounted artefacts (not degraded)" ;;
  *) fail "API started degraded with artefacts mounted: $health" ;;
esac

printf '\nSmoke checks against the loaded service\n'
if "$PYTHON" "$REPO/scripts/verify_deployment.py" --url "$API_URL"; then
  pass "verify_deployment.py against a container"
else
  fail "verify_deployment.py against a container"
fi

# ---------------------------------------------------------------------------
# The frontend container
# ---------------------------------------------------------------------------

printf '\nFrontend container: %s\n' "$WEB_IMAGE"

docker run -d --name "$WEB_NAME" -p "${WEB_PORT}:80" \
  -e API_BASE_URL="$API_URL" "$WEB_IMAGE" > /dev/null
wait_for_health "$WEB_NAME" 90

index="$(body "$WEB_URL/")"
case "$index" in
  *'<div id="root">'*) pass "index.html served" ;;
  *) fail "index.html served: root element missing" ;;
esac

# Client routes are not files. Without the try_files fallback a reload on any
# page but "/" is a 404 - the defect that only appears after deployment.
for route in /predict /scenarios /analyze /transfer /about /no/such/page; do
  code="$(status_of "$WEB_URL$route")"
  route_body="$(body "$WEB_URL$route")"
  if [ "$code" = "200" ] && [ "$route_body" = "$index" ]; then
    pass "reload of $route serves the app"
  else
    fail "reload of $route returned HTTP $code"
  fi
done

config_headers="$(headers_of "$WEB_URL/config.js")"
case "$(printf '%s' "$config_headers" | tr 'A-Z' 'a-z')" in
  *'cache-control: no-store'*) pass "/config.js is never cached" ;;
  *) fail "/config.js is cacheable; a stale copy points the page at the old API" ;;
esac

config="$(body "$WEB_URL/config.js")"
case "$config" in
  *"apiBaseUrl: \"$API_URL\""*) pass "/config.js carries the API_BASE_URL given at run time" ;;
  *) fail "/config.js does not carry the run-time API_BASE_URL: $config" ;;
esac

# One image, any backend: the whole reason the URL is not baked into the bundle.
# Restarting with a different address must change the served config without a
# rebuild.
docker rm -f "$WEB_NAME" > /dev/null
docker run -d --name "$WEB_NAME" -p "${WEB_PORT}:80" \
  -e API_BASE_URL="https://api.example.test" "$WEB_IMAGE" > /dev/null
wait_for_health "$WEB_NAME" 90
case "$(body "$WEB_URL/config.js")" in
  *'apiBaseUrl: "https://api.example.test"'*)
    pass "the same image serves a different backend without a rebuild" ;;
  *) fail "the runtime config did not follow API_BASE_URL on restart" ;;
esac

asset="$(printf '%s' "$index" | grep -o '/assets/[A-Za-z0-9._-]*\.js' | head -1)"
if [ -n "$asset" ]; then
  case "$(headers_of "$WEB_URL$asset" | tr 'A-Z' 'a-z')" in
    *immutable*) pass "hashed assets are cached hard ($asset)" ;;
    *) fail "hashed assets are not marked immutable ($asset)" ;;
  esac
else
  fail "no hashed asset referenced by index.html"
fi

# A missing asset must 404. If the SPA fallback reached /assets/ too, a stale
# bundle reference would return index.html as JavaScript, and the page would
# fail on a MIME type error that names nothing useful.
code="$(status_of "$WEB_URL/assets/does-not-exist.js")"
if [ "$code" = "404" ]; then
  pass "a missing asset is a 404, not the index page"
else
  fail "a missing asset returned HTTP $code"
fi

printf '\n%d failure(s)\n' "$failures"
[ "$failures" -eq 0 ]
