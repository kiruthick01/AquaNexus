#!/bin/sh
# Render the runtime configuration, then hand over to nginx.
#
# The bundle is built once and served anywhere: this writes /config.js from
# $API_BASE_URL at container start, so the same image can point at a local
# backend, a staging one, or production without a rebuild.
#
# The URL is resolved by the browser, not by this container, so it must be an
# address the *user* can reach - http://api:8000 works between containers and
# fails in the page.
set -eu

API_BASE_URL="${API_BASE_URL:-}"
CONFIG_PATH=/usr/share/nginx/html/config.js

cat > "$CONFIG_PATH" <<EOF
window.__AQUANEXUS_CONFIG__ = {
  apiBaseUrl: "${API_BASE_URL}",
};
EOF

if [ -z "$API_BASE_URL" ]; then
  echo "aquanexus-frontend: API_BASE_URL is unset; the page will fall back to" \
       "http://localhost:8000" >&2
else
  echo "aquanexus-frontend: API at ${API_BASE_URL}" >&2
fi

exec "$@"
