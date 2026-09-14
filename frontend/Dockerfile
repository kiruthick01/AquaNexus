# Frontend image: build with Node, serve the static bundle with nginx.
#
# The API base URL is set at RUN time, not build time: the entrypoint writes
# /config.js from $API_BASE_URL and the page reads it at load. One image serves
# any environment.
#
#     docker run -e API_BASE_URL=https://api.example.com aquanexus-frontend
#
# It is resolved by the browser, so it must be an address the user can reach.
#
# NOT BUILT OR RUN: Docker is not installed on the development machine. See
# tests/test_deployment.py for what is verified without a daemon, and DEVLOG.md
# for what remains unverified.

FROM node:22-alpine AS build

WORKDIR /app

# Dependencies first, so a source edit does not reinstall them.
COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig*.json vite.config.ts index.html ./
COPY public/ ./public/
COPY src/ ./src/

# No VITE_API_BASE_URL here: baking one in would defeat the runtime config
# above, since import.meta.env wins over it by design for local development.
RUN npm run build


FROM nginx:1.27-alpine AS serve

# The app is a single-page router: /predict is a client route, not a file, so
# nginx must fall back to index.html rather than answering 404.
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY docker-entrypoint.sh /entrypoint.sh
COPY --from=build /app/dist /usr/share/nginx/html

RUN chmod +x /entrypoint.sh

EXPOSE 80

ENTRYPOINT ["/entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]

# BusyBox wget, not GNU wget: this image is Alpine. It has --spider and -q and
# does NOT have --tries, which it rejects with "unrecognized option" - so the
# probe was failing on its own arguments while nginx served every request
# correctly, and the container reported unhealthy from the day it was written.
# Found by scripts/check_containers.sh on its first run; the run after that
# found the second half of it. `listen 80` is IPv4 only, and "localhost"
# resolves to ::1 before 127.0.0.1 here, so the probe was refused by a
# server that was answering every request on the published port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q -T 3 --spider http://127.0.0.1/ || exit 1
