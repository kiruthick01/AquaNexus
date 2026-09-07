# Frontend image: build with Node, serve the static bundle with nginx.
#
# The API base URL is baked in at build time - Vite substitutes import.meta.env
# during the bundle, so it cannot be changed by an environment variable at
# runtime. Pass it as a build argument:
#
#     docker build --build-arg VITE_API_BASE_URL=https://api.example.com .
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

ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build


FROM nginx:1.27-alpine AS serve

# The app is a single-page router: /predict is a client route, not a file, so
# nginx must fall back to index.html rather than answering 404.
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1
