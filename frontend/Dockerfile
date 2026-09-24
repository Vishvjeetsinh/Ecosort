# syntax=docker/dockerfile:1
#
# EcoSort frontend image. Build context is ./frontend.
#
# Targets:
#   dev   - Vite dev server with HMR (used by docker-compose.yml)
#   build - produces the static bundle in /app/dist
#   prod  - nginx serving that bundle and proxying /api + /models to the backend

##############################################################################
# Stage 1 - shared dependency layer
##############################################################################
FROM node:22-bookworm-slim AS base

WORKDIR /app

COPY package.json ./package.json
# The glob keeps this COPY valid when no lockfile has been committed yet.
COPY package-lock.json* ./

# `npm ci` requires a lockfile; fall back to `npm install` so a fresh clone still builds.
# devDependencies are required here - vite and tailwind are both dev deps.
RUN if [ -f package-lock.json ]; then \
      npm ci; \
    else \
      echo "no package-lock.json found - falling back to npm install" && npm install; \
    fi

##############################################################################
# Stage 2 - development server
##############################################################################
FROM base AS dev

ENV NODE_ENV=development

COPY . .

EXPOSE 5173

# --host 0.0.0.0 is required for the port to be reachable from outside the container.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]

##############################################################################
# Stage 3 - production bundle
##############################################################################
FROM base AS build

ENV NODE_ENV=production

COPY . .

RUN npm run build

##############################################################################
# Stage 4 - production static server
##############################################################################
FROM nginx:1.27-alpine AS prod

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
  CMD wget --quiet --spider --tries=1 http://127.0.0.1/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
