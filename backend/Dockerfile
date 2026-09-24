# syntax=docker/dockerfile:1
#
# EcoSort backend image.
#
# IMPORTANT: the build context is the REPOSITORY ROOT (compose sets `context: .` with
# `dockerfile: backend/Dockerfile`), because the image bakes in the pretrained MobileNetV2
# and SSDLite detector weights using scripts/fetch-*.mjs, which live outside backend/.

##############################################################################
# Stage 1 - dependencies (carries a C++ toolchain; never shipped)
##############################################################################
FROM node:22-bookworm-slim AS deps

# better-sqlite3 publishes prebuilt binaries for Node 22, but they are not guaranteed for
# every platform/libc pair. Keeping a real toolchain in this stage lets node-gyp compile it
# from source as a fallback, while the runtime stage below stays slim.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      python3 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/package.json ./package.json
# The glob keeps this COPY valid when no lockfile has been committed yet.
COPY backend/package-lock.json* ./

# `npm ci` requires a lockfile; fall back to `npm install` so a fresh clone still builds.
RUN if [ -f package-lock.json ]; then \
      npm ci --omit=dev; \
    else \
      echo "no package-lock.json found - falling back to npm install" && npm install --omit=dev; \
    fi \
 && npm cache clean --force

##############################################################################
# Stage 2 - runtime
##############################################################################
FROM node:22-bookworm-slim AS runtime

# Defaults mirror docs/ARCHITECTURE.md section 7. Compose and .env override them.
ENV NODE_ENV=development
ENV BACKEND_PORT=4000
ENV DATA_DIR=/data
ENV DB_FILE=${DATA_DIR}/ecosort.db
ENV MODELS_DIR=/models
ENV BUNDLED_MODELS_DIR=/app/bundled-models
ENV CORS_ORIGIN=http://localhost:5173
ENV MAX_IMAGE_DATA_URL=400000
ENV LOG_LEVEL=info
ENV DEFAULT_REGION=us-generic

WORKDIR /app

COPY --from=deps --chown=node:node /app/node_modules ./node_modules
COPY --chown=node:node backend/package.json ./package.json
COPY --chown=node:node backend/src ./src
COPY --chown=node:node scripts ./scripts

# /data holds the SQLite file, /models is the bind-mounted host directory (it must exist
# and be traversable even when nothing is mounted over it), /app/bundled-models receives
# the baked-in fallback model below.
RUN mkdir -p "$DATA_DIR" "$MODELS_DIR" "$BUNDLED_MODELS_DIR" \
 && chown node:node "$DATA_DIR" "$MODELS_DIR" "$BUNDLED_MODELS_DIR"

USER node

# Bake the pretrained fallback model into the image so the app works with no network at
# runtime. A failure here must NOT break the build: offline builds are supported, and the
# API then reports fallback.available=false with the `make fetch-models` remedy.
RUN node scripts/fetch-mobilenet.mjs --dest /app/bundled-models --quiet \
 || echo "WARNING: pretrained MobileNetV2 download failed; run 'make fetch-models' on the host"

# Same deal for the object detector behind the Live scan tab. It lands in
# bundled-models/detectors/, which the classifier registry never lists (see
# scripts/fetch-detector.mjs), and the tab explains the remedy when it is missing.
RUN node scripts/fetch-detector.mjs --dest /app/bundled-models --quiet \
 || echo "WARNING: object detector download failed; run 'make fetch-models' on the host"

EXPOSE 4000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.BACKEND_PORT||4000)+'/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "src/server.js"]
