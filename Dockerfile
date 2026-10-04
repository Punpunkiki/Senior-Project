# syntax=docker/dockerfile:1
#
# One image that serves both the API and the website, so there is a single
# deployment and the LIFF page sits on the same origin as /api (no CORS).
#
#   docker build -t mor-durian .
#   docker run --rm -p 8080:8080 --env-file .env mor-durian
#
# The trained checkpoint is baked in from outputs/ at build time. See
# docs/DEPLOY.md for running without one, and for the Cloud Run specifics.

# --------------------------------------------------------------------------
# Stage 1 — build the static website
# --------------------------------------------------------------------------
FROM node:22-slim AS web

WORKDIR /build

# Dependencies first so a content-only change does not reinstall them.
COPY web/package.json web/package-lock.json ./web/
RUN cd web && npm ci

# The site reads the knowledge base at build time (web/lib/diseases.ts), so
# this has to be present before `next build` runs.
COPY app/data ./app/data
COPY web ./web

ARG NEXT_PUBLIC_LINE_ID=""
ARG NEXT_PUBLIC_LIFF_ID=""
# Left empty on purpose: FastAPI serves this site from the same origin, so
# API calls stay relative. Only set it if the site is hosted separately.
ARG NEXT_PUBLIC_API_BASE=""
ENV NEXT_PUBLIC_LINE_ID=$NEXT_PUBLIC_LINE_ID \
    NEXT_PUBLIC_LIFF_ID=$NEXT_PUBLIC_LIFF_ID \
    NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE

RUN cd web && npm run build          # -> web/out

# --------------------------------------------------------------------------
# Stage 2 — runtime
# --------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch. The default PyPI wheel pulls roughly 2 GB of CUDA libraries
# that a CPU container never loads, and mixing the two wheel families is what
# produced an unimportable torch during development.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.3.1 torchvision==0.18.1

# Everything else. The torch pins above already satisfy this file's range, so
# pip leaves them alone.
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src ./src
COPY app ./app
COPY config.yaml ./config.yaml

# The trained checkpoint, when there is one. The directory always exists in
# the repo, so this never fails the build; without best.pt inside, the bot
# starts and answers "ระบบยังไม่พร้อมใช้งาน" to every photo rather than
# guessing from an untrained backbone.
COPY outputs ./outputs

COPY --from=web /build/web/out ./web/out

# Writable paths default to /tmp because that is the only writable location
# on Cloud Run. Both are ephemeral: mount a volume to keep them.
ENV DATABASE_URL="sqlite:////tmp/durian.db" \
    UPLOAD_DIR="/tmp/uploads" \
    PORT=8080

RUN useradd --create-home --uid 10001 durian && chown -R durian:durian /app
USER durian

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/health', timeout=4).status==200 else 1)"

# Shell form so ${PORT} expands: Cloud Run injects its own value.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
