# ============================================================================
# Geo Risk Engine — Production Dockerfile (Log16, hardened Log17)
#
# Log16: Lightweight build — NO transformers, NO torch, NO CUDA.
# Log17: Fixed spaCy model loading for Render deployment.
#   - spaCy model installed as pip package (not just downloaded)
#   - Ensures `import en_core_web_sm` works in runtime stage
#   - Separate API vs Worker CMD support
#   - No HTTP port required for worker mode
#   - Target: <800MB image, <200MB runtime memory
#
# Multi-stage build:
#   Stage 1 (builder): Install deps + pip install spaCy model
#   Stage 2 (runtime): Slim image with only runtime files
# ============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install build dependencies (only needed during compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# Log17: Install spaCy model as a proper pip package (not just `spacy download`)
# This ensures `import en_core_web_sm` works reliably in the runtime stage,
# even without spacy's model registry / symlinks.
RUN pip install --no-cache-dir --prefix=/install \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    # Log16: No HuggingFace models — disable HF entirely
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    # Log16: Disable torch — not installed
    PYTORCH_NO_CUDA=1 \
    # Log17: Memory optimization for Render Free Tier
    MALLOC_TRIM_THRESHOLD_=65536 \
    PYTHONMALLOC=malloc

WORKDIR /app

# Install ONLY runtime system dependencies (no build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder (includes spaCy model as package)
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app \
    && mkdir -p /app/.quota_state \
    && chown -R appuser:appuser /app/.quota_state
USER appuser

# Log17: Validate spaCy model is importable at build time
RUN python -c "import en_core_web_sm; nlp = en_core_web_sm.load(); print('spaCy model validated:', nlp.meta['name'])"

# Log17: No EXPOSE or HEALTHCHECK for worker mode.
# API mode uses: EXPOSE 8000 + uvicorn
# Worker mode uses: python -m ingestion.realtime_worker
# The correct CMD is set by docker-compose or Render service config.

# Default CMD: API mode (overridden by docker-compose for worker)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
