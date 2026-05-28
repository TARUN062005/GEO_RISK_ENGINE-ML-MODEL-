# ============================================================================
# Geo Risk Engine — Production Dockerfile (Log16)
#
# Log16: Lightweight build — NO transformers, NO torch, NO CUDA.
#   - Removed HuggingFace model pre-caching (saves ~1.5GB)
#   - Removed torch/transformers dependencies (saves ~800MB)
#   - spaCy en_core_web_sm is the only ML model (~15MB)
#   - Target: <800MB image (down from ~3GB)
#   - Runtime memory: <350MB (fits Render Free Tier 512MB)
#
# Multi-stage build:
#   Stage 1 (builder): Install deps + pre-cache spaCy model
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

# Pre-download spaCy model (the ONLY ML model needed)
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -m spacy download en_core_web_sm


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
    PYTORCH_NO_CUDA=1

WORKDIR /app

# Install ONLY runtime system dependencies (no build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy spaCy model data
COPY --from=builder /usr/local/lib/python3.11/site-packages/en_core_web_sm /usr/local/lib/python3.11/site-packages/en_core_web_sm

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app \
    && mkdir -p /app/.quota_state \
    && chown -R appuser:appuser /app/.quota_state
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
