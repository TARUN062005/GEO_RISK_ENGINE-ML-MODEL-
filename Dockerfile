# ============================================================================
# Geo Risk Engine — Production Dockerfile (Log15)
#
# Multi-stage build:
#   Stage 1 (builder): Install deps + pre-cache ML models
#   Stage 2 (runtime): Slim image with only runtime files
#
# Log15 optimizations:
#   - Multi-stage build to separate build/runtime deps
#   - HuggingFace models pre-downloaded during build (no runtime downloads)
#   - spaCy model pre-installed during build
#   - Removed build-essential from runtime
#   - Cleaned pip/apt caches aggressively
#   - Target: <1.5GB image (down from ~3GB)
# ============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache

WORKDIR /build

# Install build dependencies (only needed during compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download spaCy model
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -m spacy download en_core_web_sm

# Log15: Pre-cache HuggingFace models during build
# This eliminates runtime model downloads and reduces cold-start latency
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -c "\
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification; \
print('Downloading cross-encoder/nli-MiniLM2-L6-H768...'); \
AutoTokenizer.from_pretrained('cross-encoder/nli-MiniLM2-L6-H768'); \
AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-MiniLM2-L6-H768'); \
print('Downloading dslim/bert-base-NER...'); \
AutoTokenizer.from_pretrained('dslim/bert-base-NER'); \
AutoModel.from_pretrained('dslim/bert-base-NER'); \
print('All models pre-cached successfully.'); \
"


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

# Install ONLY runtime system dependencies (no build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /opt/hf_cache /opt/hf_cache

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
