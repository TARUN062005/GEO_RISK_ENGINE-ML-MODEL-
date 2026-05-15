"""
config/settings.py
------------------
Centralized Configuration (Log7, updated Log10)

Loads settings from environment variables with safe defaults.
Supports .env files if python-dotenv is installed.
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

# Attempt to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        logger.debug("Loaded .env from %s", _env_path)
except ImportError:
    pass  # dotenv not installed — read from environment directly


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
MONGO_URI:        str = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB:         str = os.environ.get("MONGO_DB", "geo_risk")
MONGO_COLLECTION: str = os.environ.get("MONGO_COLLECTION", "geo_events")

# ---------------------------------------------------------------------------
# API keys (optional)
# ---------------------------------------------------------------------------
NEWSAPI_KEY: str = os.environ.get("NEWSAPI_KEY", "")
GNEWS_KEY:   str = os.environ.get("GNEWS_KEY", "")

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
INGEST_INTERVAL_SECONDS: int = int(os.environ.get("INGEST_INTERVAL_SECONDS", "180"))
MAX_EVENT_AGE_HOURS:     int = int(os.environ.get("MAX_EVENT_AGE_HOURS", "72"))
MAX_FRESHNESS_MINUTES:   int = int(os.environ.get("MAX_FRESHNESS_MINUTES", "10"))

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
ROUTE_BUFFER_KM:       float = float(os.environ.get("ROUTE_BUFFER_KM", "50"))
MIN_LABEL_CONFIDENCE:  float = float(os.environ.get("MIN_LABEL_CONFIDENCE", "0.50"))
ZONE_WEIGHT:           float = float(os.environ.get("ZONE_WEIGHT", "0.30"))
EVENT_WEIGHT:          float = float(os.environ.get("EVENT_WEIGHT", "0.70"))

# ---------------------------------------------------------------------------
# Log10: Time-decay + Intelligence
# ---------------------------------------------------------------------------
RECENCY_HALF_LIFE_DAYS:      float = float(os.environ.get("RECENCY_HALF_LIFE_DAYS", "1.0"))
CORROBORATION_BOOST:         float = float(os.environ.get("CORROBORATION_BOOST", "1.15"))
SINGLE_SOURCE_PENALTY:       float = float(os.environ.get("SINGLE_SOURCE_PENALTY", "0.90"))
TTL_EXPIRY_SECONDS:          int = int(os.environ.get("TTL_EXPIRY_SECONDS", "259200"))  # 72h

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Production API
# ---------------------------------------------------------------------------
APP_ENV: str = os.environ.get("APP_ENV", "development")
API_TIMEOUT_SECONDS: float = float(os.environ.get("API_TIMEOUT_SECONDS", "45"))
MAX_REQUEST_BYTES: int = int(os.environ.get("MAX_REQUEST_BYTES", "65536"))
CORS_ALLOW_ORIGINS: str = os.environ.get("CORS_ALLOW_ORIGINS", "*")
UVICORN_WORKERS: int = int(os.environ.get("UVICORN_WORKERS", "1"))


def redact_secret(value: str) -> str:
    """Return a safe preview for logs."""
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def validate_environment() -> None:
    """
    Lightweight startup validation. Optional API keys may be absent, but core
    deployment settings must be sane and secrets must not be logged raw.
    """
    if not MONGO_URI:
        raise ValueError("MONGO_URI is required")
    if ROUTE_BUFFER_KM <= 0:
        raise ValueError("ROUTE_BUFFER_KM must be positive")
    if not (0.0 <= MIN_LABEL_CONFIDENCE <= 1.0):
        raise ValueError("MIN_LABEL_CONFIDENCE must be between 0 and 1")
    if API_TIMEOUT_SECONDS <= 0:
        raise ValueError("API_TIMEOUT_SECONDS must be positive")
    if MAX_REQUEST_BYTES < 1024:
        raise ValueError("MAX_REQUEST_BYTES must be at least 1024")
    logger.info(
        "Config loaded: mongo=%s newsapi=%s gnews=%s",
        MONGO_URI.split("@")[-1],
        "set" if NEWSAPI_KEY else "unset",
        "set" if GNEWS_KEY else "unset",
    )
