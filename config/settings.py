"""
config/settings.py
------------------
Centralized Configuration (Log7, updated Log10, extended Log15)

Loads settings from environment variables with safe defaults.
Supports .env files if python-dotenv is installed.

Log15: Added quota-aware scheduling, rate limiting, and
       entity normalization configuration.
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
# MongoDB Helper Functions
# ---------------------------------------------------------------------------
def sanitize_mongo_uri(uri: str) -> str:
    """Return a version of the MongoDB URI safe for logs (password masked)."""
    if not uri:
        return ""
    import re
    # Mask password: matches "mongodb://username:password@host" or "mongodb+srv://username:password@host"
    return re.sub(r'(mongodb(?:\+srv)?://[^:]+:)([^@]+)(@)', r'\1*****\3', uri)


def get_mongo_host(uri: str) -> str:
    """Extract host and port from MongoDB URI for clean, safe logging."""
    if not uri:
        return "none"
    try:
        # Strip protocol
        rest = uri.split("://", 1)[-1]
        # Strip credentials if present
        if "@" in rest:
            rest = rest.split("@", 1)[-1]
        # Strip path and options
        host = rest.split("/", 1)[0].split("?", 1)[0]
        return host
    except Exception:
        return "unknown"


def validate_mongo_uri(uri: str) -> None:
    """Fail fast if the MongoDB URI is malformed or contains unconfigured placeholders."""
    if not uri:
        raise ValueError("CRITICAL: MONGO_URI is empty or not provided.")

    if not (uri.startswith("mongodb://") or uri.startswith("mongodb+srv://")):
        raise ValueError(
            f"CRITICAL: Invalid MongoDB URI protocol in: '{sanitize_mongo_uri(uri)}'. "
            "Must start with 'mongodb://' or 'mongodb+srv://'."
        )

    # Check for common placeholders
    placeholders = [
        "<username>", "<password>", "<your-password>", "<cluster>",
        "username:password", "your-cluster-url", "your-database",
        "cluster.mongodb.net"  # Default cluster placeholder host
    ]
    for ph in placeholders:
        if ph in uri.lower():
            raise ValueError(
                f"CRITICAL: MongoDB URI contains unconfigured placeholder '{ph}': "
                f"'{sanitize_mongo_uri(uri)}'. Please check your environment variables."
            )


# ---------------------------------------------------------------------------
# MongoDB Configuration
# ---------------------------------------------------------------------------
MONGO_DB:         str = os.environ.get("MONGO_DB", "geo_risk")
MONGO_COLLECTION: str = os.environ.get("MONGO_COLLECTION", "geo_events")
MONGO_URI:        str = os.environ.get("MONGO_URI", "mongodb://localhost:27017")


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
# Log15: Quota-Aware Scheduling
# ---------------------------------------------------------------------------
NEWSAPI_DAILY_QUOTA: int = int(os.environ.get("NEWSAPI_DAILY_QUOTA", "100"))  # Log16: hard limit
GNEWS_DAILY_QUOTA:   int = int(os.environ.get("GNEWS_DAILY_QUOTA", "100"))    # Log16: hard limit
GDELT_DAILY_QUOTA:   int = int(os.environ.get("GDELT_DAILY_QUOTA", "500"))

# Log15: Source-specific fetch intervals (seconds)
RSS_INTERVAL_SECONDS:     int = int(os.environ.get("RSS_INTERVAL_SECONDS", "180"))
GDELT_INTERVAL_SECONDS:   int = int(os.environ.get("GDELT_INTERVAL_SECONDS", "900"))
NEWSAPI_INTERVAL_SECONDS: int = int(os.environ.get("NEWSAPI_INTERVAL_SECONDS", "1800"))
GNEWS_INTERVAL_SECONDS:   int = int(os.environ.get("GNEWS_INTERVAL_SECONDS", "1800"))

# Log15: Quota exhaustion cooldown (seconds)
QUOTA_EXHAUSTION_COOLDOWN: int = int(os.environ.get("QUOTA_EXHAUSTION_COOLDOWN", "3600"))

# ---------------------------------------------------------------------------
# Log15: GDELT Rate Limiting
# ---------------------------------------------------------------------------
GDELT_MAX_RETRIES:          int   = int(os.environ.get("GDELT_MAX_RETRIES", "5"))
GDELT_BASE_BACKOFF:         float = float(os.environ.get("GDELT_BASE_BACKOFF", "3.0"))
GDELT_MAX_BACKOFF:          float = float(os.environ.get("GDELT_MAX_BACKOFF", "120.0"))
GDELT_COOLDOWN_DURATION:    float = float(os.environ.get("GDELT_COOLDOWN_DURATION", "600.0"))

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
    validate_mongo_uri(MONGO_URI)

    if ROUTE_BUFFER_KM <= 0:
        raise ValueError("ROUTE_BUFFER_KM must be positive")
    if not (0.0 <= MIN_LABEL_CONFIDENCE <= 1.0):
        raise ValueError("MIN_LABEL_CONFIDENCE must be between 0 and 1")
    if API_TIMEOUT_SECONDS <= 0:
        raise ValueError("API_TIMEOUT_SECONDS must be positive")
    if MAX_REQUEST_BYTES < 1024:
        raise ValueError("MAX_REQUEST_BYTES must be at least 1024")
    logger.info(
        "Config loaded: mongo=%s newsapi=%s gnews=%s "
        "quotas=[newsapi=%d/day gnews=%d/day gdelt=%d/day] "
        "intervals=[rss=%ds gdelt=%ds newsapi=%ds gnews=%ds]",
        get_mongo_host(MONGO_URI),
        "set" if NEWSAPI_KEY else "unset",
        "set" if GNEWS_KEY else "unset",
        NEWSAPI_DAILY_QUOTA, GNEWS_DAILY_QUOTA, GDELT_DAILY_QUOTA,
        RSS_INTERVAL_SECONDS, GDELT_INTERVAL_SECONDS,
        NEWSAPI_INTERVAL_SECONDS, GNEWS_INTERVAL_SECONDS,
    )
