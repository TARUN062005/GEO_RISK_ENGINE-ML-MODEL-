"""
core/metrics.py
---------------
Lightweight Pipeline Metrics (Log12, extended Log15)

Provides structured observability for the geo-intelligence pipeline.
No external dependencies (no Prometheus, no StatsD).
Thread-safe via simple dict + increment pattern.

Log15: Added metrics for:
  - API quota usage (per source)
  - Source failures + feed health
  - Ingestion latency breakdown
  - Geocoding failures
  - Entity normalization stats
  - Rate limiter state

Exposes:
  - ingestion counts (per source, per cycle)
  - ML inference counts
  - cache hit rates
  - clustering stats
  - runtime timings
  - error counts
  - quota usage (Log15)
  - feed health (Log15)
  - geocoding quality (Log15)

Used by:
  - ingestion/realtime_worker.py
  - core/orchestrator.py
  - app/main.py /metrics endpoint
"""

from __future__ import annotations

import time
import threading
import logging
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# Global counters
_counters: dict[str, int] = {
    # Ingestion
    "ingestion_cycles": 0,
    "events_fetched": 0,
    "events_enriched": 0,
    "events_written": 0,
    "events_skipped": 0,
    "events_errors": 0,
    # Clustering
    "clusters_total_input": 0,
    "clusters_total_output": 0,
    "clusters_duplicates_merged": 0,
    # ML
    "ml_batch_calls": 0,
    "ml_events_classified": 0,
    # Caches
    "geocode_cache_hits": 0,
    "geocode_cache_misses": 0,
    "dedup_cache_hits": 0,
    "dedup_cache_misses": 0,
    "route_cache_hits": 0,
    "route_cache_misses": 0,
    # Analysis
    "analyses_total": 0,
    # Sources
    "source_gdelt": 0,
    "source_rss": 0,
    "source_newsapi": 0,
    "source_gnews": 0,
    # Log15: Geocoding quality
    "geocode_attempts": 0,
    "geocode_failures": 0,
    # Log15: Entity normalization
    "entities_normalized": 0,
    "entities_rejected": 0,
    # Log15: Feed health
    "feeds_suppressed": 0,
    "feeds_recovered": 0,
    # Log15: Rate limiting
    "rate_limit_429s": 0,
    "rate_limit_cooldowns": 0,
}

_timings: dict[str, list[float]] = {
    "ingestion_cycle_seconds": [],
    "analysis_seconds": [],
    "ml_batch_seconds": [],
}

_startup_time: float = time.time()


def inc(key: str, amount: int = 1) -> None:
    """Increment a counter."""
    with _lock:
        _counters[key] = _counters.get(key, 0) + amount


def record_timing(key: str, seconds: float) -> None:
    """Record a timing measurement (keeps last 100)."""
    with _lock:
        bucket = _timings.setdefault(key, [])
        bucket.append(round(seconds, 3))
        if len(bucket) > 100:
            bucket[:] = bucket[-100:]


def get_metrics() -> dict[str, Any]:
    """Return all metrics as a JSON-serializable dict."""
    with _lock:
        timing_stats = {}
        for key, values in _timings.items():
            if values:
                timing_stats[key] = {
                    "count": len(values),
                    "avg_ms": round(sum(values) / len(values) * 1000, 1),
                    "last_ms": round(values[-1] * 1000, 1),
                    "max_ms": round(max(values) * 1000, 1),
                }
            else:
                timing_stats[key] = {"count": 0}

        result = {
            "uptime_seconds": round(time.time() - _startup_time, 1),
            "counters": dict(_counters),
            "timings": timing_stats,
        }

    # Log15: Include quota states if available
    try:
        from ingestion.quota_manager import get_quota_manager
        result["quotas"] = get_quota_manager().get_all_quotas()
    except Exception:
        pass

    # Log15: Include feed health if available
    try:
        from ingestion.feed_health import get_all_feed_health
        result["feed_health"] = get_all_feed_health()
    except Exception:
        pass

    # Log15: Include rate limiter states if available
    try:
        from ingestion.rate_limiter import get_all_rate_limit_states
        result["rate_limits"] = get_all_rate_limit_states()
    except Exception:
        pass

    return result


def log_cycle_stats(stats: dict) -> None:
    """Update metrics from an ingestion cycle stats dict."""
    inc("ingestion_cycles")
    inc("events_fetched", stats.get("fetched", 0))
    inc("events_enriched", stats.get("enriched", 0))
    inc("events_written", stats.get("written", 0))
    inc("events_skipped", stats.get("skipped", 0))
    inc("events_errors", stats.get("errors", 0))

    # Source breakdown
    for src in ("gdelt", "rss", "newsapi", "gnews"):
        count = stats.get(f"source_{src}", 0)
        if count:
            inc(f"source_{src}", count)


def log_clustering_stats(input_count: int, output_count: int) -> None:
    """Update clustering metrics."""
    inc("clusters_total_input", input_count)
    inc("clusters_total_output", output_count)
    inc("clusters_duplicates_merged", max(0, input_count - output_count))


def log_cache_info(name: str, before, after) -> None:
    """Record delta cache hits/misses from functools.lru_cache cache_info()."""
    inc(f"{name}_cache_hits", max(0, after.hits - before.hits))
    inc(f"{name}_cache_misses", max(0, after.misses - before.misses))
