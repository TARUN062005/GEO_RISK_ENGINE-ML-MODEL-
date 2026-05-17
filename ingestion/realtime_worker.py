"""
ingestion/realtime_worker.py
----------------------------
Real-Time Continuous Ingestion Worker (Log5, hardened Log15)

Replaces batch-only ingestion with a continuous hybrid system:
  - Fetches from GDELT, RSS feeds, and News APIs
  - Runs every 2-5 minutes (configurable)
  - Deduplicates by URL hash across all sources
  - Verifies source credibility before storage
  - Tags events with geo zone matches
  - Stores enriched events with full provenance

Log15: Production hardening.
  - Quota-aware scheduling (per-source intervals + daily limits)
  - Source-specific fetch intervals (RSS: 3min, GDELT: 15min, APIs: 30min)
  - Structured quota/health logging
  - Feed health monitoring
  - Graceful skip when quotas exhausted

Does NOT replace ingestion/worker.py — extends it.
The existing ingest_batch() remains available for manual/cron use.

Usage:
    python -m ingestion.realtime_worker
    python -m ingestion.realtime_worker --interval 120 --once
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import signal
import sys
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

# Ensure project root is on sys.path so this works as both:
#   python ingestion/realtime_worker.py        (script)
#   python -m ingestion.realtime_worker        (module)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ingestion.gdelt import fetch_latest_events as fetch_gdelt_events, RawEvent
from ingestion.sources.rss_feeds import fetch_rss_events
from ingestion.sources.newsapi import fetch_news_api_events
from ingestion.verification import verify_source, batch_verify
from ingestion.normalize import resolve_coordinates
from ingestion.relevance_filter import is_geopolitically_relevant, is_ml_relevant
from ingestion.clustering import cluster_incidents
from ml.inference.pipeline import run_ml_inference, run_ml_inference_batch
from storage.schema import EnrichedEvent, GeoPoint, MLAnnotation, to_mongo_doc
from core.geo.zones import match_event_to_zones
from core import metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

try:
    from config.settings import (
        INGEST_INTERVAL_SECONDS,
        MONGO_COLLECTION as SETTINGS_MONGO_COLLECTION,
        MONGO_DB as SETTINGS_MONGO_DB,
        MONGO_URI as SETTINGS_MONGO_URI,
    )
except Exception:
    INGEST_INTERVAL_SECONDS = 180
    SETTINGS_MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    SETTINGS_MONGO_DB = os.environ.get("MONGO_DB", "geo_risk")
    SETTINGS_MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "geo_events")

DEFAULT_INTERVAL_SECONDS = INGEST_INTERVAL_SECONDS
MONGO_URI = SETTINGS_MONGO_URI
MONGO_DB = SETTINGS_MONGO_DB
MONGO_COLLECTION = SETTINGS_MONGO_COLLECTION

_shutdown_event = asyncio.Event()


def _is_valid_raw_event(ev: dict) -> bool:
    """Reject malformed or stale source records before verification/ML."""
    text = (ev.get("text") or "").strip()
    if len(text) < 20:
        return False
    url = ev.get("source_url") or ""
    if url and not (url.startswith("http://") or url.startswith("https://")):
        return False
    published_at = ev.get("published_at")
    if isinstance(published_at, datetime):
        try:
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
            from config.settings import MAX_EVENT_AGE_HOURS
            if datetime.now(timezone.utc) - published_at > timedelta(hours=MAX_EVENT_AGE_HOURS):
                return False
        except Exception:
            pass
    return True


# ---------------------------------------------------------------------------
# Multi-Source Fetch — Log15: Quota-Aware
# ---------------------------------------------------------------------------

async def _fetch_all_sources(max_per_source: int = 50) -> list[dict]:
    """
    Fetch events from all configured sources in parallel.

    Log15: Quota-aware fetching.
      - Each source checks its own quota before fetching
      - Sources are skipped gracefully when exhausted
      - Per-source intervals prevent over-fetching
      - Structured logging of quota state
    """
    from ingestion.quota_manager import get_quota_manager

    qm = get_quota_manager()
    results: list[dict] = []
    seen_hashes: set[str] = set()

    # Log15: Log quota state at start of cycle
    logger.info("Quota state: %s", qm.status_summary())

    # --- 1. GDELT (DOC API — Log11, hardened Log15) ---
    # GDELT checks its own quota internally via gdelt.py
    try:
        gdelt_raw = await fetch_gdelt_events(max_events=max_per_source)
        for raw in gdelt_raw:
            url_hash = hashlib.sha256(
                (raw.event_id + raw.text[:100]).encode()
            ).hexdigest()[:16]

            if url_hash not in seen_hashes:
                seen_hashes.add(url_hash)
                metrics.inc("dedup_cache_misses")
                results.append({
                    "event_id": raw.event_id,
                    "text": raw.text,
                    "source": "gdelt",
                    "source_url": raw.source_url,
                    "publisher": raw.publisher,
                    "image_url": raw.image_url,
                    "published_at": raw.published_at,
                    "lat": raw.lat,
                    "lon": raw.lon,
                    "country_code": raw.country_code,
                })
            else:
                metrics.inc("dedup_cache_hits")
    except Exception as exc:
        logger.warning("[GDELT] fetch failed (non-fatal): %s", exc)

    # --- 2. RSS Feeds (Log15: with feed health monitoring) ---
    # RSS has no quota limit — always fetch
    if qm.can_fetch("rss"):
        try:
            rss_events = await fetch_rss_events(max_per_feed=max_per_source // 5)
            qm.record_request("rss")
            for ev in rss_events:
                url_hash = hashlib.sha256(ev["event_id"].encode()).hexdigest()[:16]
                if url_hash not in seen_hashes:
                    seen_hashes.add(url_hash)
                    metrics.inc("dedup_cache_misses")
                    results.append(ev)
                else:
                    metrics.inc("dedup_cache_hits")
        except Exception as exc:
            logger.warning("[RSS] fetch failed: %s", exc)
    else:
        qm.log_skip("rss")

    # --- 3. News APIs (Log15: quota-aware — handled internally by newsapi.py) ---
    api_count_before = len(results)
    try:
        api_events = await fetch_news_api_events(max_total=max_per_source)
        for ev in api_events:
            url_hash = hashlib.sha256(ev["event_id"].encode()).hexdigest()[:16]
            if url_hash not in seen_hashes:
                seen_hashes.add(url_hash)
                metrics.inc("dedup_cache_misses")
                results.append(ev)
            else:
                metrics.inc("dedup_cache_hits")
    except Exception as exc:
        logger.warning("[NewsAPI/GNews] fetch failed: %s", exc)

    api_count = len(results) - api_count_before

    # Log9: Per-source breakdown
    source_counts = {}
    for r in results:
        src = r.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    logger.info(
        "Multi-source fetch: %d total events %s | Quotas: %s",
        len(results), source_counts, qm.status_summary(),
    )
    return results


# ---------------------------------------------------------------------------
# Enrichment Pipeline (extends _enrich_event from worker.py)
# ---------------------------------------------------------------------------

def _enrich_event_realtime(ev_dict: dict) -> Optional[EnrichedEvent]:
    """
    Synchronous enrichment with verification + zone tagging.
    Runs in thread pool via asyncio.to_thread().

    Steps:
      1. Verify source credibility
      2. Run ML inference (classify + NER + intensity)
      3. Resolve coordinates
      4. Match to geo zones
      5. Build EnrichedEvent with full metadata
    """
    event_id = ev_dict.get("event_id", str(uuid.uuid4()))
    text = ev_dict.get("text", "")

    if not text or len(text) < 20:
        return None

    # Step 0 (Log8): Pre-ML relevance filter — reject celebrity/sports/entertainment
    if not is_geopolitically_relevant(text):
        logger.debug("[%s] Rejected: not geopolitically relevant.", event_id)
        return None

    # Step 1: Source verification
    verification = verify_source(
        source_url=ev_dict.get("source_url", ""),
        publisher=ev_dict.get("publisher", ""),
        image_url=ev_dict.get("image_url"),
        published_at=ev_dict.get("published_at"),
    )

    # Step 2: ML inference
    ml_annotation = run_ml_inference(event_id=event_id, text=text)

    # Step 2b (Log8): Post-ML relevance check — reject high-confidence "safe" articles
    if not is_ml_relevant(ml_annotation.label, ml_annotation.label_confidence):
        logger.debug("[%s] Rejected: ML classified as safe (conf=%.2f).", event_id, ml_annotation.label_confidence)
        return None

    # Step 3: Resolve coordinates (Log7: pass raw_text for semantic geo tagging)
    coords = resolve_coordinates(
        location_names=ml_annotation.location_names,
        feed_lat=ev_dict.get("lat"),
        feed_lon=ev_dict.get("lon"),
        raw_text=text,
    )
    if coords is None:
        logger.debug("No coordinates for event %s — skipping.", event_id)
        return None

    lon, lat = coords

    # Step 4: Match to geo zones
    matched_zones = match_event_to_zones(lat, lon)

    # Step 5: Build EnrichedEvent
    return EnrichedEvent(
        event_id=event_id,
        source=ev_dict.get("source", "unknown"),
        raw_text=text,
        published_at=ev_dict.get("published_at", datetime.now(timezone.utc)),
        location=GeoPoint(coordinates=[lon, lat]),
        country_code=ev_dict.get("country_code"),
        ml=ml_annotation,
        ingested_at=datetime.now(timezone.utc),
        schema_version="5",
        # Log5 extensions stored in the ML annotation + mongo doc
    )


# ---------------------------------------------------------------------------
# Single Ingestion Cycle
# ---------------------------------------------------------------------------

async def ingest_cycle(mongo_collection) -> dict:
    """
    Run one full ingestion cycle across all sources.

    Log11: Batch processing pipeline:
      1. Fetch all sources (quota-aware — Log15)
      2. Verify sources
      3. Pre-filter (cheap regex) — reject irrelevant before ML
      4. Batch ML inference (one forward pass)
      5. Post-filter (reject safe)
      6. Geocode + zone tag
      7. Upsert to MongoDB

    Returns stats dict.
    """
    import time as _time
    t0 = _time.time()
    stats = {"fetched": 0, "enriched": 0, "written": 0, "skipped": 0, "errors": 0}

    # ── Step 1: Fetch from all sources ─────────────────────────────────
    raw_events = await _fetch_all_sources()
    raw_events = [ev for ev in raw_events if _is_valid_raw_event(ev)]
    stats["fetched"] = len(raw_events)
    for ev in raw_events:
        src = ev.get("source", "unknown")
        stats[f"source_{src}"] = stats.get(f"source_{src}", 0) + 1

    # ── Step 2: Verify all sources ─────────────────────────────────────
    verified_events = batch_verify(raw_events)

    # ── Step 3: Pre-filter (cheap regex, no ML) ────────────────────────
    relevant_events = []
    for ev in verified_events:
        text = ev.get("text", "")
        if not text or len(text) < 20:
            stats["skipped"] += 1
            continue
        if not is_geopolitically_relevant(text):
            stats["skipped"] += 1
            continue
        relevant_events.append(ev)

    logger.info(
        "Pre-filter: %d → %d relevant (rejected %d)",
        len(verified_events), len(relevant_events), stats["skipped"],
    )

    if not relevant_events:
        logger.info("No relevant events to process.")
        return stats

    # ── Step 4: Batch ML inference (one transformer forward pass) ──────
    ml_items = [
        (ev.get("event_id", str(uuid.uuid4())), ev.get("text", ""))
        for ev in relevant_events
    ]
    t_ml = _time.time()
    ml_results = await asyncio.to_thread(run_ml_inference_batch, ml_items)
    ml_elapsed = _time.time() - t_ml
    metrics.inc("ml_batch_calls")
    metrics.inc("ml_events_classified", len(ml_items))
    metrics.record_timing("ml_batch_seconds", ml_elapsed)

    # ── Step 5-6: Post-filter + geocode → build docs ──────────────────
    enriched_docs: list[dict] = []

    for ev_dict, ml_annotation in zip(relevant_events, ml_results):
        try:
            event_id = ev_dict.get("event_id", str(uuid.uuid4()))
            text = ev_dict.get("text", "")

            # Post-ML relevance check
            if not is_ml_relevant(ml_annotation.label, ml_annotation.label_confidence):
                stats["skipped"] += 1
                continue

            # Geocode
            coords = resolve_coordinates(
                location_names=ml_annotation.location_names,
                feed_lat=ev_dict.get("lat"),
                feed_lon=ev_dict.get("lon"),
                raw_text=text,
            )
            if coords is None:
                stats["skipped"] += 1
                continue

            lon, lat = coords

            # Build EnrichedEvent
            enriched = EnrichedEvent(
                event_id=event_id,
                source=ev_dict.get("source", "unknown"),
                raw_text=text,
                published_at=ev_dict.get("published_at", datetime.now(timezone.utc)),
                location=GeoPoint(coordinates=[lon, lat]),
                country_code=ev_dict.get("country_code"),
                ml=ml_annotation,
                ingested_at=datetime.now(timezone.utc),
                schema_version="5",
            )

            stats["enriched"] += 1

            # Build mongo document
            doc = to_mongo_doc(enriched)
            doc["verification"] = ev_dict.get("verification", {})
            doc["source_url"] = ev_dict.get("source_url", "")
            doc["publisher"] = ev_dict.get("publisher", "")
            doc["image_url"] = ev_dict.get("image_url")

            # Zone tags
            if enriched.location and enriched.location.coordinates:
                lon, lat = enriched.location.coordinates
                doc["zones"] = match_event_to_zones(lat, lon)

            enriched_docs.append(doc)

        except Exception as exc:
            logger.warning("Failed to process event %s: %s", ev_dict.get("event_id"), exc)
            stats["errors"] += 1

    # ── Step 7: Log12 — Canonical incident clustering ─────────────────
    pre_cluster = len(enriched_docs)
    if pre_cluster > 1:
        enriched_docs = cluster_incidents(enriched_docs)
    post_cluster = len(enriched_docs)
    stats["clustered_from"] = pre_cluster
    stats["clustered_to"] = post_cluster
    metrics.log_clustering_stats(pre_cluster, post_cluster)

    # ── Step 8: Write to MongoDB ──────────────────────────────────────
    for doc in enriched_docs:
        try:
            await mongo_collection.update_one(
                {"_id": doc["_id"]},
                {"$setOnInsert": doc},
                upsert=True,
            )
            stats["written"] += 1
        except Exception as exc:
            logger.warning("Failed to write event %s: %s", doc.get("_id"), exc)
            stats["errors"] += 1

    elapsed = _time.time() - t0
    metrics.record_timing("ingestion_cycle_seconds", elapsed)
    metrics.log_cycle_stats(stats)

    # Log15: Log quota state after cycle
    try:
        from ingestion.quota_manager import get_quota_manager
        qm = get_quota_manager()
        quota_summary = qm.status_summary()
    except Exception:
        quota_summary = "unavailable"

    logger.info(
        "Ingestion cycle complete: fetched=%d enriched=%d clustered=%d→%d "
        "written=%d skipped=%d errors=%d (%.1fs) | Quotas: %s",
        stats["fetched"], stats["enriched"], pre_cluster, post_cluster,
        stats["written"], stats["skipped"], stats["errors"], elapsed,
        quota_summary,
    )
    return stats


# ---------------------------------------------------------------------------
# Continuous Worker Loop
# ---------------------------------------------------------------------------

async def run_continuous(
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    run_once: bool = False,
) -> None:
    """
    Run the ingestion worker continuously.

    Args:
        interval_seconds: Seconds between fetch cycles (default 180 = 3 min)
        run_once:         If True, run one cycle and exit (for testing)
    """
    from motor.motor_asyncio import AsyncIOMotorClient
    from config.settings import redact_secret, validate_environment

    validate_environment()

    # Log15: Initialize quota manager at startup
    from ingestion.quota_manager import get_quota_manager
    qm = get_quota_manager()

    client = AsyncIOMotorClient(MONGO_URI)
    collection = client[MONGO_DB][MONGO_COLLECTION]

    # Ensure indexes exist
    await _ensure_indexes(collection)

    logger.info(
        "Real-time ingestion worker started (interval=%ds, mongo=%s/%s) | Quotas: %s",
        interval_seconds, redact_secret(MONGO_URI), MONGO_DB,
        qm.status_summary(),
    )

    cycle_count = 0
    while not _shutdown_event.is_set():
        cycle_count += 1
        logger.info("=== Ingestion cycle #%d starting ===", cycle_count)

        try:
            stats = await ingest_cycle(collection)
            logger.info(
                "Cycle #%d stats: %s",
                cycle_count, stats,
            )
        except Exception as exc:
            logger.exception("Cycle #%d failed: %s", cycle_count, exc)

        if run_once:
            break

        # Wait for next cycle (interruptible)
        try:
            await asyncio.wait_for(
                _shutdown_event.wait(),
                timeout=interval_seconds,
            )
        except asyncio.TimeoutError:
            pass  # Normal — timeout means it's time for next cycle

    client.close()
    logger.info("Real-time ingestion worker stopped after %d cycles.", cycle_count)


async def _ensure_indexes(collection) -> None:
    """Create required indexes if they don't exist."""
    try:
        await collection.create_index([("location", "2dsphere")])

        # Log10: TTL index on ingested_at — auto-expire events after 72h
        # Drop legacy TTL index on published_at if it exists with different options
        try:
            await collection.drop_index("published_at_1")
        except Exception:
            pass
        await collection.create_index([("published_at", 1)])

        try:
            from config.settings import TTL_EXPIRY_SECONDS
            await collection.create_index(
                [("ingested_at", 1)],
                expireAfterSeconds=TTL_EXPIRY_SECONDS,
                name="ttl_ingested_72h",
            )
        except Exception:
            pass  # TTL index already exists

        await collection.create_index([("ml.label", 1), ("ml.intensity_score", -1)])
        await collection.create_index([("source", 1)])
        await collection.create_index([("canonical_event_id", 1)])
        await collection.create_index([("corroboration_count", -1)])
        await collection.create_index([("verification.credibility_score", -1)])
        await collection.create_index([("zones", 1)])

        # Log10: source_url for dedup (not unique — some events share URLs)
        await collection.create_index(
            [("source_url", 1)],
            name="idx_source_url",
        )
        logger.info("MongoDB indexes ensured (Log10: TTL + dedup).")
    except Exception as exc:
        logger.warning("Index creation note: %s", exc)


# ---------------------------------------------------------------------------
# Signal Handlers + Entry Point
# ---------------------------------------------------------------------------

def _handle_shutdown(sig, frame):
    logger.info("Shutdown signal received (%s). Stopping gracefully...", sig)
    _shutdown_event.set()


if __name__ == "__main__":
    import argparse
    from core.logging_config import configure_logging

    configure_logging()

    parser = argparse.ArgumentParser(description="Geo Risk Engine — Real-Time Ingestion Worker")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS,
                        help=f"Seconds between fetch cycles (default: {DEFAULT_INTERVAL_SECONDS})")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit (for testing)")
    parser.add_argument("--uri", default=MONGO_URI,
                        help="MongoDB URI")
    args = parser.parse_args()

    # Override globals from args
    MONGO_URI = args.uri

    # Register signal handlers
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    asyncio.run(run_continuous(
        interval_seconds=args.interval,
        run_once=args.once,
    ))
