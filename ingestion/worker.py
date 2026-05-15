"""
ingestion/worker.py
-------------------
Ingestion Worker — ML-Enriched (Log2)

Updated from Log1: The worker now calls ml.inference.pipeline.run_ml_inference()
before persisting each event. All ML computation happens HERE, never in the API.

Schedule: runs on a cron/APScheduler interval (e.g. every 15 minutes).
Scale: multiple worker replicas can run in parallel (stateless).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from ingestion.gdelt import fetch_latest_events, RawEvent
from ingestion.normalize import resolve_coordinates
from ml.inference.pipeline import run_ml_inference
from storage.schema import EnrichedEvent, GeoPoint, to_mongo_doc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core worker loop
# ---------------------------------------------------------------------------

async def ingest_batch(mongo_collection) -> int:
    """
    Fetch → ML annotate → store one batch of raw events.

    Returns:
        Number of events successfully written to MongoDB.
    """
    raw_events: list[RawEvent] = await fetch_latest_events()
    logger.info("Fetched %d raw events from GDELT.", len(raw_events))

    written = 0
    for raw in raw_events:
        try:
            enriched = await asyncio.to_thread(_enrich_event, raw)
            if enriched is None:
                continue

            doc = to_mongo_doc(enriched)
            await mongo_collection.update_one(
                {"_id": doc["_id"]},
                {"$setOnInsert": doc},
                upsert=True,
            )
            written += 1

        except Exception as exc:
            logger.warning("Failed to ingest event %s: %s", raw.event_id, exc)

    logger.info("Ingestion batch complete: %d/%d written.", written, len(raw_events))
    return written


def _enrich_event(raw: "RawEvent") -> "EnrichedEvent | None":
    """
    Synchronous enrichment — runs in a thread pool via asyncio.to_thread().

    Steps:
      1. Run ML inference (classify + NER + intensity)
      2. Resolve coordinates from NER or feed metadata
      3. Build EnrichedEvent with full MLAnnotation
    """
    # Step 1: ML inference (CPU-bound — safe in thread pool)
    ml_annotation = run_ml_inference(
        event_id=raw.event_id,
        text=raw.text,
    )

    # Step 2: Resolve geo coordinates
    coords = resolve_coordinates(
        location_names=ml_annotation.location_names,
        feed_lat=getattr(raw, "lat", None),
        feed_lon=getattr(raw, "lon", None),
    )
    if coords is None:
        logger.debug("No coordinates for event %s — skipping.", raw.event_id)
        return None

    lon, lat = coords
    return EnrichedEvent(
        event_id=raw.event_id or str(uuid.uuid4()),
        source=raw.source,
        raw_text=raw.text,
        published_at=raw.published_at,
        location=GeoPoint(coordinates=[lon, lat]),
        country_code=getattr(raw, "country_code", None),
        ml=ml_annotation,
        ingested_at=datetime.now(timezone.utc),
    )
