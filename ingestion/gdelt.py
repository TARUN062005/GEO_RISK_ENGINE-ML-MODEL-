"""
ingestion/gdelt.py
------------------
GDELT Feed Fetcher (Log3, rewritten Log11)

Log11: Replaced fragile raw GKG CSV parsing with GDELT DOC 2.0 API.
The DOC API returns structured JSON — no CSV parsing issues.

If GDELT fails for any reason, returns empty list (never crashes pipeline).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx  # type: ignore

logger = logging.getLogger(__name__)

# GDELT DOC 2.0 API — structured JSON, no CSV parsing
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass
class RawEvent:
    event_id: str
    text: str
    published_at: datetime
    source: str = "gdelt"
    lat: Optional[float] = None
    lon: Optional[float] = None
    country_code: Optional[str] = None
    source_url: str = ""
    image_url: Optional[str] = None
    publisher: str = "GDELT Project"


async def fetch_latest_events(max_events: int = 25) -> list[RawEvent]:
    """
    Fetch recent geopolitical articles via GDELT DOC 2.0 API.

    Log11: Uses structured JSON API instead of raw CSV.
    Single combined query to minimize API calls (avoid 429).
    Returns up to `max_events` RawEvent objects.
    """
    events: list[RawEvent] = []

    # Single combined query — avoids double API calls and rate limits
    query = "(conflict OR military OR missile OR sanctions OR maritime OR shipping OR piracy)"

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            params = {
                "query": query,
                "mode": "ArtList",
                "maxrecords": str(max_events),
                "timespan": "180min",  # wider window, single call
                "format": "json",
                "sort": "DateDesc",
            }

            # Retry with backoff for 429
            import asyncio
            for attempt in range(3):
                try:
                    resp = await client.get(GDELT_DOC_API, params=params)
                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        logger.warning("[GDELT DOC] 429 rate limited, retrying in %ds...", wait)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except httpx.HTTPStatusError:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
            else:
                logger.warning("[GDELT DOC] All retries exhausted.")
                return []

            articles = data.get("articles", [])
            for i, article in enumerate(articles):
                title = article.get("title", "").strip()
                if not title or len(title) < 15:
                    continue

                url = article.get("url", "")
                img = article.get("socialimage", "") or None
                domain = article.get("domain", "")
                seendate = article.get("seendate", "")

                pub_dt = datetime.now(timezone.utc)
                if seendate:
                    try:
                        pub_dt = datetime.strptime(
                            seendate[:14], "%Y%m%d%H%M%S"
                        ).replace(tzinfo=timezone.utc)
                    except (ValueError, IndexError):
                        pass

                events.append(RawEvent(
                    event_id=f"gdelt-{hash(url) & 0xFFFFFFFF:08x}-{i}",
                    text=title[:1000],
                    published_at=pub_dt,
                    source="gdelt",
                    source_url=url,
                    image_url=img,
                    publisher=domain or "GDELT",
                ))

            logger.info("[GDELT DOC] %d articles fetched.", len(articles))

    except Exception as exc:
        logger.warning("[GDELT] fetch failed (non-fatal): %s", exc)

    logger.info("[GDELT] Total: %d events via DOC API", len(events))
    return events
