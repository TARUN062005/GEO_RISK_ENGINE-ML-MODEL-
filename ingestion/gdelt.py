"""
ingestion/gdelt.py
------------------
GDELT Feed Fetcher (Log3, rewritten Log11, hardened Log15)

Log11: Replaced fragile raw GKG CSV parsing with GDELT DOC 2.0 API.
The DOC API returns structured JSON — no CSV parsing issues.

Log15: Production-grade rate limit handling.
  - Exponential backoff with jitter (via rate_limiter module)
  - Per-source cooldown windows after repeated 429s
  - Retry caps with graceful degradation
  - Quota-aware scheduling integration
  - Structured logging for observability

If GDELT fails for any reason, returns empty list (never crashes pipeline).
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx  # type: ignore

logger = logging.getLogger(__name__)

# GDELT DOC 2.0 API — structured JSON, no CSV parsing
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Log15: Rate limit configuration
MAX_RETRIES = 5
BASE_BACKOFF = 3.0       # seconds
MAX_BACKOFF = 120.0      # 2 minutes max
JITTER_FACTOR = 0.5
COOLDOWN_AFTER_429S = 3  # trigger cooldown after N consecutive 429s
COOLDOWN_DURATION = 600  # 10 minutes cooldown


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
    Log15: Production-grade rate limit handling with exponential backoff + jitter.

    Returns up to `max_events` RawEvent objects.
    """
    from ingestion.rate_limiter import get_rate_limiter
    from ingestion.quota_manager import get_quota_manager

    limiter = get_rate_limiter(
        "gdelt",
        base_delay=BASE_BACKOFF,
        max_delay=MAX_BACKOFF,
        max_retries=MAX_RETRIES,
        cooldown_after_failures=COOLDOWN_AFTER_429S,
        cooldown_duration=COOLDOWN_DURATION,
    )
    qm = get_quota_manager()

    events: list[RawEvent] = []

    # Log15: Check quota before fetching
    if not qm.can_fetch("gdelt"):
        qm.log_skip("gdelt")
        return events

    # Log15: Check rate limiter cooldown
    if limiter.is_in_cooldown:
        logger.info(
            "[GDELT] Rate limiter in cooldown (%.0fs remaining). Skipping.",
            limiter.cooldown_remaining,
        )
        return events

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

            # Log15: Exponential backoff with jitter for 429 handling
            data = None
            for attempt in range(MAX_RETRIES):
                try:
                    resp = await client.get(GDELT_DOC_API, params=params)

                    if resp.status_code == 429:
                        wait = limiter.record_429()

                        if not limiter.should_retry:
                            logger.warning(
                                "[GDELT DOC] Rate limit: max retries exhausted or in cooldown. "
                                "Giving up for this cycle."
                            )
                            qm.record_failure("gdelt")
                            return []

                        # Check Retry-After header
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait = max(wait, float(retry_after))
                            except ValueError:
                                pass

                        logger.warning(
                            "[GDELT DOC] 429 rate limited (attempt %d/%d). "
                            "Backing off %.1fs with jitter.",
                            attempt + 1, MAX_RETRIES, wait,
                        )
                        await asyncio.sleep(wait)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    limiter.record_success()
                    break

                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429:
                        # Already handled above, but just in case
                        wait = limiter.record_429()
                        if not limiter.should_retry:
                            qm.record_failure("gdelt")
                            return []
                        await asyncio.sleep(wait)
                        continue

                    # Non-429 HTTP error
                    limiter.record_other_error()
                    wait = limiter.compute_backoff(attempt)
                    logger.warning(
                        "[GDELT DOC] HTTP %d on attempt %d/%d. Retrying in %.1fs.",
                        exc.response.status_code, attempt + 1, MAX_RETRIES, wait,
                    )
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(wait)
                        continue
                    raise

                except (httpx.TimeoutException, httpx.ConnectError) as exc:
                    limiter.record_other_error()
                    wait = limiter.compute_backoff(attempt)
                    logger.warning(
                        "[GDELT DOC] Connection error on attempt %d/%d: %s. Retrying in %.1fs.",
                        attempt + 1, MAX_RETRIES, type(exc).__name__, wait,
                    )
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(wait)
                        continue
                    raise

            if data is None:
                logger.warning("[GDELT DOC] All retries exhausted — no data returned.")
                qm.record_failure("gdelt")
                return []

            # Record successful API request
            qm.record_request("gdelt", count=1)

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

            logger.info(
                "[GDELT DOC] %d articles fetched. Rate limiter: %s",
                len(articles), limiter.to_dict(),
            )

    except Exception as exc:
        logger.warning("[GDELT] fetch failed (non-fatal): %s", exc)
        try:
            qm.record_failure("gdelt")
        except Exception:
            pass

    logger.info("[GDELT] Total: %d events via DOC API", len(events))
    return events
