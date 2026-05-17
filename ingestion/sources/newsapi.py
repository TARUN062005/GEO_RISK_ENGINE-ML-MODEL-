"""
ingestion/sources/newsapi.py
----------------------------
NewsAPI / GNews Fetcher (Log5, upgraded Log15)

Fetches geopolitical news from:
  - NewsAPI.org (requires API key, free tier = 100 req/day)
  - GNews.io (requires API key, free tier = 100 req/day)

Log15: Quota-aware fetching.
  - Checks QuotaManager before each API call
  - Records requests against rolling 24h quota
  - Skips sources gracefully when exhausted
  - Rotates queries to maximize coverage per request

API keys are read from environment variables:
  NEWSAPI_KEY
  GNEWS_KEY

If a key is missing, that source is silently skipped.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Log9: Import keys from centralized settings (guarantees .env is loaded)
def _get_newsapi_key() -> str:
    try:
        from config.settings import NEWSAPI_KEY
        return NEWSAPI_KEY
    except ImportError:
        return os.environ.get("NEWSAPI_KEY", "")

def _get_gnews_key() -> str:
    try:
        from config.settings import GNEWS_KEY
        return GNEWS_KEY
    except ImportError:
        return os.environ.get("GNEWS_KEY", "")

# ---------------------------------------------------------------------------
# Geopolitical search queries — rotate per fetch cycle
# ---------------------------------------------------------------------------

GEO_QUERIES = [
    "conflict war military",
    "sanctions embargo trade war",
    "terrorism attack bombing",
    "protest civil unrest",
    "natural disaster earthquake flood",
    "shipping maritime piracy",
    "airspace closure flight ban",
    "border dispute territorial",
]

# Log15: Track query rotation index across cycles
_query_rotation_index: int = 0


def _get_rotated_query() -> str:
    """Get the next query in rotation (cycles through all queries over time)."""
    global _query_rotation_index
    query = GEO_QUERIES[_query_rotation_index % len(GEO_QUERIES)]
    _query_rotation_index += 1
    return query


async def _fetch_newsapi(query: str, max_results: int = 25) -> list[dict]:
    """Fetch from NewsAPI.org /v2/everything endpoint."""
    api_key = _get_newsapi_key()
    if not api_key:
        logger.debug("[NewsAPI] No NEWSAPI_KEY configured — skipping.")
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": min(max_results, 100),
        "apiKey": api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("articles", [])
        events = []
        for art in articles:
            source_name = art.get("source", {}).get("name", "Unknown")
            url_str = art.get("url", "")
            url_hash = hashlib.sha256(url_str.encode()).hexdigest()[:16]

            title = art.get("title", "") or ""
            desc = art.get("description", "") or ""
            text = f"{title}. {desc}".strip()

            events.append({
                "event_id": f"newsapi-{url_hash}",
                "text": text[:1500],
                "source_url": url_str,
                "publisher": source_name,
                "image_url": art.get("urlToImage"),
                "published_at": datetime.now(timezone.utc),
                "source": "newsapi",
            })

        logger.info("[NewsAPI] query='%s': %d articles", query, len(events))
        return events

    except Exception as exc:
        logger.warning("[NewsAPI] Failed for query '%s': %s", query, exc)
        return []


async def _fetch_gnews(query: str, max_results: int = 10) -> list[dict]:
    """Fetch from GNews.io /api/v4/search endpoint."""
    api_key = _get_gnews_key()
    if not api_key:
        logger.debug("[GNews] No GNEWS_KEY configured — skipping.")
        return []

    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "en",
        "max": min(max_results, 10),
        "token": api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("articles", [])
        events = []
        for art in articles:
            source_name = art.get("source", {}).get("name", "Unknown")
            url_str = art.get("url", "")
            url_hash = hashlib.sha256(url_str.encode()).hexdigest()[:16]

            title = art.get("title", "") or ""
            desc = art.get("description", "") or ""
            text = f"{title}. {desc}".strip()

            events.append({
                "event_id": f"gnews-{url_hash}",
                "text": text[:1500],
                "source_url": url_str,
                "publisher": source_name,
                "image_url": art.get("image"),
                "published_at": datetime.now(timezone.utc),
                "source": "gnews",
            })

        logger.info("[GNews] query='%s': %d articles", query, len(events))
        return events

    except Exception as exc:
        logger.warning("[GNews] Failed for query '%s': %s", query, exc)
        return []


async def fetch_news_api_events(max_total: int = 100) -> list[dict]:
    """
    Fetch from all configured news API sources.

    Log15: Quota-aware fetching.
      - Checks QuotaManager before each API call
      - Records requests against rolling 24h quota
      - Uses single rotated query per cycle (conserves quota)
      - Skips sources gracefully when exhausted
    """
    from ingestion.quota_manager import get_quota_manager

    qm = get_quota_manager()
    all_events: list[dict] = []
    seen_ids: set[str] = set()

    # Log15: Use ONE rotated query per cycle instead of 3
    # This cuts API requests from 6/cycle (3 queries × 2 providers) to 2/cycle
    query = _get_rotated_query()

    # --- NewsAPI ---
    if qm.can_fetch("newsapi"):
        try:
            newsapi_events = await _fetch_newsapi(query, max_results=25)
            qm.record_request("newsapi", count=1)
            for ev in newsapi_events:
                if ev["event_id"] not in seen_ids:
                    seen_ids.add(ev["event_id"])
                    all_events.append(ev)
        except Exception as exc:
            logger.warning("[NewsAPI] Fetch failed: %s", exc)
            qm.record_failure("newsapi")
    else:
        qm.log_skip("newsapi")

    # --- GNews ---
    if qm.can_fetch("gnews"):
        try:
            gnews_events = await _fetch_gnews(query, max_results=10)
            qm.record_request("gnews", count=1)
            for ev in gnews_events:
                if ev["event_id"] not in seen_ids:
                    seen_ids.add(ev["event_id"])
                    all_events.append(ev)
        except Exception as exc:
            logger.warning("[GNews] Fetch failed: %s", exc)
            qm.record_failure("gnews")
    else:
        qm.log_skip("gnews")

    logger.info(
        "[NewsAPIs] Total: %d unique events (query='%s') | Quotas: %s",
        len(all_events), query, qm.status_summary(),
    )
    return all_events[:max_total]
