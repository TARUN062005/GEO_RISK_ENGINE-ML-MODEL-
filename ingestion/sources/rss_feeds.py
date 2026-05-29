"""
ingestion/sources/rss_feeds.py
------------------------------
RSS Feed Fetcher (Log5, upgraded Log15)

Fetches geopolitical news from major RSS feeds.

Log15: Production hardening.
  - Replaced broken feeds (Reuters DNS, AP 403, Maritime Executive 404)
  - Added feed health monitoring (dead-feed suppression)
  - Added source health logging
  - Resilient fallback: skip permanently dead sources
  - User-Agent header to avoid bot blocking

Returns RawEvent objects compatible with the existing ingestion pipeline.
Deduplicates by URL hash within a fetch cycle.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RSS Feed Configuration — Log15: Replaced broken feeds
# ---------------------------------------------------------------------------

RSS_FEEDS: list[dict[str, str]] = [
    # === General World News ===
    {
        "name": "New York Times World",
        "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "publisher": "The New York Times",
    },
    {
        "name": "BBC World",
        "url": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "publisher": "BBC News",
    },
    {
        "name": "Al Jazeera",
        "url": "https://www.aljazeera.com/xml/rss/all.xml",
        "publisher": "Al Jazeera",
    },
    # Log15: AP feed replaced (old rsshub.app URL returns 403)
    {
        "name": "Associated Press",
        "url": "https://feedx.net/rss/ap.xml",
        "publisher": "Associated Press",
    },
    {
        "name": "NPR World",
        "url": "https://feeds.npr.org/1004/rss.xml",
        "publisher": "NPR",
    },
    # Log15: Additional reliable global news feeds
    {
        "name": "France24",
        "url": "https://www.france24.com/en/rss",
        "publisher": "France 24",
    },
    {
        "name": "DW World",
        "url": "https://rss.dw.com/rdf/rss-en-world",
        "publisher": "Deutsche Welle",
    },

    # === Maritime-specific feeds ===
    {
        "name": "Splash247 Maritime",
        "url": "https://splash247.com/feed/",
        "publisher": "Splash 247",
    },
    # Log15: Maritime Executive feed replaced (old URL returns 404)
    {
        "name": "gCaptain Maritime",
        "url": "https://gcaptain.com/feed/",
        "publisher": "gCaptain",
    },
    {
        "name": "Seatrade Maritime",
        "url": "https://www.seatrade-maritime.com/rss.xml",
        "publisher": "Seatrade Maritime",
    },
    {
        "name": "FleetMon News",
        "url": "https://www.fleetmon.com/maritime-news/feed/",
        "publisher": "FleetMon",
    },
]

# User-Agent to avoid bot blocking on some feeds
_USER_AGENT = "GeoRiskEngine/1.0 (RSS Feed Aggregator; +https://github.com/geo-risk-engine)"


def _parse_rss_xml(xml_text: str, feed_config: dict) -> list[dict]:
    """
    Minimal RSS XML parser — no external dependency.
    Extracts <item> elements with <title>, <description>, <link>, <pubDate>.
    Falls back gracefully if fields are missing.
    """
    import re

    items = []
    # Find all <item>...</item> blocks
    item_pattern = re.compile(r"<item[^>]*>(.*?)</item>", re.DOTALL | re.IGNORECASE)

    for match in item_pattern.finditer(xml_text):
        block = match.group(1)

        def _extract(tag: str) -> str:
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, re.DOTALL | re.IGNORECASE)
            if m:
                text = m.group(1).strip()
                # Strip CDATA wrappers
                text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.DOTALL)
                # Strip HTML tags
                text = re.sub(r"<[^>]+>", "", text).strip()
                return text
            return ""

        title = _extract("title")
        description = _extract("description")
        link = _extract("link")
        pub_date = _extract("pubDate")

        # Extract image from media:content or enclosure
        img_match = re.search(r'<(?:media:content|enclosure)[^>]+url="([^"]+)"', block, re.IGNORECASE)
        image_url = img_match.group(1) if img_match else None

        # Also check for <media:thumbnail>
        if not image_url:
            img_match = re.search(r'<media:thumbnail[^>]+url="([^"]+)"', block, re.IGNORECASE)
            image_url = img_match.group(1) if img_match else None

        if title or description:
            items.append({
                "title": title,
                "description": description[:1000] if description else "",
                "link": link,
                "pub_date": pub_date,
                "image_url": image_url,
                "publisher": feed_config["publisher"],
            })

    return items


async def fetch_rss_events(max_per_feed: int = 30) -> list[dict]:
    """
    Fetch events from all configured RSS feeds.

    Log15: Integrated feed health monitoring.
      - Skips suppressed feeds (after repeated failures)
      - Tracks success/failure per feed
      - Logs feed health status

    Returns list of dicts with keys:
      event_id, text, source_url, publisher, image_url, published_at, source

    Deduplicates by URL hash.
    """
    from ingestion.feed_health import should_fetch_feed, get_feed_health

    all_events: list[dict] = []
    seen_hashes: set[str] = set()
    feed_stats = {"active": 0, "suppressed": 0, "failed": 0, "success": 0}

    headers = {"User-Agent": _USER_AGENT}

    async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers) as client:
        for feed in RSS_FEEDS:
            feed_name = feed["name"]

            # Log15: Check feed health before fetching
            if not should_fetch_feed(feed_name, feed["url"]):
                feed_stats["suppressed"] += 1
                continue

            feed_stats["active"] += 1
            health = get_feed_health(feed_name, feed["url"])

            try:
                resp = await client.get(feed["url"])
                resp.raise_for_status()

                items = _parse_rss_xml(resp.text, feed)
                health.record_success(items_count=len(items))
                feed_stats["success"] += 1
                logger.info("[RSS] %s: parsed %d items", feed_name, len(items))

                for item in items[:max_per_feed]:
                    # Deduplicate by URL hash
                    url_hash = hashlib.sha256(
                        (item.get("link", "") or item["title"]).encode()
                    ).hexdigest()[:16]

                    if url_hash in seen_hashes:
                        continue
                    seen_hashes.add(url_hash)

                    text = f"{item['title']}. {item['description']}" if item['description'] else item['title']

                    all_events.append({
                        "event_id": f"rss-{url_hash}",
                        "text": text[:1500],
                        "source_url": item.get("link", ""),
                        "publisher": item["publisher"],
                        "image_url": item.get("image_url"),
                        "published_at": datetime.now(timezone.utc),
                        "source": "rss",
                    })

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                health.record_failure(str(exc), status_code=status_code)
                feed_stats["failed"] += 1
                logger.warning(
                    "[RSS] %s: HTTP %d (%d consecutive failures)",
                    feed_name, status_code, health.consecutive_failures,
                )
                continue
            except Exception as exc:
                health.record_failure(str(exc))
                feed_stats["failed"] += 1
                logger.warning(
                    "[RSS] %s failed: %s (%d consecutive failures)",
                    feed_name, exc, health.consecutive_failures,
                )
                continue

    logger.info(
        "[RSS] Total: %d unique events from %d feeds "
        "(active=%d success=%d failed=%d suppressed=%d)",
        len(all_events), len(RSS_FEEDS),
        feed_stats["active"], feed_stats["success"],
        feed_stats["failed"], feed_stats["suppressed"],
    )
    return all_events
