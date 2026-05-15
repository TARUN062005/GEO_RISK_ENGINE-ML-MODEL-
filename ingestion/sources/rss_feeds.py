"""
ingestion/sources/rss_feeds.py
------------------------------
RSS Feed Fetcher (Log5)

Fetches geopolitical news from major RSS feeds:
  - Reuters (World)
  - BBC News (World)
  - Al Jazeera
  - Associated Press

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
# RSS Feed Configuration
# ---------------------------------------------------------------------------

RSS_FEEDS: list[dict[str, str]] = [
    {
        "name": "Reuters World",
        "url": "https://feeds.reuters.com/Reuters/worldNews",
        "publisher": "Reuters",
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
    {
        "name": "Associated Press",
        "url": "https://rsshub.app/apnews/topics/apf-topnews",
        "publisher": "Associated Press",
    },
    {
        "name": "NPR World",
        "url": "https://feeds.npr.org/1004/rss.xml",
        "publisher": "NPR",
    },
    # Log11: Maritime-specific feeds
    {
        "name": "Splash247 Maritime",
        "url": "https://splash247.com/feed/",
        "publisher": "Splash 247",
    },
    {
        "name": "Maritime Executive",
        "url": "https://www.maritime-executive.com/blog/feed",
        "publisher": "The Maritime Executive",
    },
    {
        "name": "Seatrade Maritime",
        "url": "https://www.seatrade-maritime.com/rss.xml",
        "publisher": "Seatrade Maritime",
    },
]


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

    Returns list of dicts with keys:
      event_id, text, source_url, publisher, image_url, published_at, source

    Deduplicates by URL hash.
    """
    all_events: list[dict] = []
    seen_hashes: set[str] = set()

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        for feed in RSS_FEEDS:
            try:
                resp = await client.get(feed["url"])
                resp.raise_for_status()

                items = _parse_rss_xml(resp.text, feed)
                logger.info("[RSS] %s: parsed %d items", feed["name"], len(items))

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

            except Exception as exc:
                logger.warning("[RSS] Failed to fetch %s: %s", feed["name"], exc)
                continue

    logger.info("[RSS] Total: %d unique events from %d feeds", len(all_events), len(RSS_FEEDS))
    return all_events
