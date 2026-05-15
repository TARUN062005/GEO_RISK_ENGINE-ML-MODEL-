"""
ingestion/normalize.py
----------------------
Coordinate Resolver (Log3, updated Log10)

Resolves (lon, lat) from:
  1. Feed-supplied coordinates (fastest)
  2. Semantic geo tagger — zone keyword matching (Log7)
  3. LRU-cached Nominatim geocoding (Log10)
  4. Returns None if resolution fails

Log10: Added in-memory geocode cache to avoid repeated Nominatim
       lookups for common locations like 'Ukraine', 'Iran', 'Gaza'.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Log10: Cached geocoding helper (avoids re-importing + rate-limit on repeats)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1024)
def _cached_geocode(name: str) -> Optional[tuple[float, float]]:
    """
    Geocode a location name and return (lon, lat) or None.
    LRU-cached — repeated lookups for same name hit cache instantly.
    """
    try:
        from core.geo.route import geocode
        result = geocode(name)
        if result is not None:
            return (result.lon, result.lat)
    except Exception as exc:
        logger.debug("Geocode failed for '%s': %s", name, exc)
    return None


def _normalize_location_name(name: str) -> str:
    """Normalize a location name for better cache hit rate."""
    return name.strip().title()


def resolve_coordinates(
    location_names: list[str],
    feed_lat: Optional[float] = None,
    feed_lon: Optional[float] = None,
    raw_text: str = "",
) -> Optional[tuple[float, float]]:
    """
    Returns (lon, lat) pair for storage in GeoJSON Point format.

    Priority:
      1. Feed-supplied lat/lon (GDELT ActionGeo fields)
      2. Semantic geo tagger — zone keyword matching (Log7)
      3. LRU-cached NER location → Nominatim geocode (Log10)
      4. None (event dropped from ingestion)
    """
    # 1. Feed coordinates
    if feed_lat is not None and feed_lon is not None:
        if -90 <= feed_lat <= 90 and -180 <= feed_lon <= 180:
            return (feed_lon, feed_lat)

    # 2. Semantic geo tagger (Log7) — fast, zero-API-call zone matching
    try:
        from ingestion.geo_tagger import extract_zone_coordinates
        text_to_scan = raw_text or " ".join(location_names[:5])
        zone_result = extract_zone_coordinates(text_to_scan)
        if zone_result is not None:
            lat, lon, zone_name = zone_result
            logger.debug("Geo-tagged via zone: '%s' → %s (%.1f, %.1f)", text_to_scan[:40], zone_name, lat, lon)
            return (lon, lat)
    except Exception as exc:
        logger.debug("Geo tagger error: %s", exc)

    # 3. Geocode first NER location (Log10: LRU-cached)
    for name in location_names[:3]:
        normalized = _normalize_location_name(name)
        if not normalized or len(normalized) < 2:
            continue
        before = _cached_geocode.cache_info()
        result = _cached_geocode(normalized)
        try:
            from core import metrics
            metrics.log_cache_info("geocode", before, _cached_geocode.cache_info())
        except Exception:
            pass
        if result is not None:
            logger.debug("Resolved '%s' → (%.4f, %.4f) [cached]", normalized, result[1], result[0])
            return result

    return None
