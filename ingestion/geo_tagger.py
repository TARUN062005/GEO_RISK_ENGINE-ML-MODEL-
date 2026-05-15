"""
ingestion/geo_tagger.py
-----------------------
Semantic Geo Tagger (Log7)

Solves the CRITICAL problem: RSS articles often lack lat/lon coordinates.
NER alone returns location names, but many geopolitical terms refer to
regions/zones, not geocodable cities:

  "Strait of Hormuz" → Nominatim may fail
  "Red Sea shipping" → no city to geocode
  "Tehran sanctions" → may geocode to wrong context

This module provides a keyword-to-zone mapping that:
  1. Scans article text for zone-related keywords
  2. Maps matches to known GeoZone center coordinates
  3. Returns coordinates for events that would otherwise be dropped

This runs BEFORE the Nominatim geocoder as a fast, zero-API-call fallback.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword → Zone Coordinate Mapping
# ---------------------------------------------------------------------------
# Each entry: pattern (case-insensitive), (lat, lon), zone_name
# Patterns are compiled once at import time for performance.

_ZONE_KEYWORD_MAP: list[tuple[re.Pattern, float, float, str]] = [
    # Maritime chokepoints
    (re.compile(r"strait\s+of\s+hormuz|hormuz\s+strait", re.I),          26.5,  56.0,  "Strait of Hormuz"),
    (re.compile(r"persian\s+gulf|arabian\s+gulf", re.I),                 26.0,  52.0,  "Strait of Hormuz"),
    (re.compile(r"red\s+sea|bab[\s-]*el[\s-]*mandeb", re.I),            13.5,  42.5,  "Red Sea / Bab el-Mandeb"),
    (re.compile(r"suez\s+canal|suez", re.I),                            30.5,  32.3,  "Suez Canal"),
    (re.compile(r"strait\s+of\s+malacca|malacca\s+strait", re.I),        2.5, 101.0,  "Strait of Malacca"),
    (re.compile(r"gulf\s+of\s+aden|aden\s+gulf", re.I),                 12.0,  47.0,  "Gulf of Aden"),
    (re.compile(r"panama\s+canal", re.I),                                 9.1, -79.7,  "Panama Canal"),
    (re.compile(r"south\s+china\s+sea", re.I),                           14.0, 115.0,  "South China Sea"),
    (re.compile(r"taiwan\s+strait|formosa\s+strait", re.I),             24.0, 119.5,  "Taiwan Strait"),
    (re.compile(r"black\s+sea", re.I),                                   43.5,  34.0,  "Black Sea"),
    (re.compile(r"houthi|yemen.*missile|yemen.*shipping", re.I),         15.5,  48.0,  "Yemen"),

    # Conflict zones
    (re.compile(r"\bukraine\b|kyiv|donetsk|zaporizhzhia|kherson", re.I), 48.5,  36.0,  "Ukraine War Zone"),
    (re.compile(r"\bgaza\b|hamas|israeli.*strike|palestine", re.I),      31.4,  34.4,  "Gaza / Southern Israel"),
    (re.compile(r"\blibya\b|tripoli|benghazi|libyan", re.I),            32.0,  20.0,  "Eastern Libya"),
    (re.compile(r"somalia|al[\s-]*shabaab|mogadishu", re.I),              5.0,  46.0,  "Somalia / Horn of Africa"),
    (re.compile(r"\bsahel\b|mali.*jihadist|niger.*coup|burkina\s+faso", re.I), 15.0, 2.0, "Sahel Region"),
    (re.compile(r"\bmyanmar\b|rohingya|junta.*myanmar", re.I),           20.0,  96.5,  "Myanmar Conflict Zone"),
    (re.compile(r"\bafghanistan\b|taliban|kabul|isis[\s-]*k", re.I),     33.9,  67.7,  "Afghanistan"),
    (re.compile(r"\byemen\b|houthi|sanaa", re.I),                        15.5,  48.0,  "Yemen"),
    (re.compile(r"\bsyria\b|damascus|aleppo|idlib", re.I),              35.5,  40.0,  "Syria / Northern Iraq"),
    (re.compile(r"\bsudan\b|darfur|khartoum|RSF\b|rapid\s+support", re.I), 13.0, 30.0, "Sudan / Darfur"),

    # Sanctions zones
    (re.compile(r"north\s+korea|pyongyang|DPRK", re.I),                 39.0, 127.5,  "North Korea Buffer"),
    (re.compile(r"\biran\b|tehran|iranian\s+sanction", re.I),           32.4,  53.7,  "Iran Sanctions Zone"),
    (re.compile(r"\bcrimea\b|sevastopol|annexed\s+territor", re.I),     45.3,  34.0,  "Crimea / Annexed Territories"),

    # Major geopolitical hotspots (not zones but frequent in news)
    (re.compile(r"south\s+korea.*north\s+korea|korean\s+peninsula|DMZ", re.I), 37.5, 127.0, "North Korea Buffer"),
    (re.compile(r"east\s+china\s+sea|senkaku|diaoyu", re.I),           27.0, 123.0,  "South China Sea"),
    (re.compile(r"pirac[y]|maritime.*attack|ship.*hijack", re.I),       12.0,  47.0,  "Gulf of Aden"),
    (re.compile(r"strait\s+of\s+gibraltar", re.I),                      36.0,  -5.5,  "Suez Canal"),  # Mediterranean approach
]


def extract_zone_coordinates(text: str) -> Optional[tuple[float, float, str]]:
    """
    Scan text for geopolitical zone keywords and return coordinates.

    Returns:
        (lat, lon, zone_name) for the FIRST match, or None.
        First match = highest priority (most specific patterns first).
    """
    if not text:
        return None

    for pattern, lat, lon, zone_name in _ZONE_KEYWORD_MAP:
        if pattern.search(text):
            logger.debug("Geo-tagged '%s' → %s (%.1f, %.1f)", text[:60], zone_name, lat, lon)
            return (lat, lon, zone_name)

    return None


def extract_all_zone_matches(text: str) -> list[dict]:
    """
    Return ALL zone keyword matches for a given text.
    Useful for multi-zone event tagging.
    """
    if not text:
        return []

    matches = []
    seen_zones = set()
    for pattern, lat, lon, zone_name in _ZONE_KEYWORD_MAP:
        if zone_name not in seen_zones and pattern.search(text):
            seen_zones.add(zone_name)
            matches.append({
                "zone": zone_name,
                "lat": lat,
                "lon": lon,
            })

    return matches
