"""
storage/repository.py
---------------------
MongoDB Repository — Geo-Aware Queries (Log3)

Provides the geo query layer used by the API.
Uses MongoDB 2dsphere index — NO full collection scan.

Key function:
  get_events_near_route(route_points, radius_km, collection)
    → list[EnrichedEvent]

Design:
  - Route is sampled into waypoints (every ~25 km arc)
  - Each waypoint issues a $geoWithin / $nearSphere query
  - Results deduplicated by event_id
  - Only events with schema_version >= "2" returned (Log2+ schema)
"""

from __future__ import annotations

import logging
import math
from typing import Any

from storage.schema import EnrichedEvent, from_mongo_doc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RADIUS_KM: float = 50.0      # Buffer around route
WAYPOINT_SPACING_KM: float = 25.0   # Sample route every N km


# ---------------------------------------------------------------------------
# Haversine distance helper (no external dep)
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Route sampling
# ---------------------------------------------------------------------------

def _sample_route_waypoints(
    route: list[tuple[float, float]],
    spacing_km: float = WAYPOINT_SPACING_KM,
) -> list[tuple[float, float]]:
    """
    Downsample a dense route to evenly-spaced waypoints.
    Ensures we query at most one point per `spacing_km`.
    Always includes first and last point.
    """
    if len(route) <= 2:
        return list(route)

    waypoints: list[tuple[float, float]] = [route[0]]
    accumulated = 0.0

    for i in range(1, len(route)):
        prev = route[i - 1]
        curr = route[i]
        accumulated += _haversine_km(prev[0], prev[1], curr[0], curr[1])
        if accumulated >= spacing_km:
            waypoints.append(curr)
            accumulated = 0.0

    if waypoints[-1] != route[-1]:
        waypoints.append(route[-1])

    return waypoints


# ---------------------------------------------------------------------------
# Distance from a point to the nearest waypoint on route
# ---------------------------------------------------------------------------

def _min_distance_to_route(
    event_lat: float,
    event_lon: float,
    route: list[tuple[float, float]],
) -> float:
    """Return distance (km) from event to the nearest route waypoint."""
    return min(
        _haversine_km(event_lat, event_lon, wlat, wlon)
        for wlat, wlon in route
    )


# ---------------------------------------------------------------------------
# Main geo query
# ---------------------------------------------------------------------------

async def get_events_near_route(
    route: list[tuple[float, float]],
    collection,                          # motor AsyncIOMotorCollection
    radius_km: float = DEFAULT_RADIUS_KM,
    max_results: int = 200,
    min_label_confidence: float = 0.50,  # Log3: confidence threshold
) -> tuple[list[EnrichedEvent], dict[str, float]]:
    """
    Query MongoDB for all enriched events within `radius_km` of any
    route waypoint. Uses the 2dsphere index — no collection scan.

    Args:
        route:               List of (lat, lon) tuples defining the route.
        collection:          Motor async MongoDB collection.
        radius_km:           Search radius around each waypoint.
        max_results:         Cap total results to prevent overload.
        min_label_confidence: Discard events below this confidence (Log3).

    Returns:
        Tuple of:
          - list[EnrichedEvent]: deduplicated events near route
          - dict[str, float]:   {event_id → distance_km from route}
    """
    waypoints = _sample_route_waypoints(route)
    logger.info("Route sampled: %d waypoints (spacing=%.0f km)", len(waypoints), WAYPOINT_SPACING_KM)

    radius_meters = radius_km * 1000.0
    seen_ids: set[str] = set()
    raw_docs: list[dict[str, Any]] = []

    for lat, lon in waypoints:
        cursor = collection.find(
            {
                "location": {
                    "$nearSphere": {
                        "$geometry": {"type": "Point", "coordinates": [lon, lat]},
                        "$maxDistance": radius_meters,
                    }
                },
                "ml.label_confidence": {"$gte": min_label_confidence},
                "schema_version": {"$gte": "2"},          # Log2+ schema only
            },
            limit=max_results // max(len(waypoints), 1),
        )
        async for doc in cursor:
            eid = str(doc.get("_id", ""))
            if eid not in seen_ids:
                seen_ids.add(eid)
                raw_docs.append(doc)

        if len(raw_docs) >= max_results:
            break

    logger.info("Geo query returned %d unique events.", len(raw_docs))

    # Deserialize + compute per-event distances
    events: list[EnrichedEvent] = []
    distances: dict[str, float] = {}

    for doc in raw_docs:
        try:
            ev = from_mongo_doc(doc)
            # Distance from event to nearest point on route
            ev_lon, ev_lat = ev.location.coordinates
            dist = _min_distance_to_route(ev_lat, ev_lon, route)
            events.append(ev)
            distances[ev.event_id] = round(dist, 3)
        except Exception as exc:
            logger.warning("Failed to deserialize event %s: %s", doc.get("_id"), exc)

    return events, distances
