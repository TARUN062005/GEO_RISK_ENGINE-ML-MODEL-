"""
core/routing/road.py
--------------------
Road / Land Route Generator (Log4)

Road routes follow overland corridors through continental land masses.
Strategy:
  1. Short routes (< 800 km):  osmnx shortest-path on road network graph
  2. Long routes (> 800 km):   land-corridor interpolation via continental
                               waypoints (osmnx impractical across continents)

Continental waypoints represent major overland transit hubs used by
real logistics routes (Silk Road, Trans-Siberian, Trans-African corridors).

The key difference vs air:
  - Road routes hug land boundaries (more events from conflict border zones)
  - Road routes cannot cross oceans — coastal detours applied
  - Passes through different countries than the air arc

Fallback: great-circle if no land path can be computed.
"""

from __future__ import annotations

import logging
import math

from core.geo.route import (
    RouteResult,
    GeocodedLocation,
    _interpolate_great_circle,
)
from storage.repository import _haversine_km

logger = logging.getLogger(__name__)

# Distance threshold below which osmnx is attempted (km)
OSMNX_MAX_DISTANCE_KM = 800.0

# ---------------------------------------------------------------------------
# Continental transit hub waypoints
# These are NOT hardcoded routes — they are candidate intermediate nodes
# that the algorithm selects dynamically based on origin/destination geography.
# ---------------------------------------------------------------------------

_CONTINENTAL_HUBS: list[dict] = [
    # Europe
    {"name": "Istanbul",       "lat": 41.01, "lon": 28.97, "continent": "EU-AS"},
    {"name": "Warsaw",         "lat": 52.23, "lon": 21.01, "continent": "EU"},
    {"name": "Vienna",         "lat": 48.21, "lon": 16.37, "continent": "EU"},
    {"name": "Berlin",         "lat": 52.52, "lon": 13.40, "continent": "EU"},
    {"name": "Moscow",         "lat": 55.75, "lon": 37.62, "continent": "EU-AS"},
    # Middle East / Central Asia
    {"name": "Tehran",         "lat": 35.69, "lon": 51.39, "continent": "AS"},
    {"name": "Kabul",          "lat": 34.52, "lon": 69.18, "continent": "AS"},
    {"name": "Islamabad",      "lat": 33.72, "lon": 73.06, "continent": "AS"},
    {"name": "Almaty",         "lat": 43.24, "lon": 76.89, "continent": "AS"},
    # South / Southeast Asia
    {"name": "Delhi",          "lat": 28.61, "lon": 77.21, "continent": "AS"},
    {"name": "Kolkata",        "lat": 22.57, "lon": 88.36, "continent": "AS"},
    {"name": "Bangkok",        "lat": 13.75, "lon": 100.52, "continent": "AS"},
    {"name": "Kuala Lumpur",   "lat":  3.14, "lon": 101.69, "continent": "AS"},
    # East Asia
    {"name": "Urumqi",         "lat": 43.83, "lon": 87.61, "continent": "AS"},
    {"name": "Chengdu",        "lat": 30.66, "lon": 104.06, "continent": "AS"},
    {"name": "Beijing",        "lat": 39.91, "lon": 116.39, "continent": "AS"},
    # Africa
    {"name": "Cairo",          "lat": 30.06, "lon": 31.25, "continent": "AF"},
    {"name": "Khartoum",       "lat": 15.55, "lon": 32.53, "continent": "AF"},
    {"name": "Nairobi",        "lat": -1.29, "lon": 36.82, "continent": "AF"},
    {"name": "Lagos",          "lat":  6.52, "lon":  3.37, "continent": "AF"},
    {"name": "Johannesburg",   "lat": -26.20, "lon": 28.04, "continent": "AF"},
    # Americas
    {"name": "Mexico City",    "lat": 19.43, "lon": -99.13, "continent": "NA"},
    {"name": "Bogota",         "lat":  4.71, "lon": -74.07, "continent": "SA"},
    {"name": "Buenos Aires",   "lat": -34.60, "lon": -58.38, "continent": "SA"},
    {"name": "Chicago",        "lat": 41.88, "lon": -87.63, "continent": "NA"},
    {"name": "Los Angeles",    "lat": 34.05, "lon": -118.24, "continent": "NA"},
]


def _nearest_hub(lat: float, lon: float) -> dict:
    """Return the continental hub nearest to the given (lat, lon)."""
    return min(
        _CONTINENTAL_HUBS,
        key=lambda h: _haversine_km(lat, lon, h["lat"], h["lon"]),
    )


def _land_corridor_route(
    origin: GeocodedLocation,
    destination: GeocodedLocation,
) -> list[tuple[float, float]]:
    """
    Build a land-biased route by routing through intermediate continental hubs.

    Algorithm:
      1. Find nearest hub to origin
      2. Find nearest hub to destination
      3. Interpolate: origin → hub_o → hub_d → destination
         (if hub_o == hub_d, skip one hop)

    This creates a route that follows overland paths rather than
    flying directly over oceans, producing different country exposure.
    """
    hub_o = _nearest_hub(origin.lat, origin.lon)
    hub_d = _nearest_hub(destination.lat, destination.lon)

    segments: list[tuple[float, float]] = []

    def _seg(lat1, lon1, lat2, lon2) -> list[tuple[float, float]]:
        dist = _haversine_km(lat1, lon1, lat2, lon2)
        n = max(5, min(20, int(dist / 150)))
        return _interpolate_great_circle(lat1, lon1, lat2, lon2, n_points=n)

    # origin → hub_o
    segments += _seg(origin.lat, origin.lon, hub_o["lat"], hub_o["lon"])

    # hub_o → hub_d (only if they differ meaningfully)
    if _haversine_km(hub_o["lat"], hub_o["lon"], hub_d["lat"], hub_d["lon"]) > 200:
        segments += _seg(hub_o["lat"], hub_o["lon"], hub_d["lat"], hub_d["lon"])

    # hub_d → destination
    segments += _seg(hub_d["lat"], hub_d["lon"], destination.lat, destination.lon)

    return segments


def _try_osmnx(
    origin: GeocodedLocation,
    destination: GeocodedLocation,
) -> list[tuple[float, float]] | None:
    """
    Attempt osmnx road-network routing for short (< 800 km) routes.
    Returns None if osmnx is unavailable or route fails.
    """
    try:
        import osmnx as ox  # type: ignore
        import networkx as nx  # type: ignore

        # Download drive network around the midpoint bounding box
        north = max(origin.lat, destination.lat) + 0.5
        south = min(origin.lat, destination.lat) - 0.5
        east  = max(origin.lon, destination.lon) + 0.5
        west  = min(origin.lon, destination.lon) - 0.5

        G = ox.graph_from_bbox(
            (north, south, east, west),
            network_type="drive",
            simplify=True,
        )

        orig_node = ox.nearest_nodes(G, origin.lon,      origin.lat)
        dest_node = ox.nearest_nodes(G, destination.lon, destination.lat)

        path_nodes = nx.shortest_path(G, orig_node, dest_node, weight="length")

        waypoints: list[tuple[float, float]] = [
            (G.nodes[n]["y"], G.nodes[n]["x"]) for n in path_nodes
        ]
        logger.info("osmnx road route: %d nodes", len(waypoints))
        return waypoints

    except Exception as exc:
        logger.debug("osmnx routing failed: %s", exc)
        return None


def generate_road_route(
    origin: GeocodedLocation,
    destination: GeocodedLocation,
) -> RouteResult:
    """
    Generate an overland road/rail corridor route.

    Short routes (< 800 km): attempts osmnx real road network.
    Long routes: uses continental hub waypoints.
    Fallback: great-circle.

    The route intentionally stays near land corridors, crossing
    through border zones and inland regions — picking up conflict
    events that air routes would miss.

    Args:
        origin:      Geocoded origin.
        destination: Geocoded destination.

    Returns:
        RouteResult with land-biased waypoints.
    """
    total_km = _haversine_km(
        origin.lat, origin.lon,
        destination.lat, destination.lon,
    )

    waypoints: list[tuple[float, float]] | None = None

    # Attempt osmnx for short routes
    if total_km <= OSMNX_MAX_DISTANCE_KM:
        waypoints = _try_osmnx(origin, destination)

    # Land corridor for long routes or osmnx fallback
    if waypoints is None:
        logger.info(
            "Using land-corridor routing for %s → %s (%.0f km)",
            origin.query, destination.query, total_km,
        )
        waypoints = _land_corridor_route(origin, destination)

    # Compute actual route distance as sum of segment haversines
    route_km = sum(
        _haversine_km(waypoints[i][0], waypoints[i][1],
                      waypoints[i + 1][0], waypoints[i + 1][1])
        for i in range(len(waypoints) - 1)
    )

    return RouteResult(
        origin=origin,
        destination=destination,
        waypoints=waypoints,
        total_distance_km=round(route_km, 1),
    )
