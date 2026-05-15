"""
core/geo/route.py
-----------------
Route Generation Module (Log3)

Converts origin/destination strings → geocoded coordinates → route waypoints.

No external routing API needed:
  - Geocoding: geopy Nominatim (free, no key)
  - Route: great-circle interpolation (accurate for risk proximity purposes)

For road/sea routes see core/routing/ (OSMnx, searoute) — plug in later.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

@dataclass
class GeocodedLocation:
    query: str          # original input string
    lat: float
    lon: float
    display_name: str   # Nominatim's resolved name


@lru_cache(maxsize=512)
def geocode(location_query: str) -> Optional[GeocodedLocation]:
    """
    Convert a location string → (lat, lon) using Nominatim.
    Results are cached (LRU 512) — identical queries hit the cache.

    Rate-limited to 1 req/sec as required by Nominatim ToS.
    Returns None if geocoding fails.
    """
    try:
        from geopy.geocoders import Nominatim          # type: ignore
        from geopy.exc import GeocoderTimedOut        # type: ignore

        geolocator = Nominatim(user_agent="geo-risk-engine/1.0")

        time.sleep(1.0)   # Nominatim rate limit: 1 req/sec
        result = geolocator.geocode(location_query, timeout=10)

        if result is None:
            logger.warning("Nominatim returned no result for: %s", location_query)
            return None

        return GeocodedLocation(
            query=location_query,
            lat=result.latitude,
            lon=result.longitude,
            display_name=result.address,
        )

    except Exception as exc:
        logger.error("Geocoding failed for '%s': %s", location_query, exc)
        return None


# ---------------------------------------------------------------------------
# Great-circle interpolation
# ---------------------------------------------------------------------------

def _interpolate_great_circle(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    n_points: int = 20,
) -> list[tuple[float, float]]:
    """
    Interpolate N evenly-spaced points along the great-circle arc
    between (lat1, lon1) and (lat2, lon2).

    Uses spherical linear interpolation (SLERP) for accuracy.

    Returns:
        List of (lat, lon) tuples including start and end.
    """
    # Convert to radians
    phi1, lam1 = math.radians(lat1), math.radians(lon1)
    phi2, lam2 = math.radians(lat2), math.radians(lon2)

    # Cartesian unit vectors
    def to_xyz(phi, lam):
        return (
            math.cos(phi) * math.cos(lam),
            math.cos(phi) * math.sin(lam),
            math.sin(phi),
        )

    def to_latlon(x, y, z):
        lat = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
        lon = math.degrees(math.atan2(y, x))
        return lat, lon

    x1, y1, z1 = to_xyz(phi1, lam1)
    x2, y2, z2 = to_xyz(phi2, lam2)

    # Angular distance
    dot = max(-1.0, min(1.0, x1 * x2 + y1 * y2 + z1 * z2))
    omega = math.acos(dot)

    points: list[tuple[float, float]] = []

    if omega < 1e-10:
        # Points are identical
        return [(lat1, lon1), (lat2, lon2)]

    for i in range(n_points):
        t = i / (n_points - 1)
        sin_omega = math.sin(omega)
        a = math.sin((1 - t) * omega) / sin_omega
        b = math.sin(t * omega) / sin_omega
        x = a * x1 + b * x2
        y = a * y1 + b * y2
        z = a * z1 + b * z2
        points.append(to_latlon(x, y, z))

    return points


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    origin: GeocodedLocation
    destination: GeocodedLocation
    waypoints: list[tuple[float, float]]   # (lat, lon) pairs
    total_distance_km: float


def generate_route(origin_query: str, destination_query: str) -> RouteResult:
    """
    Full route generation pipeline:
      1. Geocode origin + destination
      2. Interpolate great-circle waypoints
      3. Compute approximate total distance

    Args:
        origin_query:      Free-text location (e.g. "Mumbai, India")
        destination_query: Free-text location (e.g. "Cairo, Egypt")

    Returns:
        RouteResult with geocoded endpoints and waypoints list.

    Raises:
        ValueError: If either location cannot be geocoded.
    """
    origin = geocode(origin_query)
    if origin is None:
        raise ValueError(f"Could not geocode origin: '{origin_query}'")

    destination = geocode(destination_query)
    if destination is None:
        raise ValueError(f"Could not geocode destination: '{destination_query}'")

    logger.info(
        "Route: %s (%.4f, %.4f) → %s (%.4f, %.4f)",
        origin_query, origin.lat, origin.lon,
        destination_query, destination.lat, destination.lon,
    )

    # Adaptive waypoint count: ~1 waypoint per 100 km, min 10, max 50
    from storage.repository import _haversine_km
    total_km = _haversine_km(origin.lat, origin.lon, destination.lat, destination.lon)
    n_points = max(10, min(50, int(total_km / 100)))

    waypoints = _interpolate_great_circle(
        origin.lat, origin.lon,
        destination.lat, destination.lon,
        n_points=n_points,
    )

    return RouteResult(
        origin=origin,
        destination=destination,
        waypoints=waypoints,
        total_distance_km=round(total_km, 1),
    )
