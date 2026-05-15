"""
core/routing/air.py
-------------------
Air Route Generator (Log4)

Air routes follow great-circle arcs — the shortest spherical path.
Reuses core/geo/route._interpolate_great_circle() which already exists.

Output is a RouteResult compatible with the existing pipeline.
"""

from __future__ import annotations

from core.geo.route import (
    RouteResult,
    GeocodedLocation,
    _interpolate_great_circle,
)
from storage.repository import _haversine_km


def generate_air_route(
    origin: GeocodedLocation,
    destination: GeocodedLocation,
    n_points: int = 50,
) -> RouteResult:
    """
    Generate a great-circle (air) route between two geocoded points.

    Air routes fly the shortest spherical arc, crossing directly over
    any terrain (mountains, seas, borders).

    Args:
        origin:      Geocoded origin.
        destination: Geocoded destination.
        n_points:    Number of interpolation points.

    Returns:
        RouteResult with waypoints along the air arc.
    """
    total_km = _haversine_km(
        origin.lat, origin.lon,
        destination.lat, destination.lon,
    )
    # Adaptive sampling: more waypoints for longer routes
    n = max(15, min(n_points, int(total_km / 100)))

    waypoints = _interpolate_great_circle(
        origin.lat, origin.lon,
        destination.lat, destination.lon,
        n_points=n,
    )

    return RouteResult(
        origin=origin,
        destination=destination,
        waypoints=waypoints,
        total_distance_km=round(total_km, 1),
    )
