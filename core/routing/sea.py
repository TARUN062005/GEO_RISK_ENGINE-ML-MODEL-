"""
core/routing/sea.py
-------------------
Sea Route Generator (Log4)

Sea routes follow real-world shipping lanes using the `searoute` library.
This routes through straits, around capes, and through canals (Suez, Panama)
producing fundamentally different waypoints than a great-circle arc.

Fallback: great-circle if searoute fails (e.g. landlocked endpoints).
"""

from __future__ import annotations

import logging

from core.geo.route import (
    RouteResult,
    GeocodedLocation,
    _interpolate_great_circle,
)
from storage.repository import _haversine_km

logger = logging.getLogger(__name__)


def generate_sea_route(
    origin: GeocodedLocation,
    destination: GeocodedLocation,
) -> RouteResult:
    """
    Generate a maritime shipping route using the searoute library.

    Routes through real shipping lanes: Suez Canal, Panama Canal,
    Strait of Malacca, Cape of Good Hope, etc.

    Coordinates are converted to the GeoJSON [lon, lat] convention
    required by searoute, then converted back to (lat, lon) tuples
    for the rest of the pipeline.

    Falls back to great-circle if searoute cannot find a path.

    Args:
        origin:      Geocoded origin location.
        destination: Geocoded destination location.

    Returns:
        RouteResult with maritime waypoints.
    """
    try:
        import searoute as sr  # type: ignore

        # searoute expects [longitude, latitude]
        origin_ll = [origin.lon, origin.lat]
        dest_ll   = [destination.lon, destination.lat]

        route_feature = sr.searoute(origin_ll, dest_ll, units="km")

        # Extract coordinates from GeoJSON LineString geometry
        coords = route_feature["geometry"]["coordinates"]  # list of [lon, lat]

        if not coords or len(coords) < 2:
            raise ValueError("searoute returned empty geometry")

        # Convert [lon, lat] → (lat, lon) for internal pipeline
        waypoints: list[tuple[float, float]] = [
            (float(c[1]), float(c[0])) for c in coords
        ]

        # Total distance from searoute properties, fallback to haversine
        total_km = route_feature.get("properties", {}).get("length", None)
        if total_km is None:
            total_km = _haversine_km(
                origin.lat, origin.lon, destination.lat, destination.lon
            ) * 1.3   # sea routes are ~30% longer than air

        logger.info(
            "Sea route: %d raw waypoints, %.0f km",
            len(waypoints), total_km,
        )

        # Phase 3: Downsample dense searoute output to max 20 strategic checkpoints
        # Preserves: origin port (first), destination port (last), evenly-spaced midpoints
        MAX_SEA_WAYPOINTS = 20
        if len(waypoints) > MAX_SEA_WAYPOINTS:
            original_count = len(waypoints)
            step = (len(waypoints) - 1) / (MAX_SEA_WAYPOINTS - 1)
            indices = [round(i * step) for i in range(MAX_SEA_WAYPOINTS)]
            indices[-1] = len(waypoints) - 1  # ensure last point is exact destination
            waypoints = [waypoints[i] for i in indices]
            logger.info(
                "[WAYPOINT REDUCTION] sea: %d → %d waypoints (%.0f%% reduction)",
                original_count, len(waypoints),
                (1 - len(waypoints) / original_count) * 100,
            )

        return RouteResult(
            origin=origin,
            destination=destination,
            waypoints=waypoints,
            total_distance_km=round(float(total_km), 1),
        )

    except Exception as exc:
        logger.warning(
            "searoute failed for %s → %s (%s). Falling back to great-circle.",
            origin.query, destination.query, exc,
        )
        return _sea_fallback(origin, destination)


def _sea_fallback(
    origin: GeocodedLocation,
    destination: GeocodedLocation,
) -> RouteResult:
    """
    Great-circle fallback when searoute cannot compute a maritime path.
    Applied when both endpoints are landlocked or searoute errors.
    Distance inflated by 1.3× to approximate maritime detour.
    """
    total_km = _haversine_km(
        origin.lat, origin.lon, destination.lat, destination.lon
    )
    n = max(15, min(50, int(total_km / 100)))

    waypoints = _interpolate_great_circle(
        origin.lat, origin.lon,
        destination.lat, destination.lon,
        n_points=n,
    )

    return RouteResult(
        origin=origin,
        destination=destination,
        waypoints=waypoints,
        total_distance_km=round(total_km * 1.3, 1),
    )
