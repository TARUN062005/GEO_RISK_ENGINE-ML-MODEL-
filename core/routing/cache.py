"""
core/routing/cache.py
---------------------
Tiny in-process route cache for repeated live/API analyses.

Keeps memory bounded and avoids regenerating identical AIR/SEA/ROAD routes
within the same process. No external cache service required.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

from core.geo.route import GeocodedLocation, RouteResult

_MAX_ROUTES = 128
_ROUTES: OrderedDict[tuple, RouteResult] = OrderedDict()


def _key(mode: str, origin: GeocodedLocation, destination: GeocodedLocation) -> tuple:
    return (
        mode,
        round(origin.lat, 4),
        round(origin.lon, 4),
        round(destination.lat, 4),
        round(destination.lon, 4),
    )


def get_or_generate_route(
    mode: str,
    origin: GeocodedLocation,
    destination: GeocodedLocation,
    factory: Callable[[], RouteResult],
) -> RouteResult:
    key = _key(mode, origin, destination)
    cached = _ROUTES.get(key)
    if cached is not None:
        try:
            from core import metrics
            metrics.inc("route_cache_hits")
        except Exception:
            pass
        _ROUTES.move_to_end(key)
        return cached

    try:
        from core import metrics
        metrics.inc("route_cache_misses")
    except Exception:
        pass
    route = factory()
    _ROUTES[key] = route
    if len(_ROUTES) > _MAX_ROUTES:
        _ROUTES.popitem(last=False)
    return route
