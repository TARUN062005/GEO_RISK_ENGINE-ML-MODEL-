from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.geo.route import GeocodedLocation
from core.geo.zones import check_route_zone_intersections
from core.routing.air import generate_air_route
from core.routing.cache import get_or_generate_route
from ingestion.clustering import cluster_incidents
from ingestion.realtime_worker import _is_valid_raw_event


def _doc(text: str, publisher: str, url: str, lon: float = 43.0, lat: float = 14.0) -> dict:
    return {
        "_id": url,
        "raw_text": text,
        "publisher": publisher,
        "source_url": url,
        "published_at": datetime.now(timezone.utc),
        "location": {"type": "Point", "coordinates": [lon, lat]},
        "ml": {"label": "conflict", "intensity_score": 0.7},
        "verification": {"credibility_score": 0.9},
    }


def test_cluster_incidents_merges_multi_source_same_incident():
    docs = [
        _doc("Houthi missile attack reported against shipping in Red Sea", "Reuters", "https://r.example/a"),
        _doc("Red Sea shipping hit by Houthi attack, officials say", "BBC", "https://b.example/a"),
        _doc("Earthquake damages homes in Chile", "AP", "https://ap.example/chile", lon=-71.0, lat=-33.0),
    ]

    clustered = cluster_incidents(docs)

    assert len(clustered) == 2
    incident = next(d for d in clustered if d["corroboration_count"] == 2)
    assert incident["canonical_event_id"].startswith("incident-")
    assert incident["corroboration_score"] > 0
    assert sorted(incident["publishers"]) == ["BBC", "Reuters"]
    assert len(incident["source_urls"]) == 2


def test_raw_event_validation_rejects_malformed_and_stale():
    assert not _is_valid_raw_event({"text": "too short"})
    assert not _is_valid_raw_event({
        "text": "Military tensions reported near a strategic maritime route",
        "source_url": "not-a-url",
    })
    assert not _is_valid_raw_event({
        "text": "Military tensions reported near a strategic maritime route",
        "source_url": "https://example.com/a",
        "published_at": datetime.now(timezone.utc) - timedelta(days=30),
    })


def test_air_route_does_not_inherit_hormuz_maritime_zone():
    mumbai = GeocodedLocation("Mumbai", 19.076, 72.8777, "Mumbai")
    dubai = GeocodedLocation("Dubai", 25.2048, 55.2708, "Dubai")
    route = generate_air_route(mumbai, dubai)

    zones = check_route_zone_intersections(route.waypoints, transport_mode="air")
    names = {z["zone"] for z in zones}

    assert "Strait of Hormuz" not in names
    assert "Hormuz Tanker Disruption Area" not in names


def test_route_cache_returns_same_object_for_repeated_route():
    origin = GeocodedLocation("A", 1.0, 2.0, "A")
    dest = GeocodedLocation("B", 3.0, 4.0, "B")
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return generate_air_route(origin, dest)

    first = get_or_generate_route("air-test", origin, dest, factory)
    second = get_or_generate_route("air-test", origin, dest, factory)

    assert first is second
    assert calls["n"] == 1
