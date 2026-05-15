"""
core/orchestrator.py
--------------------
Orchestrator — Extended with analyze_route_real() (Log3)

Log2 base: read-only aggregation, accepts pre-enriched EnrichedEvent objects.

Log3 extension: adds analyze_route_real(origin, destination)
  — the full end-to-end async pipeline accessible from the API.
  — NO ML inference runs here (Log2 rule preserved).
  — All ML fields read from stored EnrichedEvent.ml.* (DB-side).

DO NOT modify the existing run() function below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from core.risk.features import build_event_features, EventFeatures, normalize_scores
from core.risk.model import compute, RiskScore
from storage.schema import EnrichedEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema — unchanged from Log2
# ---------------------------------------------------------------------------

@dataclass
class RouteRiskOutput:
    route_id: str
    final_risk_score: float
    risk_band: str
    event_count: int
    dominant_threat: str
    explanation: dict[str, Any]
    top_events: list[dict[str, Any]] = field(default_factory=list)
    locations_detected: list[str] = field(default_factory=list)
    processed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id":           self.route_id,
            "final_risk_score":   self.final_risk_score,
            "risk_band":          self.risk_band,
            "event_count":        self.event_count,
            "dominant_threat":    self.dominant_threat,
            "explanation":        self.explanation,
            "top_events":         self.top_events,
            "locations_detected": self.locations_detected,
            "processed_at":       self.processed_at,
        }


# ---------------------------------------------------------------------------
# run() — unchanged from Log2, kept for backward compatibility
# ---------------------------------------------------------------------------

def run(
    route_id: str,
    enriched_events: list[EnrichedEvent],
    distances_km: dict[str, float],
) -> RouteRiskOutput:
    """
    Aggregate pre-enriched MongoDB events into a route risk score.
    (Unchanged from Log2 — see Log2 notes for design rationale.)
    """
    logger.info("Aggregation start: route=%s events=%d", route_id, len(enriched_events))
    reference_time = datetime.now(timezone.utc)

    feature_list: list[EventFeatures] = []
    for ev in enriched_events:
        distance = distances_km.get(ev.event_id, 0.0)
        features = build_event_features(
            event_id=ev.event_id,
            intensity_score=ev.ml.intensity_score,
            event_label=ev.ml.label,
            classifier_confidence=ev.ml.label_confidence,
            event_time=ev.published_at,
            distance_km=distance,
            reference_time=reference_time,
        )
        feature_list.append(features)

    risk_score: RiskScore = compute(feature_list)

    all_locations: list[str] = []
    for ev in enriched_events:
        all_locations.extend(ev.ml.location_names)
    unique_locations = list(dict.fromkeys(all_locations))

    paired = sorted(
        zip(enriched_events, feature_list),
        key=lambda p: p[1].composite_risk,
        reverse=True,
    )
    top_events = [
        {
            "event_id":   ev.event_id,
            "label":      ev.ml.label,
            "confidence": ev.ml.label_confidence,
            "intensity":  ev.ml.intensity_score,
            "composite":  feat.composite_risk,
            "locations":  ev.ml.location_names[:3],
        }
        for ev, feat in paired[:5]
    ]

    output = RouteRiskOutput(
        route_id=route_id,
        final_risk_score=risk_score.final_score,
        risk_band=risk_score.risk_band,
        event_count=risk_score.event_count,
        dominant_threat=risk_score.dominant_event_label,
        explanation=risk_score.explanation,
        top_events=top_events,
        locations_detected=unique_locations[:20],
        processed_at=reference_time.isoformat(),
    )

    logger.info("Aggregation done: route=%s score=%.4f band=%s",
                route_id, output.final_risk_score, output.risk_band)
    return output


# ---------------------------------------------------------------------------
# Log3 — analyze_route_real()
# Full end-to-end async pipeline: geocode → route → DB → aggregate → output
# ---------------------------------------------------------------------------

async def analyze_route_real(
    origin: str,
    destination: str,
    mongo_collection,                    # motor AsyncIOMotorCollection
    radius_km: float = 50.0,
    min_confidence: float = 0.50,
) -> dict[str, Any]:
    """
    Log3 entrypoint: full dynamic pipeline for any origin/destination pair.

    Steps:
      1. Geocode origin + destination (Nominatim, cached)
      2. Generate great-circle route waypoints
      3. Query MongoDB for events near route (2dsphere index)
      4. Aggregate into RouteRiskOutput (no ML inference — reads stored fields)
      5. Return structured JSON response

    Args:
        origin:           Free-text origin location (e.g. "Mumbai, India")
        destination:      Free-text destination (e.g. "Cairo, Egypt")
        mongo_collection: Motor async collection (geo_events)
        radius_km:        Route buffer search radius
        min_confidence:   Filter out events with ML confidence below this

    Returns:
        Dict matching the specified API output schema.

    Raises:
        ValueError: If geocoding fails for either location.
    """
    from core.geo.route import generate_route
    from storage.repository import get_events_near_route

    # ── 1. Route generation ───────────────────────────────────────────────
    import asyncio
    route_result = await asyncio.to_thread(generate_route, origin, destination)
    logger.info(
        "Route generated: %s → %s  (%.0f km, %d waypoints)",
        origin, destination,
        route_result.total_distance_km,
        len(route_result.waypoints),
    )

    # ── 2. DB geo query ────────────────────────────────────────────────────
    enriched_events, distances_km = await get_events_near_route(
        route=route_result.waypoints,
        collection=mongo_collection,
        radius_km=radius_km,
        min_label_confidence=min_confidence,
    )
    logger.info("%d events found within %.0f km of route.", len(enriched_events), radius_km)

    # ── 3. Risk aggregation (no ML — reads ev.ml.* from DB) ───────────────
    route_id = f"{origin}→{destination}"
    risk_output = run(route_id, enriched_events, distances_km)

    # ── 4. Build API response ─────────────────────────────────────────────
    # Sort events by distance for display
    events_payload = []
    for ev in sorted(enriched_events, key=lambda e: distances_km.get(e.event_id, 9999)):
        ev_lon, ev_lat = ev.location.coordinates
        events_payload.append({
            "headline":    ev.raw_text[:120].strip() + ("…" if len(ev.raw_text) > 120 else ""),
            "location":    [ev_lat, ev_lon],
            "distance_km": distances_km.get(ev.event_id, 0.0),
            "intensity":   ev.ml.intensity_score,
            "label":       ev.ml.label,
        })

    # Safety score = 1 − risk_score (user-facing, higher = safer)
    safety_score = round(1.0 - risk_output.final_risk_score, 4)

    return {
        "origin":        route_result.origin.display_name,
        "destination":   route_result.destination.display_name,
        "alerts_count":  len(enriched_events),
        "safety_score":  safety_score,
        "status":        risk_output.risk_band,
        "total_distance_km": route_result.total_distance_km,
        "events":        events_payload[:50],   # cap display list
        "risk_detail":   risk_output.explanation,
    }


# ---------------------------------------------------------------------------
# Log4 — analyze_multi_mode()
# Three independent routes → three independent DB queries → comparison output
# ---------------------------------------------------------------------------

def _generate_risk_message(alerts: int, risk_score: float, mode: str) -> str:
    """
    Generate a human-readable risk message for a single transport mode.

    Log7: Now considers risk_score (which includes zone risk), not just alerts.
    """
    mode_label = mode.capitalize()
    if risk_score < 0.25:
        if alerts == 0:
            return f"No significant geopolitical risk along {mode_label} route"
        return f"Minor events detected near {mode_label} corridor — route considered safe"
    if risk_score < 0.50:
        return f"Moderate geopolitical risk on {mode_label} route" + (
            f" — {alerts} event(s) detected" if alerts else " — active risk zones on route"
        )
    if risk_score < 0.75:
        return f"High-risk {mode_label} route — " + (
            f"conflict zones and {alerts} event(s) nearby" if alerts
            else "route passes through active conflict/choke point zones"
        )
    return f"Critical risk on {mode_label} route — " + (
        f"significant conflict activity ({alerts} event(s))" if alerts
        else "multiple high-danger zones on route"
    )


def _build_mode_events_payload(
    enriched_events: list,
    distances_km: dict[str, float],
    limit: int = 5,
) -> list[dict]:
    """Build the events list for a single mode's output block."""
    sorted_evs = sorted(
        enriched_events,
        key=lambda e: distances_km.get(e.event_id, 9999),
    )
    return [
        {
            "headline":    ev.raw_text[:120].strip() + ("…" if len(ev.raw_text) > 120 else ""),
            "location":    [ev.location.coordinates[1], ev.location.coordinates[0]],
            "distance_km": distances_km.get(ev.event_id, 0.0),
            "intensity":   ev.ml.intensity_score,
            "label":       ev.ml.label,
        }
        for ev in sorted_evs[:limit]
    ]


async def analyze_multi_mode(
    origin: str,
    destination: str,
    mongo_collection,
    radius_km: float = 50.0,
    min_confidence: float = 0.50,
) -> dict[str, Any]:
    """
    Log4 entrypoint: compute risk independently for air, sea, and road routes.

    For a SINGLE user request:
      1. Geocode origin + destination ONCE (reused across all modes)
      2. For each mode (air, sea, road):
           a. Generate mode-specific route waypoints
           b. Query MongoDB independently (NO shared event sets)
           c. Compute risk score from stored ML fields (NO ML inference)
           d. Build mode-specific result with message
      3. Return structured comparison dict

    Design guarantees:
      - Events are NOT shared across modes (independent DB queries)
      - ML inference is NEVER called (all fields from ev.ml.*)
      - Any location works — no hardcoded coordinates
      - Sea route goes through real shipping lanes (searoute)
      - Road route follows continental land corridors

    Args:
        origin:           Free-text origin (e.g. "Singapore")
        destination:      Free-text destination (e.g. "Rotterdam, Netherlands")
        mongo_collection: Motor async MongoDB collection
        radius_km:        Buffer search radius around each route
        min_confidence:   Minimum ML label confidence threshold

    Returns:
        Structured comparison dict with "modes" key containing air/sea/road.
    """
    import asyncio
    from core.geo.route import geocode
    from core.routing.air import generate_air_route
    from core.routing.sea import generate_sea_route
    from core.routing.road import generate_road_route
    from core.routing.cache import get_or_generate_route
    from storage.repository import get_events_near_route

    # ── Step 1: Geocode once, reuse across all modes ──────────────────────
    origin_geo = await asyncio.to_thread(geocode, origin)
    if origin_geo is None:
        raise ValueError(f"Could not geocode origin: '{origin}'")

    dest_geo = await asyncio.to_thread(geocode, destination)
    if dest_geo is None:
        raise ValueError(f"Could not geocode destination: '{destination}'")

    logger.info(
        "Multi-mode analysis: %s (%.4f,%.4f) → %s (%.4f,%.4f)",
        origin, origin_geo.lat, origin_geo.lon,
        destination, dest_geo.lat, dest_geo.lon,
    )

    # ── Step 2: Generate routes for all three modes ───────────────────────
    route_generators = {
        "air":  lambda: generate_air_route(origin_geo, dest_geo),
        "sea":  lambda: generate_sea_route(origin_geo, dest_geo),
        "road": lambda: generate_road_route(origin_geo, dest_geo),
    }

    routes = {}
    for mode, gen_fn in route_generators.items():
        try:
            routes[mode] = await asyncio.to_thread(gen_fn)
            logger.info(
                "[%s] route: %d waypoints, %.0f km",
                mode, len(routes[mode].waypoints), routes[mode].total_distance_km,
            )
        except Exception as exc:
            logger.warning("[%s] route generation failed: %s", mode, exc)
            routes[mode] = None

    # ── Step 3: Independent DB queries per mode ───────────────────────────
    mode_results: dict[str, Any] = {}

    for mode, route_result in routes.items():
        if route_result is None:
            mode_results[mode] = {
                "status": "UNKNOWN",
                "safety_score": None,
                "alerts": 0,
                "message": f"{mode.capitalize()} route could not be computed",
                "distance_km": None,
            }
            continue

        try:
            # Independent query — events are NOT shared across modes
            enriched_events, distances_km = await get_events_near_route(
                route=route_result.waypoints,
                collection=mongo_collection,
                radius_km=radius_km,
                min_label_confidence=min_confidence,
            )

            logger.info("[%s] %d events found", mode, len(enriched_events))

            # Risk aggregation — reads ev.ml.*, no ML inference
            route_id = f"{origin}→{destination}[{mode}]"
            risk_output = run(route_id, enriched_events, distances_km)

            safety_score = round(1.0 - risk_output.final_risk_score, 4)
            message = _generate_risk_message(
                len(enriched_events), risk_output.final_risk_score, mode
            )

            mode_result: dict[str, Any] = {
                "status":       risk_output.risk_band,
                "safety_score": safety_score,
                "alerts":       len(enriched_events),
                "message":      message,
                "distance_km":  route_result.total_distance_km,
            }

            # Include top events only if there are alerts
            if enriched_events:
                mode_result["top_events"] = _build_mode_events_payload(
                    enriched_events, distances_km
                )

            mode_results[mode] = mode_result

        except Exception as exc:
            logger.exception("[%s] analysis failed: %s", mode, exc)
            mode_results[mode] = {
                "status": "ERROR",
                "safety_score": None,
                "alerts": 0,
                "message": f"Analysis failed: {exc}",
                "distance_km": routes[mode].total_distance_km if routes[mode] else None,
            }

    # ── Step 4: Derive recommendation ─────────────────────────────────────
    valid_modes = {m: r for m, r in mode_results.items() if r.get("safety_score") is not None}
    if valid_modes:
        safest_mode = max(valid_modes, key=lambda m: valid_modes[m]["safety_score"])
    else:
        safest_mode = "air"

    return {
        "origin":             origin_geo.display_name,
        "destination":        dest_geo.display_name,
        "recommended_mode":   safest_mode,
        "modes":              mode_results,
        "analyzed_at":        datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Log5 — Evidence-Enriched Multi-Mode Analysis
# Zone-aware + verified links + images + full provenance
# ---------------------------------------------------------------------------

def _build_evidence_payload(
    enriched_events: list,
    distances_km: dict[str, float],
    limit: int = 5,
) -> list[dict]:
    """
    Build evidence-enriched event payload for a single mode (Log5).

    Each event includes:
      - headline, source_url, image_url, location, zone, confidence
      - verification metadata (credibility_score, publisher)
    """
    sorted_evs = sorted(
        enriched_events,
        key=lambda e: e.ml.intensity_score * (1.0 / max(distances_km.get(e.event_id, 1.0), 0.1)),
        reverse=True,
    )

    evidence = []
    for ev in sorted_evs[:limit]:
        ev_lon, ev_lat = ev.location.coordinates
        zones = getattr(ev, "zones", [])
        source_url = getattr(ev, "source_url", "")
        image_url = getattr(ev, "image_url", None)
        publisher = getattr(ev, "publisher", "")
        source_urls = getattr(ev, "source_urls", []) or ([source_url] if source_url else [])
        publishers = getattr(ev, "publishers", []) or ([publisher] if publisher else [])
        corroboration_count = getattr(ev, "corroboration_count", 1) or 1
        corroboration_score = getattr(ev, "corroboration_score", 0.0) or 0.0
        combined_credibility = getattr(ev, "combined_credibility", 0.0) or 0.0

        # If verification metadata exists, use it
        verification = getattr(ev, "verification", None)
        if verification and hasattr(verification, "credibility_score"):
            credibility = verification.credibility_score
        else:
            credibility = None

        evidence.append({
            "headline":    ev.raw_text[:150].strip() + ("…" if len(ev.raw_text) > 150 else ""),
            "source_url":  source_url,
            "source_urls": source_urls,
            "image_url":   image_url,
            "publisher":   publisher,
            "publishers":  publishers,
            "location":    [ev_lat, ev_lon],
            "distance_km": distances_km.get(ev.event_id, 0.0),
            "zone":        zones[0] if zones else None,
            "zones":       zones,
            "confidence":  round(ev.ml.label_confidence, 4),
            "intensity":   round(ev.ml.intensity_score, 4),
            "label":       ev.ml.label,
            "credibility": round(credibility, 4) if credibility is not None else None,
            "combined_credibility": round(combined_credibility, 4),
            "corroboration_count": int(corroboration_count),
            "corroboration_score": round(corroboration_score, 4),
            "canonical_event_id": getattr(ev, "canonical_event_id", None) or ev.event_id,
            "published_at": ev.published_at.isoformat() if hasattr(ev.published_at, "isoformat") else str(ev.published_at),
        })

    return evidence


async def analyze_multi_mode_v5(
    origin: str,
    destination: str,
    mongo_collection,
    radius_km: float = 50.0,
    min_confidence: float = 0.50,
) -> dict[str, Any]:
    """
    Log5+Log7 entrypoint: evidence-enriched, zone-aware multi-mode risk analysis.

    Log7 upgrade: zones now CONTRIBUTE to risk score directly.
    A sea route through Hormuz/Red Sea gets HIGH risk even with zero DB events.
    Also filters out source="seed" events.
    """
    import asyncio
    from core.geo.route import geocode
    from core.routing.air import generate_air_route
    from core.routing.sea import generate_sea_route
    from core.routing.road import generate_road_route
    from core.geo.zones import check_route_zone_intersections, compute_zone_risk
    from storage.repository import get_events_near_route

    # ── Step 1: Geocode once ──────────────────────────────────────────────
    origin_geo = await asyncio.to_thread(geocode, origin)
    if origin_geo is None:
        raise ValueError(f"Could not geocode origin: '{origin}'")

    dest_geo = await asyncio.to_thread(geocode, destination)
    if dest_geo is None:
        raise ValueError(f"Could not geocode destination: '{destination}'")

    logger.info(
        "Log10 multi-mode analysis: %s (%.4f,%.4f) → %s (%.4f,%.4f)",
        origin, origin_geo.lat, origin_geo.lon,
        destination, dest_geo.lat, dest_geo.lon,
    )

    # ── Step 2: Generate routes in PARALLEL (Log10) ──────────────────────
    async def _gen_route(mode_name, gen_fn):
        try:
            return mode_name, await asyncio.to_thread(gen_fn)
        except Exception as exc:
            logger.warning("[%s] route generation failed: %s", mode_name, exc)
            return mode_name, None

    route_tasks = [
        _gen_route("air",  lambda: get_or_generate_route("air", origin_geo, dest_geo, lambda: generate_air_route(origin_geo, dest_geo))),
        _gen_route("sea",  lambda: get_or_generate_route("sea", origin_geo, dest_geo, lambda: generate_sea_route(origin_geo, dest_geo))),
        _gen_route("road", lambda: get_or_generate_route("road", origin_geo, dest_geo, lambda: generate_road_route(origin_geo, dest_geo))),
    ]
    route_results = await asyncio.gather(*route_tasks)
    routes = {name: result for name, result in route_results}

    # ── Step 3: Independent analysis per mode with zone-aware risk ────────
    mode_results: dict[str, Any] = {}

    def _risk_band(score: float) -> str:
        if score < 0.25: return "LOW"
        if score < 0.50: return "MEDIUM"
        if score < 0.75: return "HIGH"
        return "CRITICAL"

    for mode, route_result in routes.items():
        if route_result is None:
            mode_results[mode] = {
                "status": "UNKNOWN",
                "alerts": 0,
                "risk_score": None,
                "safety_score": None,
                "message": f"{mode.capitalize()} route could not be computed",
                "distance_km": None,
                "zone_intersections": [],
                "events": [],
            }
            continue

        try:
            # Log10: Mode-aware zone intersection detection
            zone_intersections = check_route_zone_intersections(
                route_result.waypoints, transport_mode=mode
            )

            # Zone base risk (Log7)
            zone_risk = compute_zone_risk(zone_intersections)

            # Independent DB query
            enriched_events, distances_km = await get_events_near_route(
                route=route_result.waypoints,
                collection=mongo_collection,
                radius_km=radius_km,
                min_label_confidence=min_confidence,
            )

            # Filter out seed events (Log7: purge synthetic data)
            enriched_events = [e for e in enriched_events if getattr(e, "source", "") != "seed"]
            distances_km = {eid: d for eid, d in distances_km.items()
                           if any(e.event_id == eid for e in enriched_events)}

            logger.info("[%s] %d real events, %d zone intersections, zone_risk=%.2f",
                        mode, len(enriched_events), len(zone_intersections), zone_risk)

            # Event-based risk aggregation
            route_id = f"{origin}→{destination}[{mode}]"
            risk_output = run(route_id, enriched_events, distances_km)
            event_risk = risk_output.final_risk_score

            # Log8: Zones as MODIFIERS, not absolute risk
            # Log10: Added source corroboration multiplier
            if event_risk > 0 and zone_risk > 0:
                final_risk = (event_risk * 0.70) + (zone_risk * 0.30)
            elif event_risk > 0:
                final_risk = event_risk
            elif zone_risk > 0:
                final_risk = zone_risk * 0.40
            else:
                final_risk = 0.0

            # Log10: Source corroboration — multi-source confirmation boosts risk
            if enriched_events:
                unique_sources = set()
                max_corroboration = 1
                for e in enriched_events:
                    publishers = getattr(e, "publishers", []) or []
                    if publishers:
                        unique_sources.update(publishers)
                    else:
                        src = getattr(e, 'source', '') or ''
                        if src:
                            unique_sources.add(src)
                    max_corroboration = max(
                        max_corroboration,
                        int(getattr(e, "corroboration_count", 1) or 1),
                    )
                n_sources = len(unique_sources)
                if n_sources >= 3 or max_corroboration >= 3:
                    final_risk = min(final_risk * 1.15, 1.0)  # 3+ sources = high corroboration
                elif n_sources <= 1 and len(enriched_events) <= 2:
                    final_risk *= 0.90  # single source, few events = lower confidence

            final_risk = round(min(final_risk, 1.0), 4)
            safety_score = round(1.0 - final_risk, 4)
            risk_band = _risk_band(final_risk)

            message = _generate_risk_message(
                len(enriched_events), final_risk, mode
            )

            # Build evidence payload
            evidence = _build_evidence_payload(enriched_events, distances_km, limit=5)

            mode_results[mode] = {
                "status":             risk_band,
                "alerts":             len(enriched_events),
                "risk_score":         final_risk,
                "safety_score":       safety_score,
                "distance_km":        route_result.total_distance_km,
                "message":            message,
                "zone_risk":          round(zone_risk, 4),
                "event_risk":         round(event_risk, 4),
                "zone_intersections": zone_intersections,
                "events":             evidence,
            }

        except Exception as exc:
            logger.exception("[%s] analysis failed: %s", mode, exc)
            mode_results[mode] = {
                "status": "ERROR",
                "alerts": 0,
                "risk_score": None,
                "message": f"Analysis failed: {exc}",
                "distance_km": routes[mode].total_distance_km if routes[mode] else None,
                "zone_intersections": [],
                "events": [],
            }

    # ── Step 4: Recommendation ────────────────────────────────────────────
    valid_modes = {m: r for m, r in mode_results.items() if r.get("safety_score") is not None}
    if valid_modes:
        safest_mode = max(valid_modes, key=lambda m: valid_modes[m]["safety_score"])
    else:
        safest_mode = "air"

    return {
        "origin":             origin_geo.display_name,
        "destination":        dest_geo.display_name,
        "recommended_mode":   safest_mode,
        "modes":              mode_results,
        "analyzed_at":        datetime.now(timezone.utc).isoformat(),
    }
