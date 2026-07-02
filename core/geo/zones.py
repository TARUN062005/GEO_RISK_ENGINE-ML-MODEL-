"""
core/geo/zones.py
-----------------
Region-Level Geo Zone Model (Log5)

Defines geopolitical zones as named circular regions on the globe.
Events are matched to zones by proximity (center + radius).
Route segments are checked for zone intersection.

Zones are NOT hardcoded event locations — they define
areas of strategic interest for risk assessment.

Used by:
  - ingestion worker: tag events with matched zones
  - orchestrator: check route-zone intersections for multi-mode analysis
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zone Definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeoZone:
    """A named circular region on the globe."""
    name: str
    center_lat: float
    center_lon: float
    radius_km: float
    category: str          # "maritime" | "conflict" | "choke_point" | "sanctions" | "airspace"
    description: str = ""
    applies_to: tuple[str, ...] = ("air", "sea", "road")  # Log10: mode-specific zones


# ---------------------------------------------------------------------------
# Global Zone Registry
# ---------------------------------------------------------------------------

ZONES: list[GeoZone] = [
    # Maritime chokepoints
    GeoZone(
        name="Strait of Hormuz",
        center_lat=26.5, center_lon=56.0, radius_km=300,
        category="choke_point",
        description="World's most important oil transit chokepoint; ~20% of global oil passes through",
        applies_to=("sea",),  # air overflies at cruise altitude; road does not transit the strait
    ),
    GeoZone(
        name="Hormuz Tanker Disruption Area",
        center_lat=26.7, center_lon=55.7, radius_km=220,
        category="maritime",
        description="Tanker seizure, harassment, mine, and naval escalation exposure near Hormuz",
        applies_to=("sea",),
    ),
    GeoZone(
        name="Red Sea / Bab el-Mandeb",
        center_lat=13.5, center_lon=42.5, radius_km=500,
        category="maritime",
        description="Critical shipping lane linking Mediterranean to Indian Ocean; Houthi attack zone",
        applies_to=("sea",),
    ),
    GeoZone(
        name="Bab el-Mandeb Shipping Advisory Area",
        center_lat=12.8, center_lon=43.3, radius_km=280,
        category="maritime",
        description="Shipping advisories for drone, missile, piracy, and naval escort disruptions",
        applies_to=("sea",),
    ),
    GeoZone(
        name="Suez Canal",
        center_lat=30.5, center_lon=32.3, radius_km=150,
        category="choke_point",
        description="Connects Mediterranean to Red Sea; handles ~12% of global trade",
        applies_to=("sea",),
    ),
    GeoZone(
        name="Strait of Malacca",
        center_lat=2.5, center_lon=101.0, radius_km=400,
        category="choke_point",
        description="Busiest shipping strait in the world; connects Indian and Pacific Oceans",
        applies_to=("sea",),
    ),
    GeoZone(
        name="Gulf of Aden",
        center_lat=12.0, center_lon=47.0, radius_km=400,
        category="maritime",
        description="Major piracy and conflict zone; approach to Red Sea",
        applies_to=("sea",),
    ),
    GeoZone(
        name="Panama Canal",
        center_lat=9.1, center_lon=-79.7, radius_km=100,
        category="choke_point",
        description="Critical Americas-Asia trade route; capacity constrained",
        applies_to=("sea",),
    ),
    GeoZone(
        name="South China Sea",
        center_lat=14.0, center_lon=115.0, radius_km=800,
        category="maritime",
        description="Disputed maritime zone; $3.4T annual trade passes through",
        applies_to=("sea", "air"),
    ),
    GeoZone(
        name="Taiwan Strait",
        center_lat=24.0, center_lon=119.5, radius_km=200,
        category="choke_point",
        description="Critical semiconductor supply chain corridor; high geopolitical tension",
        applies_to=("sea", "air"),
    ),
    GeoZone(
        name="Black Sea",
        center_lat=43.5, center_lon=34.0, radius_km=400,
        category="maritime",
        description="Active conflict zone; grain export corridor; Russian naval presence",
        applies_to=("sea", "air"),
    ),

    # Active conflict zones (all modes)
    GeoZone(
        name="Ukraine War Zone",
        center_lat=48.5, center_lon=36.0, radius_km=500,
        category="conflict",
        description="Active armed conflict; no-fly zone for commercial aviation",
    ),
    GeoZone(
        name="Gaza / Southern Israel",
        center_lat=31.4, center_lon=34.4, radius_km=150,
        category="conflict",
        description="Active armed conflict; closed airspace; humanitarian crisis",
    ),
    GeoZone(
        name="Eastern Libya",
        center_lat=32.0, center_lon=20.0, radius_km=300,
        category="conflict",
        description="Civil conflict zone; divided governance",
    ),
    GeoZone(
        name="Somalia / Horn of Africa",
        center_lat=5.0, center_lon=46.0, radius_km=500,
        category="conflict",
        description="Al-Shabaab activity; piracy base; humanitarian crisis",
    ),
    GeoZone(
        name="Sahel Region",
        center_lat=15.0, center_lon=2.0, radius_km=800,
        category="conflict",
        description="Jihadist insurgency; multiple military coups; Mali, Niger, Burkina Faso",
    ),
    GeoZone(
        name="Myanmar Conflict Zone",
        center_lat=20.0, center_lon=96.5, radius_km=400,
        category="conflict",
        description="Civil war; military junta; ethnic armed organizations",
    ),
    GeoZone(
        name="Afghanistan",
        center_lat=33.9, center_lon=67.7, radius_km=400,
        category="conflict",
        description="Taliban governance; terrorism risk; ISIS-K activity",
    ),
    GeoZone(
        name="Yemen",
        center_lat=15.5, center_lon=48.0, radius_km=400,
        category="conflict",
        description="Civil war; Houthi missile attacks on shipping; humanitarian crisis",
    ),
    GeoZone(
        name="Syria / Northern Iraq",
        center_lat=35.5, center_lon=40.0, radius_km=400,
        category="conflict",
        description="Post-conflict instability; ISIS remnants; Turkish operations",
    ),
    GeoZone(
        name="Sudan / Darfur",
        center_lat=13.0, center_lon=30.0, radius_km=500,
        category="conflict",
        description="Active civil war; RSF vs SAF; mass displacement",
    ),

    # Sanctions zones
    GeoZone(
        name="North Korea Buffer",
        center_lat=39.0, center_lon=127.5, radius_km=300,
        category="sanctions",
        description="Heavily sanctioned; no commercial transit; nuclear threat",
    ),
    GeoZone(
        name="Iran Sanctions Zone",
        center_lat=32.4, center_lon=53.7, radius_km=600,
        category="sanctions",
        description="Comprehensive US/EU sanctions; limited commercial routing",
    ),
    GeoZone(
        name="Crimea / Annexed Territories",
        center_lat=45.3, center_lon=34.0, radius_km=200,
        category="sanctions",
        description="Annexed territory; comprehensive sanctions; no commercial transit",
    ),

    # ── Log10: Airspace-specific zones ────────────────────────────────────
    GeoZone(
        name="Ukraine Airspace Closure",
        center_lat=48.5, center_lon=36.0, radius_km=600,
        category="airspace",
        description="NOTAM: complete airspace closure due to active conflict; missile risk",
        applies_to=("air",),
    ),
    GeoZone(
        name="Iran-Iraq Missile Corridor",
        center_lat=33.0, center_lon=47.0, radius_km=350,
        category="airspace",
        description="Active missile exchanges; SAM threat; Iranian air defense active",
        applies_to=("air",),
    ),
    GeoZone(
        name="Eastern Mediterranean NOTAM",
        center_lat=34.5, center_lon=35.0, radius_km=250,
        category="airspace",
        description="Temporary airspace restrictions due to regional conflict; flight rerouting required",
        applies_to=("air",),
    ),
    GeoZone(
        name="Red Sea Drone Corridor",
        center_lat=16.0, center_lon=42.0, radius_km=300,
        category="airspace",
        description="Houthi drone/missile threat to aviation; recommended altitude > FL400",
        applies_to=("air",),
    ),
    GeoZone(
        name="Persian Gulf Air Defense Corridor",
        center_lat=27.0, center_lon=51.5, radius_km=350,
        category="airspace",
        description="Elevated aviation risk from air defense alerting and missile interception activity",
        applies_to=("air",),
    ),
]


# ---------------------------------------------------------------------------
# Haversine Helper (local, avoids circular import)
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Zone Matching
# ---------------------------------------------------------------------------

def match_point_to_zones(
    lat: float,
    lon: float,
    zones: list[GeoZone] | None = None,
) -> list[dict]:
    """
    Match a (lat, lon) point to all zones whose radius contains it.

    Returns list of dicts:
      [{"zone": name, "category": cat, "distance_km": float}, ...]

    Sorted by distance (closest first).
    """
    if zones is None:
        zones = ZONES

    matches = []
    for z in zones:
        dist = _haversine_km(lat, lon, z.center_lat, z.center_lon)
        if dist <= z.radius_km:
            matches.append({
                "zone": z.name,
                "category": z.category,
                "distance_km": round(dist, 1),
            })

    return sorted(matches, key=lambda m: m["distance_km"])


def match_event_to_zones(
    event_lat: float,
    event_lon: float,
) -> list[str]:
    """
    Return zone names that contain the event point.
    Convenience wrapper for ingestion worker.
    """
    return [m["zone"] for m in match_point_to_zones(event_lat, event_lon)]


# ---------------------------------------------------------------------------
# Route-Zone Intersection (Log10: mode-aware)
# ---------------------------------------------------------------------------

def check_route_zone_intersections(
    waypoints: list[tuple[float, float]],
    zones: list[GeoZone] | None = None,
    transport_mode: str | None = None,
) -> list[dict]:
    """
    Check which zones a route passes through or near.

    Log10: Optional transport_mode filter — only returns zones
    that apply to the given mode (air, sea, road).

    Returns list of zone intersection dicts sorted by distance.
    """
    if zones is None:
        zones = ZONES

    intersections = []
    for z in zones:
        # Log10: Skip zones that don't apply to this transport mode
        if transport_mode and transport_mode not in z.applies_to:
            continue

        min_dist = float("inf")
        closest_idx = 0

        for i, (lat, lon) in enumerate(waypoints):
            dist = _haversine_km(lat, lon, z.center_lat, z.center_lon)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
            # Early exit: once we're inside the zone, no need to find exact closest
            if dist <= z.radius_km:
                break

        if min_dist <= z.radius_km:
            intersections.append({
                "zone": z.name,
                "category": z.category,
                "description": z.description,
                "min_distance_km": round(min_dist, 1),
                "intersects": True,
                "closest_waypoint_idx": closest_idx,
            })

    return sorted(intersections, key=lambda i: i["min_distance_km"])


def get_zone_risk_factor(zone_name: str) -> float:
    """
    Return a risk multiplier for a zone category.
    Used to boost risk scores for events inside known danger zones.
    """
    for z in ZONES:
        if z.name == zone_name:
            factors = {
                "conflict": 1.20,
                "choke_point": 1.10,
                "maritime": 1.05,
                "sanctions": 1.15,
                "airspace": 1.25,  # Log10
            }
            return factors.get(z.category, 1.0)
    return 1.0


# ---------------------------------------------------------------------------
# Log7 — Zone Base Risk Scores (Log10: added airspace zones)
# ---------------------------------------------------------------------------

ZONE_BASE_RISK: dict[str, float] = {
    # Conflict zones — highest base risk
    "Ukraine War Zone":           0.90,
    "Gaza / Southern Israel":     0.95,
    "Yemen":                      0.80,
    "Sudan / Darfur":             0.75,
    "Afghanistan":                0.70,
    "Somalia / Horn of Africa":   0.65,
    "Syria / Northern Iraq":      0.65,
    "Myanmar Conflict Zone":      0.55,
    "Eastern Libya":              0.55,
    "Sahel Region":               0.50,

    # Maritime / chokepoints — elevated risk
    "Red Sea / Bab el-Mandeb":    0.70,
    "Bab el-Mandeb Shipping Advisory Area": 0.68,
    "Strait of Hormuz":           0.65,
    "Hormuz Tanker Disruption Area": 0.66,
    "Gulf of Aden":               0.60,
    "Black Sea":                  0.55,
    "South China Sea":            0.40,
    "Taiwan Strait":              0.45,
    "Suez Canal":                 0.30,
    "Strait of Malacca":          0.20,
    "Panama Canal":               0.15,

    # Sanctions zones
    "North Korea Buffer":         0.80,
    "Iran Sanctions Zone":        0.60,
    "Crimea / Annexed Territories": 0.70,

    # Log10: Airspace zones (AIR-only)
    "Ukraine Airspace Closure":   0.95,
    "Iran-Iraq Missile Corridor": 0.75,
    "Eastern Mediterranean NOTAM": 0.50,
    "Red Sea Drone Corridor":     0.60,
    "Persian Gulf Air Defense Corridor": 0.45,
}


def compute_zone_risk(zone_intersections: list[dict]) -> float:
    """
    Compute aggregate zone-based risk for a route.

    Takes the list of zone intersection dicts from check_route_zone_intersections()
    and returns a combined risk score [0, 1].

    Algorithm:
      - Take the MAX zone base risk (dominant threat)
      - Add 0.05 for each additional zone crossed (compounding)
      - Cap at 0.95
    """
    if not zone_intersections:
        return 0.0

    zone_risks = []
    for zi in zone_intersections:
        zone_name = zi.get("zone", "")
        base = ZONE_BASE_RISK.get(zone_name, 0.0)
        if base > 0:
            zone_risks.append(base)

    if not zone_risks:
        return 0.0

    # Max zone risk + compounding for multiple zones
    max_risk = max(zone_risks)
    compound = min(0.20, (len(zone_risks) - 1) * 0.05)  # +5% per extra zone, capped
    return min(max_risk + compound, 0.95)
