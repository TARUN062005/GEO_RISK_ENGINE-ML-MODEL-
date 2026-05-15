"""
scripts/test_realtime.py
------------------------
Log5 Real-Time System Test Runner

Tests:
  1. Multi-source ingestion (one cycle)
  2. Source verification scoring
  3. Zone matching
  4. Evidence-enriched multi-mode analysis

Usage:
    python scripts/test_realtime.py
    python scripts/test_realtime.py --origin "Mumbai, India" --destination "Rotterdam, Netherlands"
    python scripts/test_realtime.py --ingest-only
    python scripts/test_realtime.py --verify-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import io

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Test Zone Matching
# ---------------------------------------------------------------------------

def test_zones():
    """Test the zone matching system with known coordinates."""
    from core.geo.zones import match_point_to_zones, check_route_zone_intersections

    print("\n" + "=" * 70)
    print("  LOG5 — Zone Matching Test")
    print("=" * 70)

    test_points = [
        ("Strait of Hormuz",  26.5, 56.0),
        ("Red Sea (Houthi zone)", 14.0, 43.0),
        ("Central Ukraine",    48.5, 36.0),
        ("Gaza",               31.4, 34.4),
        ("Singapore (safe)",    1.3, 103.8),
        ("London (safe)",      51.5, -0.1),
    ]

    for name, lat, lon in test_points:
        matches = match_point_to_zones(lat, lon)
        if matches:
            zone_names = [m["zone"] for m in matches]
            print(f"  [{len(matches)} zone(s)] {name:30s} -> {', '.join(zone_names)}")
        else:
            print(f"  [CLEAR   ] {name:30s} -> No zone match")

    # Test route-zone intersection
    print("\n  Route zone intersection: Mumbai -> Rotterdam (sea route)")
    test_route = [
        (19.0, 72.8),   # Mumbai
        (18.0, 66.0),   # Arabian Sea
        (14.0, 43.0),   # Red Sea
        (30.5, 32.3),   # Suez Canal
        (35.0, 25.0),   # Mediterranean
        (51.9, 4.5),    # Rotterdam
    ]
    intersections = check_route_zone_intersections(test_route)
    for zi in intersections:
        print(f"    -> {zi['zone']:30s}  ({zi['category']:12s})  dist={zi['min_distance_km']:.0f} km")

    if not intersections:
        print("    (no zone intersections)")

    print()


# ---------------------------------------------------------------------------
# 2. Test Source Verification
# ---------------------------------------------------------------------------

def test_verification():
    """Test source credibility scoring."""
    from ingestion.verification import verify_source

    print("\n" + "=" * 70)
    print("  LOG5 — Source Verification Test")
    print("=" * 70)

    test_sources = [
        ("https://www.reuters.com/world/conflict-article", "Reuters"),
        ("https://www.bbc.co.uk/news/world-123456", "BBC News"),
        ("https://www.aljazeera.com/news/2026/crisis", "Al Jazeera"),
        ("https://www.cnn.com/breaking/news", "CNN"),
        ("https://unknown-blog.xyz/article", "Random Blog"),
        ("https://state.gov/press-release", "US State Dept"),
        ("", "GDELT Project"),
    ]

    for url, publisher in test_sources:
        v = verify_source(source_url=url, publisher=publisher)
        tier_icon = {"tier1": "[T1]", "tier2": "[T2]", "tier3": "[T3]", "unknown": "[??]"}
        icon = tier_icon.get(v.credibility_tier, "[??]")
        print(f"  {icon} {v.credibility_score:.2f}  {publisher:20s}  {v.domain:30s}")

    print()


# ---------------------------------------------------------------------------
# 3. Test Ingestion (one cycle)
# ---------------------------------------------------------------------------

async def test_ingestion(mongo_uri: str):
    """Run one ingestion cycle and report stats."""
    from ingestion.realtime_worker import ingest_cycle
    from motor.motor_asyncio import AsyncIOMotorClient

    print("\n" + "=" * 70)
    print("  LOG5 — Real-Time Ingestion Test (one cycle)")
    print("=" * 70)

    client = AsyncIOMotorClient(mongo_uri)
    collection = client["geo_risk"]["geo_events"]

    try:
        # Count before
        before = await collection.count_documents({})
        print(f"  Events in DB before: {before}")

        # Run one cycle
        stats = await ingest_cycle(collection)

        # Count after
        after = await collection.count_documents({})

        print(f"\n  Ingestion Results:")
        print(f"    Fetched  : {stats['fetched']}")
        print(f"    Enriched : {stats['enriched']}")
        print(f"    Written  : {stats['written']}")
        print(f"    Skipped  : {stats['skipped']}")
        print(f"    Errors   : {stats['errors']}")
        print(f"    Events in DB after: {after} (+{after - before} new)")

        # Show some recent events with verification
        print(f"\n  Recent verified events:")
        cursor = collection.find(
            {"verification.credibility_score": {"$exists": True}},
        ).sort("ingested_at", -1).limit(5)

        async for doc in cursor:
            v = doc.get("verification", {})
            zones = doc.get("zones", [])
            text = doc.get("raw_text", "")[:80]
            credibility = v.get("credibility_score", "?")
            publisher = v.get("publisher", "?")
            source_url = doc.get("source_url", "")[:60]
            print(f"    [{credibility:.2f}] {publisher:15s} | {text}...")
            if source_url:
                print(f"           URL: {source_url}")
            if zones:
                print(f"           Zones: {', '.join(zones)}")

    finally:
        client.close()

    print()


# ---------------------------------------------------------------------------
# 4. Test Multi-Mode V5 Analysis
# ---------------------------------------------------------------------------

async def test_analysis(origin: str, destination: str, mongo_uri: str):
    """Run the Log5 evidence-enriched analysis."""
    from motor.motor_asyncio import AsyncIOMotorClient
    from core.orchestrator import analyze_multi_mode_v5

    print("\n" + "=" * 70)
    print(f"  LOG5 — Evidence-Enriched Analysis")
    print(f"  {origin}  ->  {destination}")
    print("=" * 70)

    client = AsyncIOMotorClient(mongo_uri)
    try:
        result = await analyze_multi_mode_v5(
            origin=origin,
            destination=destination,
            mongo_collection=client["geo_risk"]["geo_events"],
            radius_km=50.0,
        )

        print(f"\n  Origin      : {result['origin'][:70]}")
        print(f"  Destination : {result['destination'][:70]}")
        print(f"  Recommended : >> {result['recommended_mode'].upper()} << (lowest risk)")
        print(f"  Analyzed at : {result['analyzed_at']}")
        print()

        for mode in ["air", "sea", "road"]:
            m = result["modes"].get(mode, {})
            status = m.get("status", "UNKNOWN")
            risk = m.get("risk_score")
            safety = m.get("safety_score")
            alerts = m.get("alerts", 0)
            dist = m.get("distance_km")

            status_icons = {"LOW": "[OK]", "MEDIUM": "[!!]", "HIGH": "[XX]", "CRITICAL": "[XX]"}
            icon = status_icons.get(status, "[??]")

            print(f"  {mode.upper():5s} {icon} {status:8s}  "
                  f"risk={risk if risk is not None else 'N/A':>6}  "
                  f"alerts={alerts:3d}  "
                  f"dist={'%,.0f km' % dist if dist else 'N/A':>10}")
            print(f"        {m.get('message', '')}")

            # Zone intersections
            zones = m.get("zone_intersections", [])
            if zones:
                print(f"        Zones crossed:")
                for z in zones[:3]:
                    print(f"          -> {z['zone']} ({z['category']}, {z['min_distance_km']:.0f} km)")

            # Evidence events
            events = m.get("events", [])
            if events:
                print(f"        Top evidence:")
                for ev in events[:3]:
                    print(f"          [{ev['label']:10s}] {ev['headline'][:55]}")
                    if ev.get("source_url"):
                        print(f"                      URL: {ev['source_url'][:60]}")
                    if ev.get("image_url"):
                        print(f"                      IMG: {ev['image_url'][:60]}")
                    if ev.get("zone"):
                        print(f"                      Zone: {ev['zone']}")
                    print(f"                      confidence={ev['confidence']:.2f}  "
                          f"credibility={ev.get('credibility', 'N/A')}")
            print()

    except ValueError as exc:
        print(f"  GEOCODING ERROR: {exc}")
    except Exception as exc:
        logger.exception("Analysis error")
        print(f"  ERROR: {exc}")
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geo Risk Engine — Log5 Test Suite")
    parser.add_argument("--uri", default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--origin", default="Mumbai, India")
    parser.add_argument("--destination", default="Rotterdam, Netherlands")
    parser.add_argument("--ingest-only", action="store_true", help="Only run ingestion test")
    parser.add_argument("--verify-only", action="store_true", help="Only run verification test")
    parser.add_argument("--zones-only", action="store_true", help="Only run zone matching test")
    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis test")
    args = parser.parse_args()

    async def main():
        if args.verify_only:
            test_verification()
            return
        if args.zones_only:
            test_zones()
            return
        if args.ingest_only:
            await test_ingestion(args.uri)
            return
        if args.analyze_only:
            await test_analysis(args.origin, args.destination, args.uri)
            return

        # Full test suite
        test_zones()
        test_verification()
        await test_ingestion(args.uri)
        await test_analysis(args.origin, args.destination, args.uri)

        print("=" * 70)
        print("  LOG5 TEST SUITE COMPLETE")
        print("=" * 70)

    asyncio.run(main())
