"""
scripts/run_pipeline.py
-----------------------
End-to-End Pipeline Runner (Log6 — wired to Log5 orchestrator)

FIXED in Log6:
  - Calls analyze_multi_mode_v5() (NOT the old analyze_multi_mode)
  - Shows zone intersections per mode
  - Shows verified source_url + image_url + credibility
  - DB diagnostics: event count, latest timestamps, source presence
  - Accepts CLI input (origin, destination)

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --origin "Mumbai, India" --destination "Dubai, UAE"
    python scripts/run_pipeline.py --single
    python scripts/run_pipeline.py --uri mongodb://localhost:27017
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import io

# Force UTF-8 output on Windows to support Unicode box-drawing / emoji
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_ROUTES = [
    ("USA, Washington D.C",  "India, New Delhi"),
    ("Delhi, India",         "London, UK"),
    ("New York, USA",        "Tokyo, Japan"),
]

MULTI_MODE_ROUTES = [
    ("Mumbai, India",        "Dubai, UAE"),
    ("USA, Washington D.C",  "India, New Delhi"),
    ("Singapore",            "Rotterdam, Netherlands"),
]

_STATUS_ICON = {
    "LOW":      "[OK]",
    "MEDIUM":   "[!!]",
    "HIGH":     "[XX]",
    "CRITICAL": "[XX]",
    "UNKNOWN":  "[??]",
    "ERROR":    "[ER]",
}


# ---------------------------------------------------------------------------
# Single-mode runner (Log3)
# ---------------------------------------------------------------------------

async def run_single(origin: str, destination: str, mongo_uri: str) -> dict:
    from motor.motor_asyncio import AsyncIOMotorClient
    from core.orchestrator import analyze_route_real
    client = AsyncIOMotorClient(mongo_uri)
    try:
        return await analyze_route_real(
            origin=origin,
            destination=destination,
            mongo_collection=client["geo_risk"]["geo_events"],
            radius_km=50.0,
        )
    finally:
        client.close()


async def run_single_batch(routes: list[tuple[str, str]], mongo_uri: str) -> None:
    print("\n" + "=" * 70)
    print("  GEO RISK ENGINE -- End-to-End Pipeline Test")
    print("=" * 70)

    for origin, destination in routes:
        print(f"\n{'-' * 70}")
        print(f"  ROUTE: {origin}  ->  {destination}")
        print(f"{'-' * 70}")

        try:
            result = await run_single(origin, destination, mongo_uri)
            icon = _STATUS_ICON.get(result["status"], "[??]")
            print(f"  Status       : {icon} {result['status']}")
            print(f"  Safety Score : {result['safety_score']:.3f}  (1.0 = safest)")
            print(f"  Alerts Found : {result['alerts_count']}")
            print(f"  Distance     : {result.get('total_distance_km', '?')} km")
            print(f"  Origin       : {result['origin'][:80]}")
            print(f"  Destination  : {result['destination'][:80]}")

            if result["events"]:
                print(f"\n  Top events:")
                for ev in result["events"][:3]:
                    print(f"    [{ev['label'].upper():10s}] {ev['headline'][:60]}")
                    print(f"              dist={ev['distance_km']:.1f} km  intensity={ev['intensity']:.3f}")
            else:
                print("  No events found near this route.")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\n{'=' * 70}\n")


# ---------------------------------------------------------------------------
# DB Diagnostics (Log6)
# ---------------------------------------------------------------------------

async def _print_db_diagnostics(mongo_uri: str) -> None:
    """Print DB state: event count, sources, latest timestamps, schema versions."""
    from motor.motor_asyncio import AsyncIOMotorClient

    client = AsyncIOMotorClient(mongo_uri)
    collection = client["geo_risk"]["geo_events"]

    try:
        total = await collection.count_documents({})
        with_url = await collection.count_documents({"source_url": {"$ne": ""}})
        with_verification = await collection.count_documents({"verification": {"$exists": True}})
        with_zones = await collection.count_documents({"zones": {"$exists": True, "$ne": []}})
        schema_v5 = await collection.count_documents({"schema_version": "5"})
        schema_v2 = await collection.count_documents({"schema_version": "2"})

        # Source breakdown
        source_pipeline = [
            {"$group": {"_id": "$source", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        source_counts = {}
        async for doc in collection.aggregate(source_pipeline):
            source_counts[doc["_id"]] = doc["count"]

        # Latest event
        latest_doc = await collection.find_one(
            sort=[("ingested_at", -1)],
            projection={"ingested_at": 1, "source": 1, "raw_text": 1},
        )

        print("\n  " + "-" * 66)
        print("  DB DIAGNOSTICS")
        print("  " + "-" * 66)
        print(f"    Total events         : {total}")
        print(f"    Schema v5 (Log5)     : {schema_v5}")
        print(f"    Schema v2 (old seed) : {schema_v2}")
        print(f"    With source_url      : {with_url}")
        print(f"    With verification    : {with_verification}")
        print(f"    With zones           : {with_zones}")
        print(f"    Sources              : {source_counts}")

        if latest_doc:
            print(f"    Latest event at      : {latest_doc.get('ingested_at', '?')}")
            print(f"    Latest source        : {latest_doc.get('source', '?')}")
            txt = latest_doc.get('raw_text', '')[:80]
            print(f"    Latest text          : {txt}...")
        else:
            print("    [!] DB IS EMPTY — run the ingestion worker first!")

        if total == 0:
            print("\n    >>> WARNING: No events in DB. Run:")
            print("    >>> .venv\\Scripts\\python.exe ingestion/realtime_worker.py --once")
        elif schema_v5 == 0:
            print("\n    >>> WARNING: No Log5 events. Only old seed data exists.")
            print("    >>> The realtime worker has NOT run yet.")
            print("    >>> Run: .venv\\Scripts\\python.exe ingestion/realtime_worker.py --once")

        print()
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Multi-mode runner — Log6 (wired to analyze_multi_mode_v5)
# ---------------------------------------------------------------------------

async def run_multi_v5(origin: str, destination: str, mongo_uri: str) -> dict:
    """Call the Log5 evidence-enriched orchestrator (NOT the old Log4 one)."""
    from motor.motor_asyncio import AsyncIOMotorClient
    from core.orchestrator import analyze_multi_mode_v5   # <-- Log5 function
    client = AsyncIOMotorClient(mongo_uri)
    try:
        return await analyze_multi_mode_v5(
            origin=origin,
            destination=destination,
            mongo_collection=client["geo_risk"]["geo_events"],
            radius_km=50.0,
        )
    finally:
        client.close()


def _print_multi_result_v5(result: dict) -> None:
    """Print Log5 evidence-enriched multi-mode result with full debug output."""
    print(f"\n  {'-'*66}")
    print(f"  ORIGIN      : {result['origin'][:70]}")
    print(f"  DESTINATION : {result['destination'][:70]}")
    rec = result.get("recommended_mode", "?").upper()
    print(f"  RECOMMENDED : >> {rec} << (lowest risk)")
    print(f"  ANALYZED AT : {result.get('analyzed_at', '?')}")
    print()

    modes_order = ["air", "sea", "road"]
    mode_labels = {"air": "AIR ", "sea": "SEA ", "road": "ROAD"}

    for mode in modes_order:
        m = result["modes"].get(mode, {})
        icon   = _STATUS_ICON.get(m.get("status", "UNKNOWN"), "[??]")
        m_label = mode_labels.get(mode, mode.upper())
        safety = m.get("safety_score")
        safety_str = f"{safety:.3f}" if safety is not None else "N/A"
        risk   = m.get("risk_score")
        risk_str = f"{risk:.3f}" if risk is not None else "N/A"
        dist   = m.get("distance_km")
        dist_str = f"{dist:,.0f} km" if dist else "N/A"
        alerts = m.get("alerts", 0)

        print(f"  {m_label:5s}  {icon:6s} {m.get('status', 'UNKNOWN'):8s}  "
              f"risk={risk_str:>6}  safety={safety_str}  "
              f"alerts={alerts:3d}  dist={dist_str}")
        print(f"          {m.get('message', '')}")

        # Zone intersections (Log5)
        zones = m.get("zone_intersections", [])
        if zones:
            print(f"          ZONES CROSSED ({len(zones)}):")
            for z in zones[:5]:
                print(f"            >> {z['zone']:30s}  ({z['category']:12s})  "
                      f"dist={z['min_distance_km']:.0f} km")
                if z.get("description"):
                    print(f"               {z['description'][:65]}")

        # Evidence events with source links (Log5)
        events = m.get("events", [])
        if events:
            print(f"          TOP EVIDENCE ({len(events)} events):")
            for ev in events[:3]:
                label_str = ev.get("label", "?").upper()
                headline = ev.get("headline", "")[:60]
                print(f"            [{label_str:10s}] {headline}")
                print(f"              dist={ev.get('distance_km', 0):.1f} km  "
                      f"intensity={ev.get('intensity', 0):.3f}  "
                      f"confidence={ev.get('confidence', 0):.2f}")

                # Source URL
                url = ev.get("source_url", "")
                if url:
                    print(f"              URL: {url[:80]}")

                # Image URL
                img = ev.get("image_url")
                if img:
                    print(f"              IMG: {img[:80]}")

                # Zone
                zone = ev.get("zone")
                if zone:
                    print(f"              ZONE: {zone}")

                # Credibility
                cred = ev.get("credibility")
                if cred is not None:
                    print(f"              CREDIBILITY: {cred:.2f}")

                # Publisher
                pub = ev.get("publisher", "")
                if pub:
                    print(f"              PUBLISHER: {pub}")

        elif alerts == 0:
            print(f"          (no events near this route)")

        print()


async def run_multi_batch_v5(routes: list[tuple[str, str]], mongo_uri: str) -> None:
    """Run Log5 pipeline with full debug output."""
    print("\n" + "=" * 70)
    print("  GEO RISK ENGINE -- Log5 Evidence-Enriched Analysis (Log6 wiring)")
    print("=" * 70)

    # DB diagnostics first
    await _print_db_diagnostics(mongo_uri)

    for origin, destination in routes:
        print(f"\n  {'='*66}")
        print(f"  ANALYZING: {origin}  ->  {destination}")

        try:
            result = await run_multi_v5(origin, destination, mongo_uri)
            _print_multi_result_v5(result)
        except ValueError as exc:
            print(f"  GEOCODING ERROR: {exc}")
        except Exception as exc:
            logger.exception("Multi-mode pipeline error")
            print(f"  ERROR: {exc}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geo Risk Engine -- pipeline runner (Log6)")
    parser.add_argument("--uri",         default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--origin",      default=None)
    parser.add_argument("--destination", default=None)
    parser.add_argument("--single",      action="store_true", help="Run single-mode (Log3) instead of multi-mode")
    parser.add_argument("--legacy",      action="store_true", help="Use old Log4 orchestrator (analyze_multi_mode)")
    args = parser.parse_args()

    if args.origin and args.destination:
        single_routes = [(args.origin, args.destination)]
        multi_routes  = [(args.origin, args.destination)]
    else:
        single_routes = DEFAULT_ROUTES
        multi_routes  = MULTI_MODE_ROUTES

    if args.single:
        asyncio.run(run_single_batch(single_routes, args.uri))
    elif args.legacy:
        # Keep old Log4 runner available behind --legacy flag
        async def _run_legacy():
            from motor.motor_asyncio import AsyncIOMotorClient
            from core.orchestrator import analyze_multi_mode
            print("\n[LEGACY MODE] Using Log4 analyze_multi_mode()\n")
            for origin, destination in multi_routes:
                client = AsyncIOMotorClient(args.uri)
                try:
                    result = await analyze_multi_mode(
                        origin=origin,
                        destination=destination,
                        mongo_collection=client["geo_risk"]["geo_events"],
                    )
                    print(json.dumps(result, indent=2, default=str))
                finally:
                    client.close()
        asyncio.run(_run_legacy())
    else:
        # DEFAULT: Log5 orchestrator with evidence output
        asyncio.run(run_multi_batch_v5(multi_routes, args.uri))
