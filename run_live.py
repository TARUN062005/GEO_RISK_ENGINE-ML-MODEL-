"""
run_live.py
-----------
ONE-COMMAND Real-Time Geo-Intelligence Platform (Log7)

Usage:
    python run_live.py --origin "Mumbai, India" --destination "Dubai, UAE"

This single script:
  1. Purges any old seed data from MongoDB
  2. Runs real-time ingestion (GDELT + RSS + News APIs)
  3. Waits until sufficient real events are stored
  4. Runs zone-aware multi-mode risk analysis (air/sea/road)
  5. Prints enriched output with evidence, zones, and source URLs

NO separate terminal. NO manual ingestion step. ONE command.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import sys
import time

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_live")

# Suppress noisy loggers during live run
for noisy in ["httpx", "httpcore", "urllib3", "geopy", "shapely"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Status icons
# ---------------------------------------------------------------------------
_STATUS_ICON = {
    "LOW":      "[OK]",
    "MEDIUM":   "[!!]",
    "HIGH":     "[XX]",
    "CRITICAL": "[XX]",
    "UNKNOWN":  "[??]",
    "ERROR":    "[ER]",
}


# ---------------------------------------------------------------------------
# 1. Purge seed data
# ---------------------------------------------------------------------------

async def purge_seed_data(collection) -> int:
    """Delete ALL synthetic/seed events from MongoDB. Returns count deleted."""
    result = await collection.delete_many({"source": "seed"})
    deleted = result.deleted_count
    if deleted > 0:
        logger.info("Purged %d seed events from DB.", deleted)
    return deleted


# ---------------------------------------------------------------------------
# 2. Inline ingestion (runs in same process)
# ---------------------------------------------------------------------------

async def run_ingestion_cycle(collection) -> dict:
    """Run one ingestion cycle, return stats dict."""
    from ingestion.realtime_worker import ingest_cycle
    return await ingest_cycle(collection)


async def ensure_fresh_data(
    collection,
    min_events: int = 10,
    max_wait_seconds: int = 150,
    max_freshness_minutes: int = 10,
    print_progress: bool = True,
) -> int:
    """
    Ensure DB has enough FRESH real (non-seed) events.

    Log8 fix: checks TIMESTAMP freshness, not just count.
    Even if DB has 1000 events, if the newest is older than
    max_freshness_minutes, ingestion runs anyway.
    """
    from ingestion.realtime_worker import _ensure_indexes

    await _ensure_indexes(collection)

    # Check both count AND freshness
    real_count = await collection.count_documents({"source": {"$ne": "seed"}})
    needs_ingest = real_count < min_events

    if not needs_ingest and real_count > 0:
        # Check freshness of latest event
        latest = await collection.find_one(
            {"source": {"$ne": "seed"}},
            sort=[("ingested_at", -1)],
            projection={"ingested_at": 1},
        )
        if latest and latest.get("ingested_at"):
            from datetime import datetime, timezone, timedelta
            age = datetime.now(timezone.utc) - latest["ingested_at"].replace(tzinfo=timezone.utc)
            age_minutes = age.total_seconds() / 60.0
            if age_minutes > max_freshness_minutes:
                needs_ingest = True
                if print_progress:
                    print(f"  DB has {real_count} events but latest is {age_minutes:.0f} min old "
                          f"(threshold: {max_freshness_minutes} min). Refreshing...")
            else:
                if print_progress:
                    print(f"  DB has {real_count} real events, latest is {age_minutes:.0f} min old. Fresh enough.")
                return real_count
        else:
            needs_ingest = True

    if not needs_ingest:
        if print_progress:
            print(f"  DB has {real_count} fresh events. Skipping ingestion.")
        return real_count

    if print_progress and real_count < min_events:
        print(f"  DB has {real_count} real events (need {min_events}). Starting ingestion...")

    start = time.time()
    cycle = 0

    while time.time() - start < max_wait_seconds:
        cycle += 1
        if print_progress:
            elapsed = int(time.time() - start)
            print(f"  [Cycle {cycle}] Fetching live data... ({elapsed}s elapsed)")

        try:
            stats = await run_ingestion_cycle(collection)
            if print_progress:
                cluster_info = ""
                if stats.get("clustered_from") and stats.get("clustered_to"):
                    cluster_info = f" clustered={stats['clustered_from']}→{stats['clustered_to']}"
                print(f"  [Cycle {cycle}] fetched={stats['fetched']} "
                      f"enriched={stats['enriched']}{cluster_info} written={stats['written']} "
                      f"skipped={stats['skipped']} errors={stats.get('errors', 0)}")
        except Exception as exc:
            logger.warning("Ingestion cycle %d failed: %s", cycle, exc)
            if print_progress:
                print(f"  [Cycle {cycle}] Error: {exc}")

        real_count = await collection.count_documents({"source": {"$ne": "seed"}})
        if print_progress:
            print(f"  [Cycle {cycle}] Total real events in DB: {real_count}")

        if real_count >= min_events:
            break

        # Short wait before next cycle
        if time.time() - start < max_wait_seconds:
            await asyncio.sleep(5)

    return real_count


# ---------------------------------------------------------------------------
# 3. DB Diagnostics
# ---------------------------------------------------------------------------

async def print_db_diagnostics(collection) -> None:
    """Print DB state summary."""
    total = await collection.count_documents({})
    seed_count = await collection.count_documents({"source": "seed"})
    real_count = total - seed_count
    with_url = await collection.count_documents({"source_url": {"$nin": [None, ""]}})
    with_zones = await collection.count_documents({"zones": {"$exists": True, "$ne": []}})

    # Source breakdown
    source_pipeline = [
        {"$match": {"source": {"$ne": "seed"}}},
        {"$group": {"_id": "$source", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    sources = {}
    async for doc in collection.aggregate(source_pipeline):
        sources[doc["_id"]] = doc["count"]

    # Latest event
    latest = await collection.find_one(
        {"source": {"$ne": "seed"}},
        sort=[("ingested_at", -1)],
        projection={"ingested_at": 1, "source": 1, "raw_text": 1},
    )

    print(f"\n  {'─' * 66}")
    print(f"  DB STATUS")
    print(f"  {'─' * 66}")
    print(f"    Real events    : {real_count}")
    print(f"    With source URL: {with_url}")
    print(f"    With zones     : {with_zones}")
    print(f"    Sources        : {sources}")
    if latest:
        print(f"    Latest at      : {latest.get('ingested_at', '?')}")
        print(f"    Latest source  : {latest.get('source', '?')}")
        txt = latest.get("raw_text", "")[:80]
        print(f"    Latest text    : {txt}...")
    print()


# ---------------------------------------------------------------------------
# 4. Analysis output
# ---------------------------------------------------------------------------

def print_analysis_result(result: dict) -> None:
    """Print the full evidence-enriched multi-mode result."""
    print(f"\n  {'═' * 66}")
    print(f"  ORIGIN      : {result['origin'][:70]}")
    print(f"  DESTINATION : {result['destination'][:70]}")
    rec = result.get("recommended_mode", "?").upper()
    print(f"  RECOMMENDED : >> {rec} << (lowest risk)")
    print(f"  ANALYZED AT : {result.get('analyzed_at', '?')}")
    print(f"  {'═' * 66}")

    for mode in ["air", "sea", "road"]:
        m = result["modes"].get(mode, {})
        icon = _STATUS_ICON.get(m.get("status", "UNKNOWN"), "[??]")
        status = m.get("status", "UNKNOWN")
        risk = m.get("risk_score")
        safety = m.get("safety_score")
        zone_risk = m.get("zone_risk", 0)
        event_risk = m.get("event_risk", 0)
        alerts = m.get("alerts", 0)
        dist = m.get("distance_km")

        risk_str = f"{risk:.3f}" if risk is not None else "N/A"
        safety_str = f"{safety:.3f}" if safety is not None else "N/A"
        dist_str = f"{dist:,.0f} km" if dist else "N/A"

        print(f"\n  {mode.upper():5s}  {icon} {status:8s}  "
              f"risk={risk_str:>6}  safety={safety_str}  "
              f"alerts={alerts:3d}  dist={dist_str}")
        print(f"        {m.get('message', '')}")
        print(f"        zone_risk={zone_risk:.2f}  event_risk={event_risk:.2f}")

        # Zone intersections
        zones = m.get("zone_intersections", [])
        if zones:
            print(f"        ZONES CROSSED ({len(zones)}):")
            for z in zones[:5]:
                print(f"          >> {z['zone']:30s}  ({z['category']:12s})  "
                      f"dist={z['min_distance_km']:.0f} km")
                desc = z.get("description", "")
                if desc:
                    print(f"             {desc[:70]}")

        # Evidence events
        events = m.get("events", [])
        if events:
            print(f"        EVIDENCE ({len(events)} events):")
            for ev in events[:3]:
                label = ev.get("label", "?").upper()
                headline = ev.get("headline", "")[:60]
                print(f"          [{label:10s}] {headline}")
                print(f"            dist={ev.get('distance_km', 0):.1f} km  "
                      f"intensity={ev.get('intensity', 0):.3f}  "
                      f"confidence={ev.get('confidence', 0):.2f}")

                url = ev.get("source_url", "")
                if url:
                    print(f"            URL: {url[:80]}")

                img = ev.get("image_url")
                if img:
                    print(f"            IMG: {img[:80]}")

                zone = ev.get("zone")
                if zone:
                    print(f"            ZONE: {zone}")

                cred = ev.get("credibility")
                corr_count = ev.get("corroboration_count", 1)
                corr_score = ev.get("corroboration_score", 0.0)
                if corr_count:
                    print(f"            CORROBORATION: sources={corr_count} score={corr_score:.2f}")
                pub = ev.get("publisher", "")
                if cred is not None:
                    print(f"            PUBLISHER: {pub}  CREDIBILITY: {cred:.2f}")
                elif pub:
                    print(f"            PUBLISHER: {pub}")

        elif alerts == 0 and not zones:
            print(f"        (no events or zones near this route)")


# ---------------------------------------------------------------------------
# 5. Main orchestration
# ---------------------------------------------------------------------------

async def run_live(
    origin: str,
    destination: str,
    mongo_uri: str = "mongodb://localhost:27017",
    min_events: int = 10,
    max_wait: int = 150,
    max_freshness_minutes: int = 10,
) -> dict:
    """
    Full single-command execution pipeline:
      1. Connect to MongoDB
      2. Purge seed data
      3. Run ingestion until DB has enough FRESH events
      4. Run zone-aware multi-mode analysis
      5. Print results
    """
    from motor.motor_asyncio import AsyncIOMotorClient
    from core.orchestrator import analyze_multi_mode_v5

    client = AsyncIOMotorClient(mongo_uri)
    collection = client["geo_risk"]["geo_events"]

    print("\n" + "=" * 70)
    print("  GEO-INTELLIGENCE ENGINE — Real-Time Analysis (Log8)")
    print("=" * 70)
    print(f"  Route: {origin}  →  {destination}")
    print()

    # Step 1: Purge seed data
    print("  [1/4] Purging synthetic data...")
    purged = await purge_seed_data(collection)
    if purged > 0:
        print(f"         Removed {purged} seed events.")
    else:
        print(f"         No seed data found.")

    # Step 2: Ensure fresh real data (Log8: freshness check)
    print("\n  [2/4] Ensuring real-time data (freshness threshold: "
          f"{max_freshness_minutes} min)...")
    real_count = await ensure_fresh_data(
        collection,
        min_events=min_events,
        max_wait_seconds=max_wait,
        max_freshness_minutes=max_freshness_minutes,
    )
    print(f"         {real_count} real events available.")

    # Step 3: DB diagnostics
    print("\n  [3/4] Database status:")
    await print_db_diagnostics(collection)

    # Step 4: Run analysis
    print("  [4/4] Running zone-aware multi-mode analysis...")
    try:
        result = await analyze_multi_mode_v5(
            origin=origin,
            destination=destination,
            mongo_collection=collection,
            radius_km=50.0,
        )
        print_analysis_result(result)
    except ValueError as exc:
        print(f"\n  GEOCODING ERROR: {exc}")
        result = {"error": str(exc)}
    except Exception as exc:
        logger.exception("Analysis failed")
        print(f"\n  ERROR: {exc}")
        result = {"error": str(exc)}

    print("\n" + "=" * 70 + "\n")
    client.close()
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geo-Intelligence Engine — Real-Time Single-Command Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_live.py --origin "Mumbai, India" --destination "Dubai, UAE"
  python run_live.py --origin "Singapore" --destination "Rotterdam, Netherlands"
  python run_live.py --origin "New York, USA" --destination "London, UK"
        """,
    )
    parser.add_argument("--origin", required=True, help="Origin location (free text)")
    parser.add_argument("--destination", required=True, help="Destination location (free text)")
    parser.add_argument("--uri", default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
                        help="MongoDB URI")
    parser.add_argument("--min-events", type=int, default=10,
                        help="Minimum real events before analysis (default: 10)")
    parser.add_argument("--max-wait", type=int, default=150,
                        help="Max seconds to wait for ingestion (default: 150)")
    parser.add_argument("--freshness", type=int, default=10,
                        help="Max age of latest event in minutes before re-ingesting (default: 10)")
    args = parser.parse_args()

    asyncio.run(run_live(
        origin=args.origin,
        destination=args.destination,
        mongo_uri=args.uri,
        min_events=args.min_events,
        max_wait=args.max_wait,
        max_freshness_minutes=args.freshness,
    ))
