"""
scripts/seed_db.py
------------------
Database Seeder (Log3)

Seeds MongoDB with realistic synthetic events for development/testing
when GDELT ingestion hasn't been run yet.

Events span multiple continents to exercise the dynamic routing pipeline.
All events follow the EnrichedEvent v2 schema (Log2).

Usage:
    python scripts/seed_db.py
    python scripts/seed_db.py --uri mongodb://localhost:27017 --count 50
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta, timezone

from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed data — geographically spread events
# ---------------------------------------------------------------------------

_SEED_EVENTS = [
    # Middle East
    {"text": "Armed clashes reported near Baghdad following military escalation.",      "lat": 33.34, "lon": 44.40, "label": "conflict",  "intensity": 0.88, "cc": "IQ"},
    {"text": "Suicide bombing in Kabul kills dozens of civilians.",                    "lat": 34.52, "lon": 69.18, "label": "terrorism", "intensity": 0.95, "cc": "AF"},
    {"text": "Israeli airstrikes target Gaza infrastructure overnight.",               "lat": 31.35, "lon": 34.45, "label": "conflict",  "intensity": 0.90, "cc": "PS"},
    {"text": "Protests erupt in Tehran over economic sanctions.",                      "lat": 35.69, "lon": 51.39, "label": "protest",   "intensity": 0.50, "cc": "IR"},
    {"text": "US imposes new sanctions on Iranian oil exports.",                       "lat": 35.69, "lon": 51.39, "label": "sanction",  "intensity": 0.60, "cc": "IR"},
    # Eastern Europe
    {"text": "Russia launches missile strikes on Kyiv during overnight hours.",        "lat": 50.45, "lon": 30.52, "label": "conflict",  "intensity": 0.92, "cc": "UA"},
    {"text": "Shelling reported near Kharkiv as fighting intensifies in eastern Ukraine.", "lat": 49.99, "lon": 36.23, "label": "conflict", "intensity": 0.87, "cc": "UA"},
    {"text": "Belarus opposition protests suppressed by authorities.",                 "lat": 53.90, "lon": 27.57, "label": "protest",   "intensity": 0.45, "cc": "BY"},
    # Africa
    {"text": "Coup attempt reported in Sudan as military factions clash in Khartoum.", "lat": 15.55, "lon": 32.53, "label": "conflict",  "intensity": 0.82, "cc": "SD"},
    {"text": "Al-Shabaab attack on Mogadishu hotel kills 15.",                        "lat":  2.05, "lon": 45.34, "label": "terrorism", "intensity": 0.89, "cc": "SO"},
    {"text": "Widespread flooding in Nigeria displaces thousands.",                    "lat":  9.05, "lon":  7.49, "label": "disaster",  "intensity": 0.65, "cc": "NG"},
    {"text": "Mali government declares state of emergency after rebel advances.",      "lat": 12.65, "lon": -8.00, "label": "conflict",  "intensity": 0.78, "cc": "ML"},
    # Asia
    {"text": "North Korea launches ballistic missile into Sea of Japan.",              "lat": 39.02, "lon": 125.75, "label": "conflict",  "intensity": 0.80, "cc": "KP"},
    {"text": "Border skirmishes reported between India and China in Ladakh.",         "lat": 34.17, "lon":  77.58, "label": "conflict",  "intensity": 0.70, "cc": "IN"},
    {"text": "Myanmar military conducts airstrikes on civilian areas.",               "lat": 21.97, "lon":  96.08, "label": "conflict",  "intensity": 0.85, "cc": "MM"},
    {"text": "Protests in Hong Kong against new security legislation.",               "lat": 22.32, "lon": 114.17, "label": "protest",   "intensity": 0.42, "cc": "HK"},
    # South America
    {"text": "Venezuela opposition leader arrested amid political tensions.",          "lat": 10.48, "lon": -66.88, "label": "protest",   "intensity": 0.48, "cc": "VE"},
    {"text": "Colombia cartel violence kills 20 in border region.",                   "lat":  4.71, "lon": -74.07, "label": "conflict",  "intensity": 0.72, "cc": "CO"},
    # Stable regions (safe events)
    {"text": "EU-Canada trade summit reaches agreement on tariff reductions.",        "lat": 50.85, "lon":   4.35, "label": "safe",      "intensity": 0.05, "cc": "BE"},
    {"text": "G7 nations pledge cooperation on climate and energy security.",         "lat": 51.50, "lon":  -0.12, "label": "safe",      "intensity": 0.05, "cc": "GB"},
    {"text": "Japan and South Korea strengthen diplomatic ties.",                     "lat": 35.68, "lon": 139.69, "label": "safe",      "intensity": 0.08, "cc": "JP"},
    # Sea routes
    {"text": "Houthi militants attack commercial shipping in Red Sea.",               "lat": 14.00, "lon":  43.00, "label": "terrorism", "intensity": 0.88, "cc": "YE"},
    {"text": "Piracy incident reported near Gulf of Aden.",                           "lat": 11.00, "lon":  49.00, "label": "terrorism", "intensity": 0.75, "cc": "SO"},
    {"text": "Tensions in South China Sea over disputed shipping lanes.",             "lat": 12.00, "lon": 114.00, "label": "conflict",  "intensity": 0.68, "cc": "CN"},
]


async def seed(mongo_uri: str, db: str, count: int) -> None:
    client = AsyncIOMotorClient(mongo_uri)
    collection = client[db]["geo_events"]

    # Ensure 2dsphere index
    await collection.create_index([("location", "2dsphere")])
    await collection.create_index([("published_at", 1)], expireAfterSeconds=2592000)
    await collection.create_index([("ml.label", 1), ("ml.intensity_score", -1)])
    logger.info("Indexes ensured.")

    base = _SEED_EVENTS * (count // len(_SEED_EVENTS) + 1)
    inserted = 0

    for i, seed_ev in enumerate(base[:count]):
        age_days = random.uniform(0, 6)   # all within last week
        published = datetime.now(timezone.utc) - timedelta(days=age_days)

        # Jitter coordinates slightly so events aren't stacked
        lat = seed_ev["lat"] + random.uniform(-0.5, 0.5)
        lon = seed_ev["lon"] + random.uniform(-0.5, 0.5)

        doc = {
            "_id":            str(uuid.uuid4()),
            "source":         "seed",
            "raw_text":       seed_ev["text"],
            "published_at":   published,
            "location":       {"type": "Point", "coordinates": [lon, lat]},
            "country_code":   seed_ev["cc"],
            "ml": {
                "label":                seed_ev["label"],
                "label_confidence":     round(random.uniform(0.70, 0.98), 4),
                "label_scores":         {seed_ev["label"]: 0.85},
                "classification_method": "seed",
                "location_names":       [seed_ev["cc"]],
                "ner_method":           "seed",
                "intensity_score":      round(seed_ev["intensity"] + random.uniform(-0.05, 0.05), 4),
                "intensity_method":     "seed",
                "intensity_explanation": {},
            },
            "ingested_at":    datetime.now(timezone.utc),
            "schema_version": "2",
        }

        await collection.insert_one(doc)
        inserted += 1

    logger.info("Seeded %d events into %s.geo_events", inserted, db)
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri",   default="mongodb://localhost:27017")
    parser.add_argument("--db",    default="geo_risk")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(seed(args.uri, args.db, args.count))
