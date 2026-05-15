"""
storage/schema.py
-----------------
Enriched Event Schema (Log2)

NEW in Log2: Events stored in MongoDB now carry ALL pre-computed
ML fields. The API never re-runs classification, NER, or intensity
scoring — it reads what ingestion already wrote.

MongoDB collection: geo_events
Index: 2dsphere on location.coordinates
       TTL on published_at (configurable)
       Compound: (label, intensity_score) for fast filtering
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class GeoPoint(BaseModel):
    """GeoJSON Point — enables MongoDB 2dsphere queries."""
    type: str = "Point"
    coordinates: list[float]   # [longitude, latitude]


class MLAnnotation(BaseModel):
    """
    All ML-computed fields written by the ingestion worker.
    The API layer reads these; it NEVER recomputes them.
    """
    # A. Event Classification (ml/inference/classifier.py)
    label: str                          # conflict | protest | sanction | disaster | terrorism | safe
    label_confidence: float             # [0, 1]
    label_scores: dict[str, float]      # full per-label score dict
    classification_method: str          # "zero_shot" | "heuristic"

    # B. NER (ml/inference/ner.py)
    location_names: list[str]           # ["Kyiv", "Ukraine", ...]
    ner_method: str                     # "spacy" | "hf_token" | "regex"

    # C. Intensity Score (ml/inference/scoring.py)
    intensity_score: float              # [0, 1]
    intensity_method: str               # "logistic" | "rule_based"
    intensity_explanation: dict[str, float]   # feature → contribution


class SourceVerificationDoc(BaseModel):
    """
    Source verification metadata (Log5).
    Stored alongside events for evidence provenance.
    """
    source_url: str = ""
    publisher: str = ""
    credibility_score: float = 0.0       # [0, 1]
    credibility_tier: str = "unknown"    # "tier1" | "tier2" | "tier3" | "unknown"
    image_url: Optional[str] = None
    domain: str = ""
    retrieved_at: Optional[datetime] = None


class EnrichedEvent(BaseModel):
    """
    Complete stored event document (Log2 base, extended Log5).
    Written by ingestion worker; read by API and aggregation.
    """
    model_config = ConfigDict(extra="ignore")

    event_id: str
    source: str                         # "gdelt" | "rss" | "newsapi" | "gnews" | "manual"
    raw_text: str
    published_at: datetime              # UTC

    # Geo (resolved from NER or feed metadata)
    location: GeoPoint                  # GeoJSON for 2dsphere index
    country_code: Optional[str] = None  # ISO 3166-1 alpha-2

    # ML annotation block (all pre-computed at ingestion)
    ml: MLAnnotation

    # Log5: Source verification + evidence metadata
    source_url: str = ""                 # verified article URL
    publisher: str = ""                  # publisher name
    image_url: Optional[str] = None      # article image URL
    verification: Optional[SourceVerificationDoc] = None   # credibility metadata
    zones: list[str] = Field(default_factory=list)          # matched geo zone names

    # Log12: Canonical incident grouping metadata.
    canonical_event_id: Optional[str] = None
    representative_headline: Optional[str] = None
    source_urls: list[str] = Field(default_factory=list)
    publishers: list[str] = Field(default_factory=list)
    corroboration_count: int = 1
    corroboration_score: float = 0.0
    combined_credibility: float = 0.0
    latest_source_at: Optional[datetime] = None

    # Ingestion bookkeeping
    ingested_at: datetime
    schema_version: str = "2"          # "2" for legacy, "5" for Log5 events


# ---------------------------------------------------------------------------
# MongoDB document converter
# ---------------------------------------------------------------------------

def to_mongo_doc(event: EnrichedEvent) -> dict[str, Any]:
    """Serialize EnrichedEvent to a MongoDB-ready dict."""
    doc = event.model_dump()
    doc["_id"] = doc.pop("event_id")
    return doc


def from_mongo_doc(doc: dict[str, Any]) -> EnrichedEvent:
    """Deserialize MongoDB document back to EnrichedEvent."""
    doc = dict(doc)
    doc["event_id"] = str(doc.pop("_id"))
    return EnrichedEvent(**doc)
