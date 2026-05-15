"""
ingestion/clustering.py
-----------------------
Canonical Incident Clustering (Log12)

Problem:
  Reuters, BBC, NPR, AP all report "Houthi attack in Red Sea"
  → system stores 4 separate events for ONE real-world incident.

Solution:
  Lightweight incident clustering using:
  1. Headline similarity (SequenceMatcher — zero model cost)
  2. Location proximity (haversine < 100km)
  3. Time proximity (published within 24h)
  4. ML label agreement

Output:
  Merged canonical events with:
  - source_count (corroboration)
  - all source URLs
  - all publishers
  - best headline (longest)
  - highest credibility source
  - averaged intensity

This runs AFTER enrichment, BEFORE MongoDB insert.
No embeddings, no vector DB, no extra models — pure algorithmic.
"""

from __future__ import annotations

import hashlib
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HEADLINE_SIM_THRESHOLD = 0.45   # SequenceMatcher ratio — lowered for partial matches
LOCATION_PROXIMITY_KM = 100.0   # Events within 100km are same-location candidates
TIME_WINDOW_HOURS = 24.0        # Events within 24h are same-time candidates

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "with", "after",
    "over", "new", "says", "said", "amid", "into", "near",
}


# ---------------------------------------------------------------------------
# Haversine (local copy — avoids circular import)
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------

def _headline_similarity(a: str, b: str) -> float:
    """Fast headline similarity using SequenceMatcher (no ML required)."""
    if not a or not b:
        return 0.0
    # Normalize: lowercase, strip punctuation
    na = a.lower().strip()
    nb = b.lower().strip()
    return SequenceMatcher(None, na[:200], nb[:200]).ratio()


def _token_cosine(a: str, b: str) -> float:
    """Cheap semantic-ish similarity over meaningful headline tokens."""
    import re
    toks_a = [t for t in re.findall(r"[a-z0-9]+", a.lower()) if t not in _STOPWORDS and len(t) > 2]
    toks_b = [t for t in re.findall(r"[a-z0-9]+", b.lower()) if t not in _STOPWORDS and len(t) > 2]
    if not toks_a or not toks_b:
        return 0.0
    freq_a: dict[str, int] = {}
    freq_b: dict[str, int] = {}
    for tok in toks_a:
        freq_a[tok] = freq_a.get(tok, 0) + 1
    for tok in toks_b:
        freq_b[tok] = freq_b.get(tok, 0) + 1
    common = set(freq_a) & set(freq_b)
    dot = sum(freq_a[t] * freq_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in freq_a.values()))
    norm_b = math.sqrt(sum(v * v for v in freq_b.values()))
    return dot / max(norm_a * norm_b, 1e-9)


def _location_close(doc_a: dict, doc_b: dict) -> bool:
    """Check if two events are geographically close."""
    try:
        loc_a = doc_a.get("location", {}).get("coordinates", [])
        loc_b = doc_b.get("location", {}).get("coordinates", [])
        if len(loc_a) < 2 or len(loc_b) < 2:
            return True  # If no coords, assume possible match
        # coordinates = [lon, lat]
        dist = _haversine_km(loc_a[1], loc_a[0], loc_b[1], loc_b[0])
        return dist <= LOCATION_PROXIMITY_KM
    except Exception:
        return True  # Conservative: assume close if coords broken


def _time_close(doc_a: dict, doc_b: dict) -> bool:
    """Check if two events are within the time window."""
    try:
        t_a = doc_a.get("published_at")
        t_b = doc_b.get("published_at")
        if t_a is None or t_b is None:
            return True  # Assume close if no timestamps
        if isinstance(t_a, str):
            t_a = datetime.fromisoformat(t_a)
        if isinstance(t_b, str):
            t_b = datetime.fromisoformat(t_b)
        delta = abs((t_a - t_b).total_seconds()) / 3600.0
        return delta <= TIME_WINDOW_HOURS
    except Exception:
        return True


def _same_label(doc_a: dict, doc_b: dict) -> bool:
    """Check if two events have the same ML classification."""
    ml_a = doc_a.get("ml", {})
    ml_b = doc_b.get("ml", {})
    label_a = ml_a.get("label", "") if isinstance(ml_a, dict) else getattr(ml_a, "label", "")
    label_b = ml_b.get("label", "") if isinstance(ml_b, dict) else getattr(ml_b, "label", "")
    if not label_a or not label_b:
        return True  # Assume match if labels missing
    return label_a == label_b


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _are_same_incident(doc_a: dict, doc_b: dict) -> bool:
    """
    Determine if two event documents describe the same real-world incident.

    Requires:
    - Headline similarity >= threshold  OR  same ML label + location
    - Geographic proximity <= 100km
    - Time proximity <= 24h
    """
    text_a = doc_a.get("raw_text", "")
    text_b = doc_b.get("raw_text", "")

    sim = max(_headline_similarity(text_a, text_b), _token_cosine(text_a, text_b))
    headline_match = sim >= HEADLINE_SIM_THRESHOLD

    # Also check: same label + close location (catches differently-worded reports)
    label_match = _same_label(doc_a, doc_b) and _location_close(doc_a, doc_b)

    # Must pass one of: headline similarity OR (label + location)
    if not (headline_match or (label_match and sim >= 0.25)):
        return False

    # Time proximity is mandatory
    if not _time_close(doc_a, doc_b):
        return False

    # Location proximity is mandatory
    if not _location_close(doc_a, doc_b):
        return False

    return True


def cluster_incidents(docs: list[dict]) -> list[dict]:
    """
    Cluster enriched event documents into canonical incidents.

    Algorithm:
      Simple greedy single-pass clustering (O(n²) but n < 200).
      For each document, check if it matches any existing cluster.
      If yes, merge into that cluster. If no, start new cluster.

    Returns:
      list of canonical docs — one per incident, with merged metadata.
    """
    if len(docs) <= 1:
        return docs

    clusters: list[list[dict]] = []

    for doc in docs:
        merged = False
        for cluster in clusters:
            # Compare against cluster representative (first doc)
            if _are_same_incident(cluster[0], doc):
                cluster.append(doc)
                merged = True
                break
        if not merged:
            clusters.append([doc])

    # Merge each cluster into one canonical document
    canonical_docs: list[dict] = []
    total_merged = 0

    for cluster in clusters:
        canonical = _merge_cluster(cluster)
        canonical_docs.append(canonical)
        if len(cluster) > 1:
            total_merged += len(cluster) - 1

    if total_merged > 0:
        logger.info(
            "Log12 clustering: %d events → %d incidents (merged %d duplicates)",
            len(docs), len(canonical_docs), total_merged,
        )
    else:
        logger.info("Log12 clustering: %d events, 0 duplicates found.", len(docs))

    return canonical_docs


def _merge_cluster(cluster: list[dict]) -> dict:
    """
    Merge a cluster of related event documents into ONE canonical document.

    Strategy:
    - headline: longest text (most informative)
    - location: from highest-credibility source
    - sources: collect ALL unique publishers + URLs
    - intensity: average across cluster
    - credibility: max across cluster
    - published_at: most recent
    """
    if len(cluster) == 1:
        return _with_canonical_metadata(dict(cluster[0]), cluster)

    # Sort by text length (longest = most informative)
    by_length = sorted(cluster, key=lambda d: len(d.get("raw_text", "")), reverse=True)
    canonical = dict(by_length[0])  # Use longest as base

    # Collect all sources
    all_publishers: list[str] = []
    all_urls: list[str] = []
    all_images: list[str] = []
    max_credibility = 0.0
    intensity_sum = 0.0
    intensity_count = 0

    for doc in cluster:
        pub = doc.get("publisher", "")
        url = doc.get("source_url", "")
        img = doc.get("image_url", "")
        if pub and pub not in all_publishers:
            all_publishers.append(pub)
        if url and url not in all_urls:
            all_urls.append(url)
        if img and img not in all_images:
            all_images.append(img)

        # Track max credibility
        verif = doc.get("verification", {})
        if isinstance(verif, dict):
            cred = verif.get("credibility_score", 0.0)
            if cred > max_credibility:
                max_credibility = cred

        # Accumulate intensity
        ml = doc.get("ml", {})
        if isinstance(ml, dict):
            score = ml.get("intensity_score", 0.0)
        else:
            score = getattr(ml, "intensity_score", 0.0)
        if score > 0:
            intensity_sum += score
            intensity_count += 1

    # Use most recent timestamp
    try:
        latest = max(
            cluster,
            key=lambda d: d.get("published_at", datetime.min) if isinstance(
                d.get("published_at"), datetime) else datetime.min
        )
        canonical["published_at"] = latest.get("published_at", canonical.get("published_at"))
    except Exception:
        pass

    # Average intensity
    if intensity_count > 0:
        avg_intensity = intensity_sum / intensity_count
        ml = canonical.get("ml", {})
        if isinstance(ml, dict):
            ml["intensity_score"] = round(avg_intensity, 4)
        canonical["ml"] = ml

    # Best image (from first available)
    if all_images and not canonical.get("image_url"):
        canonical["image_url"] = all_images[0]

    return _with_canonical_metadata(canonical, cluster)


def _with_canonical_metadata(doc: dict, cluster: list[dict]) -> dict:
    """Attach public canonical incident fields consumed by storage/API."""
    publishers: list[str] = []
    urls: list[str] = []
    credibility_values: list[float] = []
    timestamps: list[datetime] = []

    for item in cluster:
        pub = item.get("publisher", "")
        url = item.get("source_url", "")
        if pub and pub not in publishers:
            publishers.append(pub)
        if url and url not in urls:
            urls.append(url)

        verif = item.get("verification", {})
        if isinstance(verif, dict):
            cred = float(verif.get("credibility_score", 0.0) or 0.0)
            if cred > 0:
                credibility_values.append(cred)

        ts = item.get("published_at")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = None
        if isinstance(ts, datetime):
            timestamps.append(ts)

    canonical_key_parts = sorted(urls) or sorted(
        item.get("raw_text", "")[:120].lower()
        for item in cluster
        if item.get("raw_text")
    )
    digest = hashlib.sha1("|".join(canonical_key_parts).encode("utf-8", errors="ignore")).hexdigest()[:16]
    canonical_id = f"incident-{digest}"

    source_count = max(1, len(publishers) or len(urls) or len(cluster))
    fallback_credibility = 0.0
    verif = doc.get("verification", {})
    if isinstance(verif, dict):
        fallback_credibility = float(verif.get("credibility_score", 0.0) or 0.0)
    avg_credibility = (
        sum(credibility_values) / len(credibility_values)
        if credibility_values else fallback_credibility
    )
    corroboration_score = min(1.0, (source_count / 4.0) * max(avg_credibility, 0.25))

    doc["_id"] = canonical_id
    doc["canonical_event_id"] = canonical_id
    doc["representative_headline"] = doc.get("raw_text", "")[:180]
    doc["source_urls"] = urls
    doc["publishers"] = publishers
    doc["corroboration_count"] = source_count
    doc["corroboration_score"] = round(corroboration_score, 4)
    doc["combined_credibility"] = round(avg_credibility, 4)
    doc["latest_source_at"] = max(timestamps) if timestamps else doc.get("published_at")
    doc["_cluster_size"] = len(cluster)
    doc["_cluster_sources"] = publishers
    doc["_cluster_urls"] = urls
    return doc
