"""
core/risk/features.py
---------------------
Feature Engineering Module (Log1)

Computes derived risk features from raw ML outputs and geo context.

Features produced:
  - recency_weight  : exponential time-decay for event freshness
  - proximity_weight: proximity to route buffer (inverse distance)
  - normalized_score: final normalized composite risk feature

All functions are PURE (no I/O, no side effects) for testability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Recency decay half-life in days (score halves every N days)
# Log10: tightened from 7d → 1d for real-time intelligence
RECENCY_HALF_LIFE_DAYS: float = 1.0

# Proximity scale: distance (km) at which weight reaches 0.5
PROXIMITY_HALF_DISTANCE_KM: float = 50.0

# Maximum proximity weight cap
PROXIMITY_MAX: float = 1.0

# Minimum recency weight before event is considered stale
RECENCY_FLOOR: float = 0.01


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class EventFeatures:
    """All engineered features for a single news event."""
    event_id: str
    intensity_score: float        # raw [0,1] from ml.scoring
    recency_weight: float         # [0,1] time-decay
    proximity_weight: float       # [0,1] distance-decay
    event_label: str              # e.g. "conflict"
    classifier_confidence: float  # [0,1]

    # Derived composite
    composite_risk: float = 0.0   # populated by compute_composite()


# ---------------------------------------------------------------------------
# Recency decay
# ---------------------------------------------------------------------------

def compute_recency_weight(
    event_time: datetime,
    reference_time: datetime | None = None,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> float:
    """
    Exponential decay: weight = 2^(-age_days / half_life_days)

    Returns 1.0 for brand-new events, approaching 0 for old events.
    Result is clamped to [RECENCY_FLOOR, 1.0].

    Args:
        event_time: UTC timestamp of the event.
        reference_time: Current time (defaults to UTC now).
        half_life_days: Days until weight halves (default 7).
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    # Ensure timezone-aware comparison
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    age_seconds = (reference_time - event_time).total_seconds()
    age_days = max(age_seconds / 86400.0, 0.0)

    weight = math.pow(2.0, -age_days / half_life_days)
    return max(weight, RECENCY_FLOOR)


# ---------------------------------------------------------------------------
# Proximity weighting
# ---------------------------------------------------------------------------

def compute_proximity_weight(
    distance_km: float,
    half_distance_km: float = PROXIMITY_HALF_DISTANCE_KM,
) -> float:
    """
    Inverse-distance weight using logistic decay:
        weight = 1 / (1 + (distance / half_distance)^2)

    Returns PROXIMITY_MAX for distance=0, decays as distance grows.
    Result is clamped to [0, PROXIMITY_MAX].

    Args:
        distance_km: Distance from event to route buffer edge (km).
        half_distance_km: Distance at which weight = 0.5.
    """
    if distance_km <= 0.0:
        return PROXIMITY_MAX

    ratio = distance_km / half_distance_km
    weight = 1.0 / (1.0 + ratio * ratio)
    return min(max(weight, 0.0), PROXIMITY_MAX)


# ---------------------------------------------------------------------------
# Composite risk feature
# ---------------------------------------------------------------------------

# Weights for the composite formula (must sum to 1.0)
_COMPOSITE_WEIGHTS = {
    "intensity":  0.50,
    "recency":    0.30,
    "proximity":  0.20,
}


def compute_composite_risk(
    intensity_score: float,
    recency_weight: float,
    proximity_weight: float,
) -> float:
    """
    Weighted combination of risk sub-features.

    composite = w_i * intensity + w_r * recency * intensity + w_p * proximity * intensity

    Proximity and recency modulate intensity (not additive),
    ensuring: if intensity=0, composite=0 regardless of distance/time.

    Result: [0, 1]
    """
    w = _COMPOSITE_WEIGHTS

    composite = (
        w["intensity"] * intensity_score
        + w["recency"]    * recency_weight    * intensity_score
        + w["proximity"]  * proximity_weight  * intensity_score
    )
    return round(min(max(composite, 0.0), 1.0), 6)


# ---------------------------------------------------------------------------
# Batch normalization
# ---------------------------------------------------------------------------

def normalize_scores(scores: list[float]) -> list[float]:
    """
    Min-max normalize a list of raw composite scores to [0, 1].
    If all scores are equal, returns uniform 0.5 list.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    return [round((s - lo) / (hi - lo), 6) for s in scores]


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------

def build_event_features(
    event_id: str,
    intensity_score: float,
    event_label: str,
    classifier_confidence: float,
    event_time: datetime,
    distance_km: float,
    reference_time: datetime | None = None,
) -> EventFeatures:
    """
    One-shot constructor that computes all derived features and
    fills the composite_risk field.
    """
    recency = compute_recency_weight(event_time, reference_time)
    proximity = compute_proximity_weight(distance_km)
    composite = compute_composite_risk(intensity_score, recency, proximity)

    return EventFeatures(
        event_id=event_id,
        intensity_score=round(intensity_score, 6),
        recency_weight=round(recency, 6),
        proximity_weight=round(proximity, 6),
        event_label=event_label,
        classifier_confidence=round(classifier_confidence, 6),
        composite_risk=composite,
    )
