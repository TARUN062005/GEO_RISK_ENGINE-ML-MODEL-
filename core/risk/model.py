"""
core/risk/model.py
------------------
Risk Aggregation Model (Log1)

Aggregates per-event risk features across all events within a route buffer
into a single, explainable final risk score.

Design decisions:
  - Weighted-max + weighted-mean hybrid (avoids dominance of outliers)
  - Event label severity multiplier for domain-prior injection
  - Percentile-capped aggregation to dampen noise
  - Full explanation dict returned alongside score
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

from core.risk.features import EventFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Weight of max-score vs mean-score in final blend
BLEND_MAX_WEIGHT: float = 0.60
BLEND_MEAN_WEIGHT: float = 0.40   # must sum to 1.0

# Label-level severity multipliers (caps at 1.0 after application)
_LABEL_MULTIPLIERS: dict[str, float] = {
    "conflict":  1.00,
    "terrorism": 1.05,
    "sanction":  0.75,
    "protest":   0.65,
    "disaster":  0.85,
    "safe":      0.10,
}

# Minimum event count to trust aggregation; below this → uncertainty penalty
MIN_RELIABLE_EVENT_COUNT: int = 3


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class RiskScore:
    """Final aggregated risk output for a route segment."""
    final_score: float                     # [0, 1]
    risk_band: str                         # LOW | MEDIUM | HIGH | CRITICAL
    event_count: int
    dominant_event_label: str
    explanation: dict[str, float | str | int] = field(default_factory=dict)
    contributing_events: list[str] = field(default_factory=list)  # event IDs


# ---------------------------------------------------------------------------
# Risk band thresholds
# ---------------------------------------------------------------------------

def _risk_band(score: float) -> str:
    if score < 0.25:
        return "LOW"
    elif score < 0.50:
        return "MEDIUM"
    elif score < 0.75:
        return "HIGH"
    return "CRITICAL"


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def compute(events: list[EventFeatures]) -> RiskScore:
    """
    Aggregate a list of EventFeatures into a single RiskScore.

    Algorithm:
      1. Compute per-event adjusted score (composite × label_multiplier)
      2. Blend: 60% max-score + 40% trimmed-mean
      3. Apply uncertainty penalty if event count < threshold
      4. Clamp to [0, 1]

    Args:
        events: List of engineered feature objects for events in route buffer.

    Returns:
        RiskScore with final score, band, and explanation dict.
    """
    if not events:
        return RiskScore(
            final_score=0.0,
            risk_band="LOW",
            event_count=0,
            dominant_event_label="none",
            explanation={"reason": "no_events_in_buffer"},
        )

    # Step 1: compute adjusted scores
    adjusted: list[tuple[EventFeatures, float]] = []
    for ev in events:
        multiplier = _LABEL_MULTIPLIERS.get(ev.event_label, 0.5)
        adj = min(ev.composite_risk * multiplier, 1.0)
        adjusted.append((ev, adj))

    adjusted.sort(key=lambda x: x[1], reverse=True)
    adj_scores = [adj for _, adj in adjusted]

    # Step 2: max component
    max_score = adj_scores[0]

    # Step 3: trimmed mean (drop top 10% / bottom 10% for robustness)
    trimmed = _trimmed_mean(adj_scores)

    # Step 4: blend
    blended = BLEND_MAX_WEIGHT * max_score + BLEND_MEAN_WEIGHT * trimmed

    # Step 5: uncertainty penalty for sparse data
    uncertainty_penalty = 1.0
    if len(events) < MIN_RELIABLE_EVENT_COUNT:
        uncertainty_penalty = 0.85
        logger.debug("Sparse event set (%d events); applying uncertainty penalty.", len(events))

    final = min(blended * uncertainty_penalty, 1.0)

    # Dominant label = most common label in top-50% events
    top_half = [ev.event_label for ev, _ in adjusted[: max(1, len(adjusted) // 2)]]
    dominant_label = max(set(top_half), key=top_half.count)

    explanation = {
        "max_adjusted_score": round(max_score, 6),
        "trimmed_mean_score": round(trimmed, 6),
        "blend_max_weight":   BLEND_MAX_WEIGHT,
        "blend_mean_weight":  BLEND_MEAN_WEIGHT,
        "uncertainty_penalty": round(uncertainty_penalty, 4),
        "blended_pre_penalty": round(blended, 6),
        "event_count":        len(events),
        "dominant_label":     dominant_label,
    }

    return RiskScore(
        final_score=round(final, 6),
        risk_band=_risk_band(final),
        event_count=len(events),
        dominant_event_label=dominant_label,
        explanation=explanation,
        contributing_events=[ev.event_id for ev, _ in adjusted[:5]],  # top-5
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trimmed_mean(scores: list[float], trim_fraction: float = 0.10) -> float:
    """
    Compute mean after removing the top and bottom `trim_fraction` of values.
    Falls back to simple mean for very small lists.
    """
    n = len(scores)
    if n < 4:
        return sum(scores) / n

    k = max(1, math.floor(n * trim_fraction))
    trimmed = sorted(scores)[k: n - k]
    return sum(trimmed) / len(trimmed)
