"""
tests/test_risk.py
------------------
Unit tests for pure risk logic (Log1)

Tests:
  - Recency decay correctness
  - Proximity weighting correctness
  - Composite risk formula
  - Risk aggregation model
  - Edge cases: zero events, single event, all-safe events
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from core.risk.features import (
    compute_recency_weight,
    compute_proximity_weight,
    compute_composite_risk,
    build_event_features,
    normalize_scores,
    RECENCY_HALF_LIFE_DAYS,
    RECENCY_FLOOR,
)
from core.risk.model import compute, RiskScore


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_event(
    event_id: str = "e1",
    intensity_score: float = 0.8,
    event_label: str = "conflict",
    confidence: float = 0.9,
    age_days: float = 0.0,
    distance_km: float = 10.0,
) -> "EventFeatures":  # noqa: F821
    event_time = _now() - timedelta(days=age_days)
    return build_event_features(
        event_id=event_id,
        intensity_score=intensity_score,
        event_label=event_label,
        classifier_confidence=confidence,
        event_time=event_time,
        distance_km=distance_km,
    )


# ---------------------------------------------------------------------------
# Recency decay tests
# ---------------------------------------------------------------------------

class TestRecencyDecay:
    def test_fresh_event_weight_near_one(self):
        w = compute_recency_weight(_now() - timedelta(minutes=5))
        assert w > 0.99

    def test_half_life_at_configured_days(self):
        half_life = RECENCY_HALF_LIFE_DAYS
        w = compute_recency_weight(_now() - timedelta(days=half_life))
        assert abs(w - 0.5) < 0.01, f"Expected ~0.5 at half-life, got {w}"

    def test_old_event_weight_floored(self):
        w = compute_recency_weight(_now() - timedelta(days=365))
        assert w == RECENCY_FLOOR

    def test_future_event_weight_is_one(self):
        w = compute_recency_weight(_now() + timedelta(hours=1))
        assert w == 1.0  # age_days clamped to 0


# ---------------------------------------------------------------------------
# Proximity weighting tests
# ---------------------------------------------------------------------------

class TestProximityWeight:
    def test_zero_distance_max_weight(self):
        w = compute_proximity_weight(0.0)
        assert w == 1.0

    def test_half_distance_equals_half_weight(self):
        from core.risk.features import PROXIMITY_HALF_DISTANCE_KM
        w = compute_proximity_weight(PROXIMITY_HALF_DISTANCE_KM)
        assert abs(w - 0.5) < 0.01, f"Expected ~0.5 at half-distance, got {w}"

    def test_weight_decreases_with_distance(self):
        w1 = compute_proximity_weight(10)
        w2 = compute_proximity_weight(100)
        w3 = compute_proximity_weight(500)
        assert w1 > w2 > w3

    def test_very_far_weight_near_zero(self):
        w = compute_proximity_weight(10_000)
        assert w < 0.01


# ---------------------------------------------------------------------------
# Composite risk formula tests
# ---------------------------------------------------------------------------

class TestCompositeRisk:
    def test_zero_intensity_yields_zero_composite(self):
        c = compute_composite_risk(0.0, 1.0, 1.0)
        assert c == 0.0

    def test_composite_bounded_zero_one(self):
        c = compute_composite_risk(1.0, 1.0, 1.0)
        assert 0.0 <= c <= 1.0

    def test_high_intensity_high_recency_high_proximity_near_one(self):
        c = compute_composite_risk(1.0, 1.0, 1.0)
        assert c > 0.9

    def test_composite_monotone_in_intensity(self):
        c_low = compute_composite_risk(0.2, 0.8, 0.8)
        c_high = compute_composite_risk(0.9, 0.8, 0.8)
        assert c_high > c_low


# ---------------------------------------------------------------------------
# Aggregation model tests
# ---------------------------------------------------------------------------

class TestRiskAggregation:
    def test_empty_events_returns_low(self):
        result: RiskScore = compute([])
        assert result.final_score == 0.0
        assert result.risk_band == "LOW"

    def test_single_conflict_event_high_score(self):
        ev = _make_event(event_label="conflict", intensity_score=0.9, distance_km=5.0)
        result = compute([ev])
        assert result.risk_band in ("HIGH", "CRITICAL")

    def test_safe_events_yield_low_score(self):
        events = [_make_event(event_id=str(i), event_label="safe", intensity_score=0.05)
                  for i in range(5)]
        result = compute(events)
        assert result.final_score < 0.2

    def test_mixed_events_dominated_by_conflict(self):
        events = [
            _make_event("e1", event_label="conflict", intensity_score=0.95, distance_km=5),
            _make_event("e2", event_label="safe",     intensity_score=0.05, distance_km=5),
            _make_event("e3", event_label="protest",  intensity_score=0.40, distance_km=5),
        ]
        result = compute(events)
        assert result.dominant_event_label == "conflict"

    def test_explanation_dict_populated(self):
        ev = _make_event()
        result = compute([ev])
        assert "max_adjusted_score" in result.explanation
        assert "event_count" in result.explanation

    def test_risk_band_thresholds(self):
        from core.risk.model import _risk_band
        assert _risk_band(0.10) == "LOW"
        assert _risk_band(0.35) == "MEDIUM"
        assert _risk_band(0.60) == "HIGH"
        assert _risk_band(0.85) == "CRITICAL"


# ---------------------------------------------------------------------------
# Normalize scores
# ---------------------------------------------------------------------------

class TestNormalizeScores:
    def test_normalization_range(self):
        scores = [0.1, 0.5, 0.9, 0.3, 0.7]
        norm = normalize_scores(scores)
        assert min(norm) == pytest.approx(0.0)
        assert max(norm) == pytest.approx(1.0)

    def test_uniform_scores_return_half(self):
        norm = normalize_scores([0.5, 0.5, 0.5])
        assert all(s == 0.5 for s in norm)

    def test_empty_input(self):
        assert normalize_scores([]) == []
