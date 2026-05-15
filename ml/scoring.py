"""
ml/scoring.py
-------------
Intensity Scoring Model (Log1)

Converts raw classifier output + contextual signals into a
normalized intensity score in [0, 1].

Two modes:
  1. Rule-based scorer  – deterministic, zero-dep, explainable
  2. Learned scorer     – LogisticRegression on hand-crafted features
                          (trains on labeled dataset if available)

The final score feeds into core.risk.model for risk aggregation.
"""

from __future__ import annotations

import math
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class IntensityInput:
    """Structured input to the intensity scorer."""
    event_label: str           # e.g. "conflict", "protest"
    classifier_confidence: float  # [0, 1] from classifier
    ner_entity_count: int      # number of geo entities found
    text_length: int           # raw token/char count
    keyword_hits: int          # matched high-severity keywords


@dataclass
class IntensityResult:
    score: float               # normalized [0, 1]
    explanation: dict[str, float]  # feature → contribution
    method: str                # "rule_based" | "logistic"


# ---------------------------------------------------------------------------
# Label-based base severity (domain knowledge)
# ---------------------------------------------------------------------------

_LABEL_BASE_SEVERITY: dict[str, float] = {
    "conflict":   0.90,
    "terrorism":  0.95,
    "sanction":   0.55,
    "protest":    0.40,
    "disaster":   0.70,
    "safe":       0.05,
}

# High-severity keywords for bonus signal
_HIGH_SEVERITY_KEYWORDS = [
    "nuclear", "chemical weapon", "genocide", "massacre", "civilian",
    "coup", "assassination", "blockade", "siege", "escalation",
    "warhead", "bombing", "airstrike", "casualt",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_intensity_features(inp: IntensityInput) -> np.ndarray:
    """
    Returns a 1-D numpy feature vector (6 features):
      [0] label_severity        – base score from event label
      [1] classifier_confidence – model certainty
      [2] ner_density           – geo entities / sqrt(text_length)
      [3] keyword_intensity     – keyword_hits / 14 (max known)
      [4] text_richness         – log(text_length + 1) / 10
      [5] composite             – label_severity * confidence
    """
    label_sev = _LABEL_BASE_SEVERITY.get(inp.event_label, 0.3)
    ner_density = inp.ner_entity_count / max(math.sqrt(inp.text_length + 1), 1)
    kw_intensity = min(inp.keyword_hits / len(_HIGH_SEVERITY_KEYWORDS), 1.0)
    text_rich = math.log(inp.text_length + 1) / 10.0
    composite = label_sev * inp.classifier_confidence

    return np.array([
        label_sev,
        inp.classifier_confidence,
        min(ner_density, 1.0),
        kw_intensity,
        min(text_rich, 1.0),
        composite,
    ], dtype=np.float32)


def count_high_severity_keywords(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for kw in _HIGH_SEVERITY_KEYWORDS if kw in text_lower)


# ---------------------------------------------------------------------------
# Rule-based scorer (primary for Log1)
# ---------------------------------------------------------------------------

_FEATURE_WEIGHTS = np.array([0.30, 0.20, 0.10, 0.20, 0.05, 0.15], dtype=np.float32)


def _rule_based_score(features: np.ndarray) -> tuple[float, dict[str, float]]:
    feature_names = [
        "label_severity", "classifier_confidence", "ner_density",
        "keyword_intensity", "text_richness", "composite",
    ]
    contributions = {name: float(features[i] * _FEATURE_WEIGHTS[i])
                     for i, name in enumerate(feature_names)}
    raw_score = float(np.dot(features, _FEATURE_WEIGHTS))
    return min(max(raw_score, 0.0), 1.0), contributions


# ---------------------------------------------------------------------------
# Learned scorer (LogisticRegression wrapper)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "artifacts" / "intensity_lr.pkl"


class LearnedIntensityScorer:
    """
    Thin wrapper around a persisted sklearn LogisticRegression.
    Falls back to rule-based if model artifact is missing.
    """
    _model = None
    _load_attempted = False  # Log10: prevent log spam

    def _load(self):
        if LearnedIntensityScorer._load_attempted:
            return
        LearnedIntensityScorer._load_attempted = True

        if MODEL_PATH.exists():
            try:
                with MODEL_PATH.open("rb") as f:
                    LearnedIntensityScorer._model = pickle.load(f)
                logger.info("Loaded intensity LR model from %s", MODEL_PATH)
            except Exception as exc:
                logger.warning("Failed to load LR model (%s). Using rule-based.", exc)
        else:
            logger.info("No trained model artifact at %s. Using rule-based.", MODEL_PATH)

    def predict(self, features: np.ndarray) -> Optional[float]:
        self._load()
        if LearnedIntensityScorer._model is None:
            return None
        try:
            proba = LearnedIntensityScorer._model.predict_proba(features.reshape(1, -1))[0]
            # Assume binary: class 1 = "high intensity"
            return float(proba[1])
        except Exception as exc:
            logger.warning("LR inference error (%s).", exc)
            return None


_learned_scorer = LearnedIntensityScorer()


# ---------------------------------------------------------------------------
# Training helper (offline use)
# ---------------------------------------------------------------------------

def train_intensity_model(
    X: np.ndarray,
    y: np.ndarray,
    save_path: Path = MODEL_PATH,
) -> None:
    """
    Train and persist LogisticRegression on labeled feature matrix.
    X shape: (n_samples, 6)
    y shape: (n_samples,) – binary (0=low, 1=high)
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.preprocessing import StandardScaler      # type: ignore
    from sklearn.pipeline import Pipeline                 # type: ignore

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
    ])
    clf.fit(X, y)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as f:
        pickle.dump(clf, f)
    logger.info("Saved intensity model to %s", save_path)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def score_intensity(inp: IntensityInput) -> IntensityResult:
    """
    Main entry — returns normalized intensity score in [0, 1].
    Tries learned model first; falls back to rule-based.
    """
    features = extract_intensity_features(inp)

    # Attempt learned model
    learned_score = _learned_scorer.predict(features)
    if learned_score is not None:
        _, contributions = _rule_based_score(features)  # reuse for explainability
        contributions["method_override"] = learned_score
        return IntensityResult(
            score=learned_score,
            explanation=contributions,
            method="logistic",
        )

    # Rule-based fallback
    score, contributions = _rule_based_score(features)
    return IntensityResult(score=score, explanation=contributions, method="rule_based")
