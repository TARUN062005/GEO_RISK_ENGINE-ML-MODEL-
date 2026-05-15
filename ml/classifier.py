"""
ml/classifier.py
----------------
Event Classification Module (Log1)

Classifies raw news text into geopolitical event categories.
Strategy: Zero-shot classification with facebook/bart-large-mnli
Fallback: Keyword-based heuristic classifier (CPU-safe, zero-dep)

Categories: conflict, protest, sanction, disaster, terrorism, safe
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    label: str
    confidence: float
    scores: dict[str, float]
    method: str  # "zero_shot" | "heuristic"


# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------

EVENT_LABELS = [
    "armed conflict",
    "protest or civil unrest",
    "economic sanction",
    "natural disaster",
    "terrorist attack",
    "political stability",
]

LABEL_CANONICAL = {
    "armed conflict": "conflict",
    "protest or civil unrest": "protest",
    "economic sanction": "sanction",
    "natural disaster": "disaster",
    "terrorist attack": "terrorism",
    "political stability": "safe",
}

# ---------------------------------------------------------------------------
# Heuristic fallback (keyword rules – no model required)
# ---------------------------------------------------------------------------

_KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("conflict",   ["war", "airstrike", "shelling", "military", "troops", "invasion", "battle", "missile"]),
    ("protest",    ["protest", "demonstration", "riot", "march", "strike", "rally", "unrest", "clashes"]),
    ("sanction",   ["sanction", "embargo", "tariff", "trade ban", "restriction", "blacklist"]),
    ("disaster",   ["earthquake", "flood", "hurricane", "cyclone", "tsunami", "wildfire", "drought"]),
    ("terrorism",  ["bomb", "explosion", "terror", "attack", "suicide bomber", "ied", "hostage"]),
    ("safe",       ["peace", "ceasefire", "agreement", "treaty", "diplomat", "cooperation"]),
]


def _heuristic_classify(text: str) -> ClassificationResult:
    """Keyword frequency-based fallback classifier."""
    lower = text.lower()
    scores: dict[str, float] = {label: 0.0 for label, _ in _KEYWORD_RULES}

    for label, keywords in _KEYWORD_RULES:
        hits = sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", lower))
        scores[label] = min(hits / max(len(keywords), 1), 1.0)

    best_label = max(scores, key=scores.__getitem__)
    best_score = scores[best_label]

    # Default to "safe" if no signal found
    if best_score == 0.0:
        best_label, best_score = "safe", 0.5

    return ClassificationResult(
        label=best_label,
        confidence=round(best_score, 4),
        scores=scores,
        method="heuristic",
    )


# ---------------------------------------------------------------------------
# Zero-shot classifier (transformers)
# ---------------------------------------------------------------------------

class ZeroShotClassifier:
    """
    Wraps HuggingFace zero-shot pipeline.
    Model: facebook/bart-large-mnli (~1.6 GB, CPU-friendly)
    Lite alternative: cross-encoder/nli-MiniLM2-L6-H768 (~90 MB)
    """

    _pipeline = None  # lazy singleton

    def __init__(self, model_name: str = "cross-encoder/nli-MiniLM2-L6-H768"):
        self.model_name = model_name

    def _load(self):
        if ZeroShotClassifier._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore
                logger.info("Loading zero-shot model: %s", self.model_name)
                ZeroShotClassifier._pipeline = pipeline(
                    "zero-shot-classification",
                    model=self.model_name,
                    device=-1,  # CPU
                )
                logger.info("Model loaded successfully.")
            except Exception as exc:
                logger.warning("Model load failed (%s). Using heuristic fallback.", exc)
                ZeroShotClassifier._pipeline = None

    def classify(self, text: str) -> ClassificationResult:
        self._load()

        if ZeroShotClassifier._pipeline is None:
            return _heuristic_classify(text)

        try:
            result = ZeroShotClassifier._pipeline(
                text[:512],          # truncate for speed
                candidate_labels=EVENT_LABELS,
                multi_label=False,
            )
            scores = {
                LABEL_CANONICAL[lbl]: round(score, 4)
                for lbl, score in zip(result["labels"], result["scores"])
            }
            top_label = LABEL_CANONICAL[result["labels"][0]]
            top_score = result["scores"][0]

            return ClassificationResult(
                label=top_label,
                confidence=round(top_score, 4),
                scores=scores,
                method="zero_shot",
            )
        except Exception as exc:
            logger.warning("Zero-shot inference error (%s). Falling back.", exc)
            return _heuristic_classify(text)

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """
        Log11: Batch classification — processes multiple texts in one forward pass.
        Falls back to heuristic per-event if model unavailable.
        """
        if not texts:
            return []

        self._load()

        if ZeroShotClassifier._pipeline is None:
            return [_heuristic_classify(t) for t in texts]

        try:
            truncated = [t[:512] for t in texts]
            results = ZeroShotClassifier._pipeline(
                truncated,
                candidate_labels=EVENT_LABELS,
                multi_label=False,
                batch_size=8,
            )
            # Pipeline returns single dict for single input, list for multiple
            if isinstance(results, dict):
                results = [results]

            out: list[ClassificationResult] = []
            for result in results:
                scores = {
                    LABEL_CANONICAL[lbl]: round(score, 4)
                    for lbl, score in zip(result["labels"], result["scores"])
                }
                out.append(ClassificationResult(
                    label=LABEL_CANONICAL[result["labels"][0]],
                    confidence=round(result["scores"][0], 4),
                    scores=scores,
                    method="zero_shot",
                ))
            return out
        except Exception as exc:
            logger.warning("Batch zero-shot failed (%s). Using heuristic.", exc)
            return [_heuristic_classify(t) for t in texts]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_classifier: Optional[ZeroShotClassifier] = None


def get_classifier(model_name: str = "cross-encoder/nli-MiniLM2-L6-H768") -> ZeroShotClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ZeroShotClassifier(model_name=model_name)
    return _classifier


def classify_event(text: str) -> ClassificationResult:
    """Main entrypoint — classify a single news snippet."""
    return get_classifier().classify(text)


def classify_batch(texts: list[str]) -> list[ClassificationResult]:
    """Log11: Batch entrypoint — classify multiple texts in one pass."""
    return get_classifier().classify_batch(texts)
