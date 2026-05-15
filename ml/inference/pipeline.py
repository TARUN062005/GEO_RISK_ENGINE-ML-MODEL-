"""
ml/inference/pipeline.py
------------------------
ML Inference Pipeline — Ingestion Layer (Log2, upgraded Log11)

Log11: Added run_ml_inference_batch() for batch processing.
  - Batch classifier: one forward pass for N texts
  - NER + scoring still per-event (spaCy is already fast)
  - Net effect: ~2-4x fewer transformer forward passes

The API NEVER imports this module.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ml.classifier import classify_event, classify_batch
from ml.ner import extract_locations
from ml.scoring import (
    score_intensity,
    IntensityInput,
    count_high_severity_keywords,
)
from storage.schema import MLAnnotation

logger = logging.getLogger(__name__)


def run_ml_inference(
    event_id: str,
    text: str,
) -> MLAnnotation:
    """
    Execute the full ML inference chain for a single raw news text.

    Called by ingestion/worker.py after fetching raw articles.
    Result is stored in MongoDB as part of EnrichedEvent.

    Args:
        event_id: GDELT or feed-assigned identifier (for logging).
        text:     Raw article/snippet text.

    Returns:
        MLAnnotation — all pre-computed ML fields, ready for storage.

    Raises:
        Never. All errors are caught internally; falls back to safe defaults.
    """
    try:
        # ── A. Event Classification ──────────────────────────────────────
        classification = classify_event(text)
        logger.debug("[%s] label=%s conf=%.3f method=%s",
                     event_id, classification.label,
                     classification.confidence, classification.method)

        # ── B. NER ──────────────────────────────────────────────────────
        ner_result = extract_locations(text)
        logger.debug("[%s] NER entities=%d method=%s",
                     event_id, len(ner_result.entities), ner_result.method)

        # ── C. Intensity Scoring ─────────────────────────────────────────
        kw_hits = count_high_severity_keywords(text)
        intensity = score_intensity(IntensityInput(
            event_label=classification.label,
            classifier_confidence=classification.confidence,
            ner_entity_count=len(ner_result.entities),
            text_length=len(text),
            keyword_hits=kw_hits,
        ))
        logger.debug("[%s] intensity=%.4f method=%s",
                     event_id, intensity.score, intensity.method)

        return MLAnnotation(
            label=classification.label,
            label_confidence=classification.confidence,
            label_scores=classification.scores,
            classification_method=classification.method,
            location_names=ner_result.unique_locations,
            ner_method=ner_result.method,
            intensity_score=intensity.score,
            intensity_method=intensity.method,
            intensity_explanation=intensity.explanation,
        )

    except Exception as exc:
        logger.exception("[%s] ML inference failed: %s — using safe defaults", event_id, exc)
        return MLAnnotation(
            label="safe",
            label_confidence=0.0,
            label_scores={},
            classification_method="error",
            location_names=[],
            ner_method="error",
            intensity_score=0.0,
            intensity_method="error",
            intensity_explanation={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Log11: Batch ML inference
# ---------------------------------------------------------------------------

def run_ml_inference_batch(
    items: list[tuple[str, str]],
) -> list[MLAnnotation]:
    """
    Log11: Batch ML inference for multiple events.

    Classifier runs in one forward pass (batch_size=8).
    NER + scoring run per-event (spaCy is already fast).

    Args:
        items: list of (event_id, text) tuples

    Returns:
        list of MLAnnotation in same order as input.
    """
    if not items:
        return []

    event_ids = [eid for eid, _ in items]
    texts = [txt for _, txt in items]

    # ── A. Batch classification (one forward pass) ────────────────────
    try:
        classifications = classify_batch(texts)
    except Exception as exc:
        logger.warning("Batch classify failed: %s — falling back to sequential", exc)
        classifications = [classify_event(t) for t in texts]

    logger.info("[Batch ML] Classified %d events in one pass.", len(texts))

    # ── B + C. NER + Intensity per event (already fast) ───────────────
    results: list[MLAnnotation] = []
    for i, (event_id, text) in enumerate(items):
        try:
            cls = classifications[i]
            ner_result = extract_locations(text)

            kw_hits = count_high_severity_keywords(text)
            intensity = score_intensity(IntensityInput(
                event_label=cls.label,
                classifier_confidence=cls.confidence,
                ner_entity_count=len(ner_result.entities),
                text_length=len(text),
                keyword_hits=kw_hits,
            ))

            results.append(MLAnnotation(
                label=cls.label,
                label_confidence=cls.confidence,
                label_scores=cls.scores,
                classification_method=cls.method,
                location_names=ner_result.unique_locations,
                ner_method=ner_result.method,
                intensity_score=intensity.score,
                intensity_method=intensity.method,
                intensity_explanation=intensity.explanation,
            ))

        except Exception as exc:
            logger.warning("[%s] Batch item ML failed: %s", event_id, exc)
            results.append(MLAnnotation(
                label="safe",
                label_confidence=0.0,
                label_scores={},
                classification_method="error",
                location_names=[],
                ner_method="error",
                intensity_score=0.0,
                intensity_method="error",
                intensity_explanation={"error": str(exc)},
            ))

    return results
