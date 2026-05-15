"""
ml/train/intensity_trainer.py
-----------------------------
Offline Training Pipeline — Intensity Scorer (Log2)

Updated from Log1: train_intensity_model() has moved from ml/scoring.py
into this dedicated training module. ml/scoring.py is now inference-only.

Run offline via:
    python -m ml.train.intensity_trainer

Never imported during API startup or ingestion.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np  # type: ignore

logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).parent.parent / "artifacts" / "intensity_lr.pkl"


def train(X: np.ndarray, y: np.ndarray, save_path: Path = ARTIFACT_PATH) -> None:
    """
    Train and persist a LogisticRegression intensity scorer.

    Args:
        X: Feature matrix, shape (n_samples, 6).
           Build with ml.scoring.extract_intensity_features().
        y: Binary labels (0=low intensity, 1=high intensity).
        save_path: Path to write the serialized sklearn Pipeline.
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.preprocessing import StandardScaler      # type: ignore
    from sklearn.pipeline import Pipeline                 # type: ignore
    from sklearn.model_selection import cross_val_score   # type: ignore

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
    ])

    # Cross-validate before persisting
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    logger.info("CV ROC-AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    clf.fit(X, y)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as f:
        pickle.dump(clf, f)

    logger.info("Saved intensity model to %s  (CV AUC=%.4f)", save_path, cv_scores.mean())


def build_training_data_from_mongo(
    mongo_uri: str,
    db_name: str = "geo_risk",
    collection: str = "geo_events",
    min_confidence: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pull labeled events from MongoDB and convert to (X, y) training arrays.

    Labeling convention:
        intensity_score >= 0.5  →  y = 1 (high)
        intensity_score <  0.5  →  y = 0 (low)

    Args:
        mongo_uri:      MongoDB connection string.
        min_confidence: Minimum label_confidence to include a sample.
    """
    from pymongo import MongoClient                              # type: ignore
    from ml.scoring import extract_intensity_features, IntensityInput

    client = MongoClient(mongo_uri)
    docs = list(
        client[db_name][collection].find(
            {"ml.label_confidence": {"$gte": min_confidence}},
            {"ml": 1, "raw_text": 1},
        )
    )
    logger.info("Fetched %d training samples from MongoDB.", len(docs))

    X_rows, y_rows = [], []
    for doc in docs:
        ml = doc["ml"]
        inp = IntensityInput(
            event_label=ml["label"],
            classifier_confidence=ml["label_confidence"],
            ner_entity_count=len(ml.get("location_names", [])),
            text_length=len(doc.get("raw_text", "")),
            keyword_hits=0,   # recalculate if needed
        )
        from ml.scoring import extract_intensity_features
        X_rows.append(extract_intensity_features(inp))
        y_rows.append(1 if ml["intensity_score"] >= 0.5 else 0)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    X, y = build_training_data_from_mongo(mongo_uri)
    if len(X) < 20:
        logger.warning("Too few samples (%d) — skipping training.", len(X))
    else:
        train(X, y)
