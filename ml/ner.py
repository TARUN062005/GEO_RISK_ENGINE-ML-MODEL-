"""
ml/ner.py
---------
Location Extraction Module (Log1)

Extracts geopolitical entities (countries, cities, regions) from news text.
Strategy: spaCy en_core_web_sm  →  HuggingFace token-classifier fallback
Output: list of GeoEntity objects with label, text, and confidence

Labels of interest: GPE (Geo-Political Entity), LOC (Location), FAC (Facility)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class GeoEntity:
    text: str                   # Raw extracted text
    label: str                  # GPE | LOC | FAC | REGION
    start: int                  # Character offset in source text
    end: int                    # Character offset in source text
    confidence: float = 1.0     # spaCy uses 1.0; HF models provide scores


@dataclass
class NERResult:
    entities: list[GeoEntity] = field(default_factory=list)
    method: str = "spacy"       # "spacy" | "hf_token" | "regex"

    @property
    def unique_locations(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for ent in self.entities:
            norm = ent.text.strip().title()
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out


# ---------------------------------------------------------------------------
# Strategy 1: spaCy (primary, lightweight ~15 MB)
# ---------------------------------------------------------------------------

class SpacyNER:
    _nlp = None

    def _load(self):
        if SpacyNER._nlp is None:
            try:
                import spacy  # type: ignore
                SpacyNER._nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded: en_core_web_sm")
            except Exception as exc:
                logger.warning("spaCy load failed (%s).", exc)
                SpacyNER._nlp = None

    def extract(self, text: str) -> Optional[NERResult]:
        self._load()
        if SpacyNER._nlp is None:
            return None

        doc = SpacyNER._nlp(text[:1024])
        entities = [
            GeoEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0,
            )
            for ent in doc.ents
            if ent.label_ in {"GPE", "LOC", "FAC", "NORP"}
        ]
        return NERResult(entities=entities, method="spacy")


# ---------------------------------------------------------------------------
# Strategy 2: HuggingFace token classifier (fallback ~67 MB)
# ---------------------------------------------------------------------------

class HFTokenNER:
    _pipeline = None
    MODEL_NAME = "dslim/bert-base-NER"

    def _load(self):
        if HFTokenNER._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore
                HFTokenNER._pipeline = pipeline(
                    "ner",
                    model=self.MODEL_NAME,
                    aggregation_strategy="simple",
                    device=-1,
                )
                logger.info("HF NER model loaded: %s", self.MODEL_NAME)
            except Exception as exc:
                logger.warning("HF NER load failed (%s).", exc)
                HFTokenNER._pipeline = None

    def extract(self, text: str) -> Optional[NERResult]:
        self._load()
        if HFTokenNER._pipeline is None:
            return None

        try:
            raw = HFTokenNER._pipeline(text[:512])
            entities = [
                GeoEntity(
                    text=item["word"],
                    label=item["entity_group"],
                    start=item["start"],
                    end=item["end"],
                    confidence=round(item["score"], 4),
                )
                for item in raw
                if item["entity_group"] in {"LOC", "GPE"}
            ]
            return NERResult(entities=entities, method="hf_token")
        except Exception as exc:
            logger.warning("HF NER inference error (%s).", exc)
            return None


# ---------------------------------------------------------------------------
# Strategy 3: Regex heuristic (last resort — zero dependencies)
# ---------------------------------------------------------------------------

# Country name list (abbreviated – extend as needed)
_KNOWN_COUNTRIES = {
    "Afghanistan", "Algeria", "Armenia", "Australia", "Azerbaijan",
    "Belarus", "Brazil", "Cambodia", "China", "Colombia", "Croatia",
    "Egypt", "Ethiopia", "France", "Georgia", "Germany", "Ghana",
    "India", "Indonesia", "Iran", "Iraq", "Israel", "Jordan",
    "Kazakhstan", "Kenya", "Lebanon", "Libya", "Malaysia", "Mali",
    "Mexico", "Morocco", "Myanmar", "Nigeria", "North Korea",
    "Pakistan", "Palestine", "Peru", "Philippines", "Poland",
    "Russia", "Rwanda", "Saudi Arabia", "Serbia", "Somalia",
    "South Africa", "South Korea", "Sudan", "Syria", "Taiwan",
    "Turkey", "Ukraine", "United States", "Venezuela", "Yemen",
    "Zimbabwe",
}


def _regex_extract(text: str) -> NERResult:
    entities: list[GeoEntity] = []
    for country in _KNOWN_COUNTRIES:
        for m in re.finditer(rf"\b{re.escape(country)}\b", text, re.IGNORECASE):
            entities.append(GeoEntity(
                text=m.group(),
                label="GPE",
                start=m.start(),
                end=m.end(),
                confidence=0.7,
            ))
    return NERResult(entities=entities, method="regex")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_spacy_ner = SpacyNER()
_hf_ner = HFTokenNER()


def extract_locations(text: str) -> NERResult:
    """
    Cascading NER: spaCy → HF token classifier → regex fallback.
    Always returns a NERResult (never raises).
    """
    result = _spacy_ner.extract(text)
    if result is not None:
        return result

    result = _hf_ner.extract(text)
    if result is not None:
        return result

    logger.warning("All NER strategies failed. Using regex fallback.")
    return _regex_extract(text)
