"""
ml/ner.py
---------
Location Extraction Module (Log1, optimized Log16)

Extracts geopolitical entities (countries, cities, regions) from news text.
Strategy: spaCy en_core_web_sm  →  regex fallback
Output: list of GeoEntity objects with label, text, and confidence

Log16: Removed HuggingFace token-classifier fallback (dslim/bert-base-NER).
  - Eliminates ~400MB+ of transformer + torch runtime memory
  - spaCy en_core_web_sm is the primary NER (~15MB, accurate for locations)
  - Expanded regex fallback with more countries and major cities
  - Zero transformer dependencies

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
    confidence: float = 1.0     # spaCy uses 1.0; regex uses 0.7


@dataclass
class NERResult:
    entities: list[GeoEntity] = field(default_factory=list)
    method: str = "spacy"       # "spacy" | "regex"

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
# Strategy 2: Regex heuristic (fallback — zero dependencies)
# Log16: Expanded with more countries and major strategic cities/regions
# ---------------------------------------------------------------------------

_KNOWN_LOCATIONS = {
    # Countries (comprehensive geopolitical coverage)
    "Afghanistan", "Algeria", "Angola", "Argentina", "Armenia", "Australia",
    "Azerbaijan", "Bahrain", "Bangladesh", "Belarus", "Belgium", "Brazil",
    "Cambodia", "Cameroon", "Canada", "Chad", "Chile", "China", "Colombia",
    "Congo", "Croatia", "Cuba", "Cyprus", "Czech Republic",
    "Denmark", "Djibouti", "Ecuador", "Egypt", "Eritrea", "Estonia",
    "Ethiopia", "Finland", "France", "Gabon", "Georgia", "Germany", "Ghana",
    "Greece", "Guatemala", "Guinea", "Haiti", "Honduras", "Hungary",
    "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
    "Ivory Coast", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya",
    "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Libya",
    "Lithuania", "Madagascar", "Malaysia", "Mali", "Mauritania", "Mexico",
    "Moldova", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar",
    "Namibia", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger",
    "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman",
    "Pakistan", "Palestine", "Panama", "Paraguay", "Peru", "Philippines",
    "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda",
    "Saudi Arabia", "Senegal", "Serbia", "Sierra Leone", "Singapore",
    "Slovakia", "Slovenia", "Somalia", "South Africa", "South Korea",
    "South Sudan", "Spain", "Sri Lanka", "Sudan", "Sweden", "Switzerland",
    "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Togo",
    "Trinidad", "Tunisia", "Turkey", "Turkmenistan", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay",
    "Uzbekistan", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",
    # Strategic cities and regions
    "Kyiv", "Moscow", "Beijing", "Tehran", "Baghdad", "Kabul", "Damascus",
    "Beirut", "Tripoli", "Mogadishu", "Khartoum", "Sanaa", "Aden",
    "Gaza", "Jerusalem", "Tel Aviv", "Riyadh", "Ankara", "Istanbul",
    "Taipei", "Pyongyang", "Seoul", "Doha", "Dubai", "Abu Dhabi",
    "Islamabad", "Karachi", "Mumbai", "New Delhi", "Shanghai", "Hong Kong",
    "Crimea", "Donbas", "Donetsk", "Luhansk", "Kherson", "Zaporizhzhia",
    "Idlib", "Aleppo", "Raqqa", "Mosul", "Basra", "Kirkuk",
    "Xinjiang", "Tibet", "Kashmir",
    # Strategic waterways and regions
    "Red Sea", "Black Sea", "South China Sea", "East China Sea",
    "Persian Gulf", "Arabian Sea", "Gulf of Aden", "Mediterranean",
    "Strait of Hormuz", "Suez Canal", "Bab el-Mandeb", "Malacca Strait",
    "Taiwan Strait", "Baltic Sea", "Caspian Sea", "Arctic",
    "Sahel", "Horn of Africa", "Caucasus", "Balkans", "Levant",
}

# Pre-compile regex patterns for each location (done once at import)
_LOCATION_PATTERNS: list[tuple[str, re.Pattern]] = [
    (loc, re.compile(rf"\b{re.escape(loc)}\b", re.IGNORECASE))
    for loc in _KNOWN_LOCATIONS
]


def _regex_extract(text: str) -> NERResult:
    """Regex-based location extraction — zero dependencies."""
    entities: list[GeoEntity] = []
    seen_spans: set[tuple[int, int]] = set()

    for loc_name, pattern in _LOCATION_PATTERNS:
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            if span not in seen_spans:
                seen_spans.add(span)
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


def extract_locations(text: str) -> NERResult:
    """
    Log16: Cascading NER — spaCy → regex fallback.
    HuggingFace NER removed to eliminate transformer/torch dependency.
    Always returns a NERResult (never raises).
    """
    result = _spacy_ner.extract(text)
    if result is not None:
        return result

    logger.debug("spaCy NER unavailable. Using regex fallback.")
    return _regex_extract(text)
