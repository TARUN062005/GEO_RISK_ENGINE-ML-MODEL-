"""
ingestion/entity_normalizer.py
------------------------------
Entity Normalization Pipeline (Log15)

Cleans NER-extracted location names before geocoding to reduce
Nominatim failures and improve cache hit rates.

Pipeline:
  1. Unicode normalization (NFC)
  2. Strip control characters and HTML entities
  3. Remove possessives and trailing punctuation
  4. Filter garbage tokens (numbers-only, single chars, URLs, etc.)
  5. Reject non-geographic entities (people names, organizations)
  6. Normalize common abbreviations
  7. Quality scoring

Reduces: "Nominatim returned no result" errors
Improves: Geocode cache hit rates
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Garbage patterns — tokens that should never be geocoded
# ---------------------------------------------------------------------------

# Common non-location entities that NER misclassifies as GPE/LOC
_NON_LOCATION_PATTERNS = [
    # People / organizations
    r"^(?:Mr|Mrs|Dr|Prof|Gen|Col|Sgt|Cpl|Lt|Maj|Capt|Adm|Sen|Rep|Gov|Pres)\b",
    r"^(?:UN|NATO|EU|ASEAN|OPEC|WHO|IMF|FBI|CIA|ISIS|ISIL|Hamas|Hezbollah|Taliban)$",
    # News artifacts
    r"^(?:AP|AFP|Reuters|BBC|CNN|NPR|Fox|NBC|CBS|ABC)$",
    r"^(?:BREAKING|UPDATE|URGENT|EXCLUSIVE|DEVELOPING|LIVE|OPINION|ANALYSIS)$",
    # Dates / time references
    r"^\d{4}$",  # year only
    r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$",
    r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)$",
    # Generic noise
    r"^[A-Z]{1,2}$",  # Single/double letters
    r"^\d+$",  # Pure numbers
    r"^https?://",  # URLs
    r"^[^\w\s]+$",  # Pure punctuation
    r"^.{1,2}$",  # Tokens under 3 chars
]

_COMPILED_NON_LOCATION = [re.compile(p, re.IGNORECASE) for p in _NON_LOCATION_PATTERNS]

# Common abbreviation → full name mappings
_ABBREVIATION_MAP = {
    "US": "United States",
    "USA": "United States",
    "U.S.": "United States",
    "U.S.A.": "United States",
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "UAE": "United Arab Emirates",
    "U.A.E.": "United Arab Emirates",
    "DPRK": "North Korea",
    "ROK": "South Korea",
    "PRC": "China",
    "KSA": "Saudi Arabia",
    "DRC": "Democratic Republic of Congo",
    "CAR": "Central African Republic",
    "S. Korea": "South Korea",
    "N. Korea": "North Korea",
    "S. Africa": "South Africa",
}


# ---------------------------------------------------------------------------
# Normalization pipeline
# ---------------------------------------------------------------------------

def normalize_entity(raw: str) -> Optional[str]:
    """
    Normalize a raw NER entity for geocoding.

    Returns:
        Cleaned entity string, or None if the entity should be rejected.
    """
    if not raw or not isinstance(raw, str):
        return None

    text = raw

    # 1. Unicode normalization (NFC — canonical decomposition + composition)
    text = unicodedata.normalize("NFC", text)

    # 2. Strip HTML entities
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)

    # 3. Strip control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

    # 4. Collapse whitespace
    text = " ".join(text.split())

    # 5. Remove possessives
    text = re.sub(r"'s$", "", text)
    text = re.sub(r"'s$", "", text)  # curly apostrophe

    # 6. Remove trailing/leading punctuation (but keep internal hyphens, dots for place names)
    text = text.strip(".,;:!?\"'`()[]{}#*@^~|\\/<>")

    # 7. Check abbreviation map
    upper = text.upper().strip(".")
    if text in _ABBREVIATION_MAP:
        text = _ABBREVIATION_MAP[text]
    elif upper in _ABBREVIATION_MAP:
        text = _ABBREVIATION_MAP[upper]

    # 8. Quality checks
    if not text or len(text) < 2:
        return None

    # Reject garbage patterns
    for pattern in _COMPILED_NON_LOCATION:
        if pattern.search(text):
            logger.debug("Entity rejected (garbage pattern): '%s'", raw)
            return None

    # Reject if >50% digits
    digit_count = sum(1 for ch in text if ch.isdigit())
    if len(text) > 0 and digit_count / len(text) > 0.5:
        logger.debug("Entity rejected (too many digits): '%s'", raw)
        return None

    # Reject if no alphabetic characters
    if not any(ch.isalpha() for ch in text):
        logger.debug("Entity rejected (no alphabetic chars): '%s'", raw)
        return None

    # 9. Title case normalization for consistency
    # But preserve all-caps acronyms that we already expanded
    if text.isupper() and len(text) > 3:
        text = text.title()
    elif not text[0].isupper():
        text = text.title()

    return text


def normalize_entities(raw_names: list[str]) -> list[str]:
    """
    Normalize a list of NER entities.
    Deduplicates and filters garbage.

    Returns cleaned list (may be shorter than input).
    """
    seen: set[str] = set()
    result: list[str] = []

    for raw in raw_names:
        cleaned = normalize_entity(raw)
        if cleaned is None:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)

    if len(raw_names) > len(result):
        logger.debug(
            "Entity normalization: %d → %d (filtered %d garbage tokens)",
            len(raw_names), len(result), len(raw_names) - len(result),
        )

    return result
