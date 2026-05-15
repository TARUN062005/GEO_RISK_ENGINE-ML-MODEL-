"""
ingestion/relevance_filter.py
-----------------------------
Geopolitical Relevance Filter (Log8)

Rejects articles that are NOT related to geopolitical risk.

Problem this solves:
  RSS feeds return celebrity news, sports, entertainment, etc.
  These waste DB space and corrupt risk analysis.

Strategy:
  1. Fast keyword pre-filter (reject obvious irrelevant content)
  2. ML label check post-classification (reject "safe" with high confidence)

Categories KEPT:
  - armed conflict, military operations
  - sanctions, embargoes, trade restrictions
  - terrorism, extremism
  - maritime threats (piracy, shipping disruption)
  - airspace closures
  - geopolitical instability
  - logistics / supply chain disruption
  - natural disasters with strategic impact

Categories REJECTED:
  - celebrity, entertainment, gossip
  - sports, culture
  - lifestyle, fashion, cooking
  - local crime (non-geopolitical)
  - weather (non-disaster)
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rejection patterns — if ANY match, article is REJECTED immediately
# ---------------------------------------------------------------------------

_REJECT_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # Entertainment / celebrity
        r"\b(celebrity|gossip|red\s+carpet|box\s+office|movie\s+review|album\s+release)\b",
        r"\b(reality\s+tv|award\s+show|grammy|oscar|emmy|golden\s+globe|brit\s+award)\b",
        r"\b(kardashian|beyonce|taylor\s+swift|justin\s+bieber|kanye)\b",
        # Sports
        r"\b(premier\s+league|champions\s+league|world\s+cup|super\s+bowl|olympics|nba|nfl|fifa)\b",
        r"\b(cricket\s+match|tennis\s+open|formula\s+one|grand\s+prix)\b",
        r"\b(goal\s+scored|hat[\s-]trick|match\s+result|season\s+finale)\b",
        # Lifestyle
        r"\b(recipe|cooking|fashion\s+week|beauty\s+tips|diet\s+plan|wellness)\b",
        r"\b(royal\s+visit|royal\s+family|king\s+charles.*visit|prince.*charity)\b",
        # Technology (non-geopolitical)
        r"\b(iphone|android\s+update|app\s+store|tech\s+review|gadget|release\s+date|streaming)\b",
        r"\b(video\s+game|playstation|xbox|nintendo|smartphone|wearable|smartwatch)\b",
        # Local/trivial
        r"\b(lottery\s+winner|dog\s+show|spelling\s+bee|bake[\s-]off)\b",
    ]
]

# ---------------------------------------------------------------------------
# Acceptance patterns — if ANY match, article is KEPT regardless
# ---------------------------------------------------------------------------

_ACCEPT_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # Conflict
        r"\b(war|military|troops|invasion|airstrike|missile|drone\s+strike|shelling)\b",
        r"\b(battle|combat|offensive|ceasefire|casualt|killed|wounded|conflict)\b",
        # Terrorism
        r"\b(terror|bomb|explosion|ied|hostage|extremis|insurgent|militant)\b",
        # Sanctions / trade
        r"\b(sanction|embargo|tariff|trade\s+ban|blacklist|restriction|export\s+control)\b",
        # Maritime
        r"\b(piracy|hijack|ship.*attack|naval|maritime|chokepoint|strait|shipping\s+lane)\b",
        r"\b(blockade|port\s+closure|canal|strait.*hormuz|red\s+sea|suez)\b",
        # Geopolitical
        r"\b(coup|junta|regime|annexation|territorial|sovereignty|border\s+clash)\b",
        r"\b(nuclear|chemical\s+weapon|warhead|escalation|mobilization)\b",
        r"\b(refugee|displacement|humanitarian\s+crisis|famine)\b",
        # Supply chain / logistics
        r"\b(supply\s+chain|logistics\s+disruption|cargo|freight|port\s+congestion)\b",
        r"\b(airspace\s+clos|flight\s+ban|no[\s-]fly\s+zone|diversion)\b",
        # Disasters (strategic)
        r"\b(earthquake|tsunami|hurricane|cyclone|flood|wildfire|volcano)\b",
    ]
]


def is_geopolitically_relevant(text: str) -> bool:
    """
    Determine if article text is relevant to geopolitical risk analysis.

    Returns True if article should be KEPT, False if it should be REJECTED.

    Algorithm:
      1. If text matches any ACCEPT pattern → KEEP (priority)
      2. If text matches any REJECT pattern → REJECT
      3. If no pattern matches → KEEP (benefit of the doubt for ambiguous)
    """
    if not text or len(text) < 20:
        return False

    # Accept patterns have priority — geopolitical content always passes
    for pattern in _ACCEPT_PATTERNS:
        if pattern.search(text):
            return True

    # Reject patterns filter noise
    for pattern in _REJECT_PATTERNS:
        if pattern.search(text):
            logger.debug("Rejected non-geopolitical: %.60s...", text)
            return False

    # Ambiguous — keep for ML classification to decide
    return True


def is_ml_relevant(label: str, confidence: float) -> bool:
    """
    Post-ML check: reject events classified as "safe" with high confidence.
    Called AFTER ML classification in the enrichment pipeline.
    """
    if label == "safe" and confidence >= 0.70:
        return False
    return True
