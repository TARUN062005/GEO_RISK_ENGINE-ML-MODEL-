"""
ml/classifier.py
----------------
Event Classification Module (Log1, rewritten Log16)

Classifies raw news text into geopolitical event categories.

Log16: Removed ALL transformer-based classification (zero-shot, cross-encoder).
  - Replaced with expanded keyword + regex heuristic classifier
  - Covers 10+ geopolitical categories (military, sanctions, shipping, piracy,
    cyber, protests, airspace, conflict, terrorism, diplomacy, disaster)
  - Uses weighted keyword scoring with confidence heuristics
  - Zero ML model dependencies — pure CPU regex
  - Memory: <1MB (vs ~300-500MB for transformer pipeline)

Categories: conflict, protest, sanction, disaster, terrorism, safe,
            military, shipping, piracy, cyber, airspace, diplomacy
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    label: str
    confidence: float
    scores: dict[str, float]
    method: str  # "heuristic"


# ---------------------------------------------------------------------------
# Log16: Expanded geopolitical keyword taxonomy
# ---------------------------------------------------------------------------

# Each entry: (label, [(keyword, weight), ...])
# Higher weight = stronger signal for that category
_KEYWORD_RULES: list[tuple[str, list[tuple[str, float]]]] = [
    ("military", [
        ("military", 1.0), ("troops", 1.0), ("army", 0.9), ("navy", 0.9),
        ("air force", 1.0), ("defense minister", 1.0), ("pentagon", 0.9),
        ("deployment", 0.8), ("military exercise", 1.2), ("warship", 1.0),
        ("fighter jet", 1.0), ("battalion", 1.0), ("regiment", 0.9),
        ("garrison", 0.8), ("military base", 1.0), ("conscription", 0.9),
        ("mobilization", 1.0), ("armed forces", 1.0), ("defense budget", 0.7),
        ("weapons system", 0.9), ("arms deal", 0.9), ("military aid", 0.9),
    ]),
    ("conflict", [
        ("war", 1.2), ("airstrike", 1.2), ("shelling", 1.1), ("invasion", 1.2),
        ("battle", 1.0), ("missile", 1.1), ("drone strike", 1.2),
        ("offensive", 0.9), ("combat", 1.0), ("casualt", 1.0),
        ("killed", 0.8), ("wounded", 0.8), ("civilian", 0.7),
        ("bombardment", 1.1), ("siege", 1.0), ("escalation", 0.9),
        ("frontline", 1.0), ("ceasefire violation", 1.1), ("artillery", 1.0),
        ("crossfire", 1.0), ("armed clash", 1.1), ("ground offensive", 1.1),
    ]),
    ("terrorism", [
        ("bomb", 0.9), ("explosion", 0.8), ("terror", 1.2), ("suicide bomber", 1.3),
        ("ied", 1.2), ("hostage", 1.0), ("extremis", 1.0), ("militant", 0.9),
        ("insurgent", 1.0), ("radicali", 0.9), ("jihad", 1.1),
        ("car bomb", 1.2), ("mass shooting", 1.1), ("terrorist attack", 1.3),
        ("lone wolf", 1.0), ("isis", 1.2), ("al-qaeda", 1.2), ("boko haram", 1.2),
        ("al-shabaab", 1.2), ("taliban", 1.0), ("hamas", 1.0), ("hezbollah", 1.0),
    ]),
    ("sanctions", [
        ("sanction", 1.2), ("embargo", 1.1), ("tariff", 0.9), ("trade ban", 1.1),
        ("restriction", 0.7), ("blacklist", 1.0), ("export control", 1.0),
        ("asset freeze", 1.1), ("financial penalty", 0.9), ("trade war", 1.0),
        ("economic pressure", 0.8), ("ofac", 1.0), ("treasury department", 0.7),
        ("economic sanction", 1.2), ("import ban", 1.0), ("trade restriction", 1.0),
    ]),
    ("shipping", [
        ("shipping", 0.8), ("cargo", 0.7), ("vessel", 0.8), ("port", 0.6),
        ("shipping lane", 1.1), ("maritime trade", 1.0), ("freight", 0.7),
        ("container ship", 0.9), ("tanker", 0.8), ("supply chain", 0.8),
        ("port closure", 1.1), ("canal", 0.7), ("strait of hormuz", 1.2),
        ("red sea", 0.9), ("suez canal", 1.0), ("bab el-mandeb", 1.1),
        ("malacca strait", 1.0), ("shipping disruption", 1.2),
        ("maritime security", 1.0), ("chokepoint", 1.1),
        ("port congestion", 0.9), ("logistics disruption", 1.0),
    ]),
    ("piracy", [
        ("piracy", 1.3), ("pirate", 1.2), ("hijack", 1.1), ("maritime attack", 1.2),
        ("ship seized", 1.2), ("vessel attack", 1.1), ("boarding", 0.8),
        ("ransom", 0.9), ("gulf of aden", 1.0), ("somali coast", 1.0),
        ("armed robbery at sea", 1.3), ("ship hijack", 1.3),
        ("maritime robbery", 1.2), ("pirate attack", 1.3),
        ("houthi", 1.0), ("ship attack", 1.1),
    ]),
    ("cyber", [
        ("cyber attack", 1.3), ("cyberattack", 1.3), ("hacking", 1.0),
        ("data breach", 0.9), ("ransomware", 1.1), ("malware", 1.0),
        ("cyber warfare", 1.2), ("ddos", 1.0), ("cyber espionage", 1.2),
        ("state-sponsored hack", 1.3), ("critical infrastructure", 0.7),
        ("cyber threat", 1.0), ("phishing", 0.7), ("zero-day", 0.9),
        ("cyber operation", 1.1), ("information warfare", 1.0),
    ]),
    ("protest", [
        ("protest", 1.0), ("demonstration", 0.9), ("riot", 1.0),
        ("march", 0.6), ("strike", 0.7), ("rally", 0.7), ("unrest", 0.9),
        ("clashes", 0.9), ("uprising", 1.1), ("revolution", 1.0),
        ("civil unrest", 1.1), ("tear gas", 1.0), ("water cannon", 0.9),
        ("mass protest", 1.1), ("political unrest", 1.0),
        ("anti-government", 1.0), ("crackdown", 0.9), ("dissent", 0.8),
    ]),
    ("airspace", [
        ("airspace", 1.2), ("no-fly zone", 1.3), ("flight ban", 1.2),
        ("airspace closure", 1.3), ("flight diversion", 1.0),
        ("notam", 1.0), ("restricted airspace", 1.2),
        ("air traffic", 0.7), ("grounded flights", 1.0),
        ("aviation security", 0.9), ("flight suspension", 1.0),
        ("airspace violation", 1.2), ("flight restriction", 1.0),
    ]),
    ("diplomacy", [
        ("peace", 0.6), ("ceasefire", 0.9), ("agreement", 0.5),
        ("treaty", 0.8), ("diplomat", 0.8), ("cooperation", 0.4),
        ("summit", 0.7), ("negotiation", 0.7), ("bilateral", 0.6),
        ("ambassador", 0.7), ("foreign minister", 0.8), ("united nations", 0.7),
        ("security council", 0.8), ("diplomatic crisis", 1.0),
        ("expelled diplomat", 1.0), ("recalled ambassador", 1.0),
        ("diplomatic ties", 0.7), ("peace talks", 0.8), ("mediation", 0.6),
    ]),
    ("disaster", [
        ("earthquake", 1.1), ("flood", 0.9), ("hurricane", 1.0),
        ("cyclone", 1.0), ("tsunami", 1.2), ("wildfire", 0.9),
        ("drought", 0.8), ("volcanic eruption", 1.0), ("landslide", 0.8),
        ("natural disaster", 1.1), ("relief effort", 0.7),
        ("emergency declaration", 0.8), ("evacuation", 0.7),
        ("famine", 0.9), ("humanitarian crisis", 0.9),
    ]),
]

# Compile regex patterns for each category (done once at import)
_COMPILED_RULES: list[tuple[str, list[tuple[re.Pattern, float]]]] = []
for _label, _keywords in _KEYWORD_RULES:
    _compiled = []
    for kw, weight in _keywords:
        try:
            _compiled.append((re.compile(rf"\b{re.escape(kw)}", re.IGNORECASE), weight))
        except re.error:
            _compiled.append((re.compile(re.escape(kw), re.IGNORECASE), weight))
    _COMPILED_RULES.append((_label, _compiled))


# ---------------------------------------------------------------------------
# Heuristic classifier (Log16: primary and only method)
# ---------------------------------------------------------------------------

def _heuristic_classify(text: str) -> ClassificationResult:
    """
    Log16: Weighted keyword scoring classifier.

    For each category, sums the weights of all matching keywords.
    Normalizes to [0, 1] confidence range.
    Returns the category with highest weighted score.
    """
    lower = text.lower()
    scores: dict[str, float] = {}

    for label, patterns in _COMPILED_RULES:
        total_weight = 0.0
        match_count = 0
        for pattern, weight in patterns:
            if pattern.search(lower):
                total_weight += weight
                match_count += 1
        # Normalize: confidence based on match count + accumulated weight
        # 1 match with weight 1.0 → ~0.45 confidence
        # 2 matches → ~0.65 confidence
        # 3+ matches → 0.75-0.95 confidence
        if match_count > 0:
            # Base confidence from match count
            count_factor = min(match_count / 3.0, 1.0)  # saturates at 3 matches
            # Weight factor from accumulated keyword weights
            weight_factor = min(total_weight / 3.0, 1.0)  # saturates at 3.0 total weight
            # Combined score: average of count and weight signals
            raw = (count_factor + weight_factor) / 2.0
            scores[label] = min(0.30 + raw * 0.65, 0.95)  # range: [0.30, 0.95]
        else:
            scores[label] = 0.0

    best_label = max(scores, key=scores.__getitem__)
    best_score = scores[best_label]

    # Default to "safe" if no signal found
    if best_score == 0.0:
        best_label = "safe"
        best_score = 0.5
        scores["safe"] = 0.5

    return ClassificationResult(
        label=best_label,
        confidence=round(best_score, 4),
        scores={k: round(v, 4) for k, v in scores.items()},
        method="heuristic",
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify_event(text: str) -> ClassificationResult:
    """Main entrypoint — classify a single news snippet."""
    return _heuristic_classify(text)


def classify_batch(texts: list[str]) -> list[ClassificationResult]:
    """Log11/Log16: Batch entrypoint — classify multiple texts."""
    return [_heuristic_classify(t) for t in texts]
