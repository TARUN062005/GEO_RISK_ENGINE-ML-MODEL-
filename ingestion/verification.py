"""
ingestion/verification.py
-------------------------
Source Verification Layer (Log5)

Every ingested event must carry verified metadata:
  - source_url, publisher, credibility_score, image_url, timestamps

Credibility scoring uses:
  1. Trusted source whitelist (Tier 1 = 0.95, Tier 2 = 0.80, Tier 3 = 0.65)
  2. Domain reputation heuristics (age, TLD, known news domains)
  3. Penalty for unknown/unverifiable domains (base = 0.30)
  4. OPTIONAL: integration point for domain reputation APIs

The verification result is stored alongside the event in MongoDB.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trusted Source Registry
# ---------------------------------------------------------------------------

# Tier 1: Major wire services and global newspapers of record
_TIER1_DOMAINS: set[str] = {
    "reuters.com", "apnews.com", "bbc.co.uk", "bbc.com",
    "aljazeera.com", "theguardian.com", "nytimes.com",
    "washingtonpost.com", "ft.com", "bloomberg.com",
    "economist.com", "npr.org", "pbs.org", "dw.com",
    "france24.com", "afp.com",
}

# Tier 2: Major regional / specialty news outlets
_TIER2_DOMAINS: set[str] = {
    "cnn.com", "cnbc.com", "abc.net.au", "cbc.ca",
    "scmp.com", "hindustantimes.com", "timesofindia.indiatimes.com",
    "ndtv.com", "dawn.com", "jpost.com", "haaretz.com",
    "rt.com", "tass.com", "xinhuanet.com", "japantimes.co.jp",
    "straitstimes.com", "channelnewsasia.com",
    "politico.com", "thehill.com", "foreignaffairs.com",
    "defensenews.com", "janes.com", "maritimeexecutive.com",
}

# Tier 3: Known news aggregators / secondary outlets
_TIER3_DOMAINS: set[str] = {
    "yahoo.com", "msn.com", "news.google.com",
    "huffpost.com", "dailymail.co.uk", "foxnews.com",
    "newsweek.com", "usatoday.com", "independent.co.uk",
    "telegraph.co.uk", "mirror.co.uk",
}

_TIER_SCORES = {
    1: 0.95,
    2: 0.80,
    3: 0.65,
}

# Base score for unknown domains
_UNKNOWN_BASE_SCORE: float = 0.30


# ---------------------------------------------------------------------------
# Verification Result
# ---------------------------------------------------------------------------

@dataclass
class SourceVerification:
    """Verification metadata for a single event source."""
    source_url: str
    publisher: str
    credibility_score: float          # [0, 1]
    credibility_tier: str             # "tier1" | "tier2" | "tier3" | "unknown"
    image_url: Optional[str]
    published_at: datetime
    retrieved_at: datetime
    domain: str                       # extracted domain

    def to_dict(self) -> dict:
        return {
            "source_url": self.source_url,
            "publisher": self.publisher,
            "credibility_score": round(self.credibility_score, 4),
            "credibility_tier": self.credibility_tier,
            "image_url": self.image_url,
            "published_at": self.published_at.isoformat() if isinstance(self.published_at, datetime) else str(self.published_at),
            "retrieved_at": self.retrieved_at.isoformat() if isinstance(self.retrieved_at, datetime) else str(self.retrieved_at),
            "domain": self.domain,
        }


# ---------------------------------------------------------------------------
# Domain Extraction
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """Extract the base domain from a URL, removing www. prefix."""
    if not url:
        return "unknown"
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        domain = domain.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Credibility Scoring
# ---------------------------------------------------------------------------

def _score_domain(domain: str) -> tuple[float, str]:
    """
    Score a domain based on the trusted source registry.

    Returns:
        (credibility_score, tier_label)
    """
    if not domain or domain == "unknown":
        return _UNKNOWN_BASE_SCORE, "unknown"

    # Check exact domain match
    if domain in _TIER1_DOMAINS:
        return _TIER_SCORES[1], "tier1"
    if domain in _TIER2_DOMAINS:
        return _TIER_SCORES[2], "tier2"
    if domain in _TIER3_DOMAINS:
        return _TIER_SCORES[3], "tier3"

    # Check if domain is a subdomain of any known domain
    for known_domain in _TIER1_DOMAINS:
        if domain.endswith(f".{known_domain}"):
            return _TIER_SCORES[1] * 0.95, "tier1"
    for known_domain in _TIER2_DOMAINS:
        if domain.endswith(f".{known_domain}"):
            return _TIER_SCORES[2] * 0.95, "tier2"

    # Heuristic bonuses for domain characteristics
    score = _UNKNOWN_BASE_SCORE

    # Government domains
    if domain.endswith(".gov") or domain.endswith(".gov.uk"):
        score = max(score, 0.75)
        return score, "tier2"

    # Educational domains
    if domain.endswith(".edu") or domain.endswith(".ac.uk"):
        score = max(score, 0.70)
        return score, "tier2"

    # International org domains
    if domain.endswith(".org") or domain.endswith(".int"):
        score = max(score, 0.50)
        return score, "tier3"

    # Known news TLDs
    if domain.endswith(".news") or "news" in domain:
        score = max(score, 0.45)
        return score, "tier3"

    return score, "unknown"


def _adjust_publisher_score(base_score: float, publisher: str) -> float:
    """
    Adjust credibility based on publisher name when domain is unknown.
    Useful for events from aggregated feeds where URL might not match publisher.
    """
    if not publisher:
        return base_score

    publisher_lower = publisher.lower()

    # Known publisher name → boost
    known_publishers = {
        "reuters": 0.95,
        "associated press": 0.95,
        "bbc news": 0.95,
        "bbc": 0.95,
        "al jazeera": 0.90,
        "the guardian": 0.90,
        "cnn": 0.80,
        "npr": 0.90,
        "france 24": 0.85,
        "deutsche welle": 0.85,
    }

    for name, score in known_publishers.items():
        if name in publisher_lower:
            return max(base_score, score)

    return base_score


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def verify_source(
    source_url: str = "",
    publisher: str = "",
    image_url: Optional[str] = None,
    published_at: Optional[datetime] = None,
) -> SourceVerification:
    """
    Verify and score a news source.

    Computes credibility score based on:
      1. Domain whitelist lookup
      2. Publisher name matching
      3. Domain heuristics (TLD, structure)

    Args:
        source_url:   Full URL of the source article
        publisher:    Publisher/outlet name
        image_url:    Image URL from the article (passed through)
        published_at: Original publication timestamp

    Returns:
        SourceVerification with all metadata + credibility score
    """
    domain = _extract_domain(source_url)
    domain_score, tier = _score_domain(domain)
    final_score = _adjust_publisher_score(domain_score, publisher)

    return SourceVerification(
        source_url=source_url,
        publisher=publisher or domain,
        credibility_score=final_score,
        credibility_tier=tier if final_score == domain_score else (
            "tier1" if final_score >= 0.90 else
            "tier2" if final_score >= 0.75 else
            "tier3" if final_score >= 0.60 else "unknown"
        ),
        image_url=image_url,
        published_at=published_at or datetime.now(timezone.utc),
        retrieved_at=datetime.now(timezone.utc),
        domain=domain,
    )


def batch_verify(events: list[dict]) -> list[dict]:
    """
    Add verification metadata to a batch of event dicts.
    Each dict is expected to have: source_url, publisher, image_url, published_at.
    Returns the same dicts with a 'verification' key added.
    """
    for ev in events:
        v = verify_source(
            source_url=ev.get("source_url", ""),
            publisher=ev.get("publisher", ""),
            image_url=ev.get("image_url"),
            published_at=ev.get("published_at"),
        )
        ev["verification"] = v.to_dict()

    return events
