"""
ingestion/feed_health.py
------------------------
RSS Feed Health Monitor (Log15)

Tracks feed health with:
  - Consecutive failure counting
  - Dead feed suppression after N failures
  - Retry tracking with exponential backoff
  - Per-feed health status reporting

Prevents wasting time on permanently broken feeds
(e.g., Reuters DNS failure, AP 403, Maritime Executive 404).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# After this many consecutive failures, suppress the feed
MAX_CONSECUTIVE_FAILURES = 5

# How long to suppress a dead feed before retrying (seconds)
SUPPRESSION_DURATION = 3600  # 1 hour

# After this many suppressions, extend suppression (backoff)
LONG_SUPPRESSION_THRESHOLD = 3
LONG_SUPPRESSION_DURATION = 14400  # 4 hours


@dataclass
class FeedHealthState:
    """Health state for a single RSS feed."""
    name: str
    url: str
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    suppression_count: int = 0
    suppressed_until: float = 0.0
    last_failure_at: float = 0.0
    last_success_at: float = 0.0
    last_error: str = ""
    last_status_code: Optional[int] = None

    def record_success(self, items_count: int = 0) -> None:
        """Record successful fetch."""
        self.consecutive_failures = 0
        self.total_successes += 1
        self.last_success_at = time.time()
        self.last_error = ""
        self.last_status_code = 200

    def record_failure(self, error: str, status_code: Optional[int] = None) -> None:
        """Record a fetch failure."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure_at = time.time()
        self.last_error = str(error)[:200]
        self.last_status_code = status_code

        # Check if we should suppress
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self._suppress()

    def _suppress(self) -> None:
        """Suppress the feed for a duration."""
        self.suppression_count += 1
        if self.suppression_count >= LONG_SUPPRESSION_THRESHOLD:
            duration = LONG_SUPPRESSION_DURATION
        else:
            duration = SUPPRESSION_DURATION

        self.suppressed_until = time.time() + duration
        logger.warning(
            "[FeedHealth] Suppressing '%s' for %ds after %d consecutive failures "
            "(last error: %s, status: %s)",
            self.name, duration, self.consecutive_failures,
            self.last_error[:80], self.last_status_code,
        )

    @property
    def is_suppressed(self) -> bool:
        """Whether the feed is currently suppressed."""
        return time.time() < self.suppressed_until

    @property
    def is_healthy(self) -> bool:
        """Whether the feed is considered healthy."""
        return self.consecutive_failures < MAX_CONSECUTIVE_FAILURES and not self.is_suppressed

    @property
    def suppression_remaining(self) -> float:
        """Seconds remaining in suppression."""
        return max(0, self.suppressed_until - time.time())

    @property
    def success_rate(self) -> float:
        """Historical success rate."""
        total = self.total_successes + self.total_failures
        if total == 0:
            return 1.0
        return self.total_successes / total

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "healthy": self.is_healthy,
            "suppressed": self.is_suppressed,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "success_rate": round(self.success_rate, 3),
            "suppression_count": self.suppression_count,
            "suppression_remaining_s": round(self.suppression_remaining, 0),
            "last_error": self.last_error,
            "last_status_code": self.last_status_code,
        }


# ---------------------------------------------------------------------------
# Global feed health tracker
# ---------------------------------------------------------------------------

_feed_states: dict[str, FeedHealthState] = {}


def get_feed_health(name: str, url: str = "") -> FeedHealthState:
    """Get or create a feed health state."""
    if name not in _feed_states:
        _feed_states[name] = FeedHealthState(name=name, url=url)
    return _feed_states[name]


def should_fetch_feed(name: str, url: str = "") -> bool:
    """
    Check if a feed should be fetched.
    Returns False if the feed is suppressed due to repeated failures.
    """
    state = get_feed_health(name, url)
    if state.is_suppressed:
        logger.debug(
            "[FeedHealth] '%s' suppressed (%.0fs remaining, %d failures)",
            name, state.suppression_remaining, state.consecutive_failures,
        )
        return False
    return True


def get_all_feed_health() -> dict:
    """Get health status for all tracked feeds."""
    return {name: state.to_dict() for name, state in _feed_states.items()}
