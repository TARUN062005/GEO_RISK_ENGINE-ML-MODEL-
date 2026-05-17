"""
ingestion/rate_limiter.py
-------------------------
Exponential Backoff + Rate Limit Handler (Log15)

Provides production-grade rate limiting for external API calls:
  - Exponential backoff with jitter
  - Per-source cooldown windows
  - Retry caps with graceful degradation
  - Structured logging for observability

Designed specifically for GDELT 429 handling, but usable by any source.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimitState:
    """Tracks rate limit state for a single source."""

    def __init__(
        self,
        source: str,
        base_delay: float = 2.0,
        max_delay: float = 300.0,
        max_retries: int = 5,
        jitter_factor: float = 0.5,
        cooldown_after_failures: int = 3,
        cooldown_duration: float = 600.0,  # 10 min cooldown
    ):
        self.source = source
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.jitter_factor = jitter_factor
        self.cooldown_after_failures = cooldown_after_failures
        self.cooldown_duration = cooldown_duration

        # State
        self.consecutive_429s: int = 0
        self.total_429s: int = 0
        self.last_429_at: float = 0.0
        self.cooldown_until: float = 0.0
        self.last_success_at: float = 0.0

    def compute_backoff(self, attempt: int) -> float:
        """
        Compute delay with exponential backoff + jitter.

        delay = base * 2^attempt * (1 ± jitter)
        Capped at max_delay.
        """
        delay = self.base_delay * (2 ** attempt)
        jitter = delay * self.jitter_factor * (2 * random.random() - 1)
        delay = max(0.5, min(delay + jitter, self.max_delay))
        return delay

    def record_429(self) -> float:
        """
        Record a 429 response. Returns recommended wait time.
        May trigger cooldown if too many consecutive 429s.
        """
        self.consecutive_429s += 1
        self.total_429s += 1
        self.last_429_at = time.time()

        # Trigger cooldown after repeated 429s
        if self.consecutive_429s >= self.cooldown_after_failures:
            self.cooldown_until = time.time() + self.cooldown_duration
            logger.warning(
                "[%s] Rate limit cooldown triggered after %d consecutive 429s. "
                "Cooling down for %.0fs.",
                self.source, self.consecutive_429s, self.cooldown_duration,
            )
            return self.cooldown_duration

        wait = self.compute_backoff(self.consecutive_429s - 1)
        logger.warning(
            "[%s] 429 rate limited (attempt %d/%d). Backing off %.1fs.",
            self.source, self.consecutive_429s, self.max_retries, wait,
        )
        return wait

    def record_success(self) -> None:
        """Record a successful request. Resets consecutive failure counter."""
        self.consecutive_429s = 0
        self.last_success_at = time.time()

    def record_other_error(self) -> None:
        """Record a non-429 error."""
        self.consecutive_429s += 1

    @property
    def is_in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    @property
    def cooldown_remaining(self) -> float:
        return max(0, self.cooldown_until - time.time())

    @property
    def should_retry(self) -> bool:
        """Whether we should retry (under max retries and not in cooldown)."""
        return self.consecutive_429s < self.max_retries and not self.is_in_cooldown

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "consecutive_429s": self.consecutive_429s,
            "total_429s": self.total_429s,
            "in_cooldown": self.is_in_cooldown,
            "cooldown_remaining_s": round(self.cooldown_remaining, 0),
        }


# ---------------------------------------------------------------------------
# Global rate limit states
# ---------------------------------------------------------------------------

_rate_states: dict[str, RateLimitState] = {}


def get_rate_limiter(
    source: str,
    base_delay: float = 2.0,
    max_delay: float = 300.0,
    max_retries: int = 5,
    cooldown_after_failures: int = 3,
    cooldown_duration: float = 600.0,
) -> RateLimitState:
    """Get or create a rate limiter for a source."""
    if source not in _rate_states:
        _rate_states[source] = RateLimitState(
            source=source,
            base_delay=base_delay,
            max_delay=max_delay,
            max_retries=max_retries,
            cooldown_after_failures=cooldown_after_failures,
            cooldown_duration=cooldown_duration,
        )
    return _rate_states[source]


async def fetch_with_backoff(
    source: str,
    fetch_fn,
    *args,
    max_retries: int = 5,
    base_delay: float = 2.0,
    **kwargs,
):
    """
    Execute an async fetch function with exponential backoff on 429.

    Args:
        source: Source identifier for logging/tracking.
        fetch_fn: Async callable that may raise httpx.HTTPStatusError with 429.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds for backoff.

    Returns:
        The result of fetch_fn, or None if all retries exhausted.
    """
    import httpx

    limiter = get_rate_limiter(source, base_delay=base_delay, max_retries=max_retries)

    if limiter.is_in_cooldown:
        logger.info(
            "[%s] Rate limiter in cooldown (%.0fs remaining). Skipping.",
            source, limiter.cooldown_remaining,
        )
        return None

    for attempt in range(max_retries):
        try:
            result = await fetch_fn(*args, **kwargs)
            limiter.record_success()
            return result

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                wait = limiter.record_429()

                if not limiter.should_retry:
                    logger.warning(
                        "[%s] Rate limit: max retries exhausted or in cooldown. Giving up.",
                        source,
                    )
                    return None

                # Check for Retry-After header
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass

                await asyncio.sleep(wait)
                continue

            else:
                limiter.record_other_error()
                logger.warning("[%s] HTTP %d: %s", source, exc.response.status_code, exc)
                if attempt < max_retries - 1:
                    await asyncio.sleep(limiter.compute_backoff(attempt))
                    continue
                return None

        except Exception as exc:
            limiter.record_other_error()
            logger.warning("[%s] Fetch error (attempt %d): %s", source, attempt + 1, exc)
            if attempt < max_retries - 1:
                await asyncio.sleep(limiter.compute_backoff(attempt))
                continue
            return None

    return None


def get_all_rate_limit_states() -> dict:
    """Get all rate limit states for metrics."""
    return {source: state.to_dict() for source, state in _rate_states.items()}
