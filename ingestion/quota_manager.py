"""
ingestion/quota_manager.py
--------------------------
Quota-Aware Ingestion Scheduler (Log15)

Manages per-source API quotas with:
  - Rolling 24h request counters
  - Automatic cooldown when quota exhausted
  - Quota persistence to disk (survives restarts)
  - Source-specific scheduling intervals
  - Structured logging of quota state

Designed for free-tier API limits:
  - NewsAPI: max 100 requests/day (default budget: 50)
  - GNews:   max 100 requests/day (default budget: 50)
  - GDELT:   rate-limited (no hard daily cap, but 429-prone)
  - RSS:     unlimited (no API key required)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

# Daily quota budgets (conservative — well under free-tier caps)
DEFAULT_QUOTAS = {
    "newsapi": int(os.environ.get("NEWSAPI_DAILY_QUOTA", "50")),
    "gnews": int(os.environ.get("GNEWS_DAILY_QUOTA", "50")),
    "gdelt": int(os.environ.get("GDELT_DAILY_QUOTA", "500")),  # soft limit
}

# Minimum interval between fetches for each source (seconds)
SOURCE_INTERVALS = {
    "rss": int(os.environ.get("RSS_INTERVAL_SECONDS", "180")),       # 3 min
    "gdelt": int(os.environ.get("GDELT_INTERVAL_SECONDS", "900")),   # 15 min
    "newsapi": int(os.environ.get("NEWSAPI_INTERVAL_SECONDS", "1800")),  # 30 min
    "gnews": int(os.environ.get("GNEWS_INTERVAL_SECONDS", "1800")),  # 30 min
}

# Cooldown after quota exhaustion (seconds)
QUOTA_EXHAUSTION_COOLDOWN = int(os.environ.get("QUOTA_EXHAUSTION_COOLDOWN", "3600"))  # 1h

# Persistence file path
_QUOTA_STATE_DIR = os.environ.get(
    "QUOTA_STATE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".quota_state"),
)
_QUOTA_STATE_FILE = os.path.join(_QUOTA_STATE_DIR, "quota_state.json")


# ---------------------------------------------------------------------------
# Quota State
# ---------------------------------------------------------------------------

class _SourceQuotaState:
    """Tracks quota usage for a single source."""

    def __init__(self, source: str, daily_limit: int):
        self.source = source
        self.daily_limit = daily_limit
        self.requests: list[float] = []  # timestamps of requests within rolling window
        self.last_fetch_at: float = 0.0
        self.cooldown_until: float = 0.0
        self.consecutive_failures: int = 0

    def record_request(self, count: int = 1) -> None:
        """Record API request(s)."""
        now = time.time()
        for _ in range(count):
            self.requests.append(now)
        self.last_fetch_at = now
        self._prune()

    def record_failure(self) -> None:
        """Record a failure (for exponential backoff)."""
        self.consecutive_failures += 1

    def record_success(self) -> None:
        """Reset failure counter on success."""
        self.consecutive_failures = 0

    def set_cooldown(self, seconds: int) -> None:
        """Set a cooldown period."""
        self.cooldown_until = time.time() + seconds
        logger.warning(
            "[%s] Cooldown set for %ds (until %s)",
            self.source, seconds,
            datetime.fromtimestamp(self.cooldown_until, tz=timezone.utc).isoformat(),
        )

    def _prune(self) -> None:
        """Remove requests older than 24 hours."""
        cutoff = time.time() - 86400
        self.requests = [t for t in self.requests if t > cutoff]

    @property
    def used_today(self) -> int:
        """Count requests in the rolling 24h window."""
        self._prune()
        return len(self.requests)

    @property
    def remaining(self) -> int:
        """Remaining quota for rolling 24h window."""
        return max(0, self.daily_limit - self.used_today)

    @property
    def is_exhausted(self) -> bool:
        """Whether daily quota is exhausted."""
        return self.remaining <= 0

    @property
    def is_in_cooldown(self) -> bool:
        """Whether source is in cooldown."""
        return time.time() < self.cooldown_until

    @property
    def next_reset_at(self) -> Optional[datetime]:
        """When the oldest request expires from the rolling window."""
        self._prune()
        if not self.requests:
            return None
        oldest = min(self.requests)
        return datetime.fromtimestamp(oldest + 86400, tz=timezone.utc)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "daily_limit": self.daily_limit,
            "requests": self.requests,
            "last_fetch_at": self.last_fetch_at,
            "cooldown_until": self.cooldown_until,
            "consecutive_failures": self.consecutive_failures,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "_SourceQuotaState":
        state = cls(source=data["source"], daily_limit=data.get("daily_limit", 50))
        state.requests = data.get("requests", [])
        state.last_fetch_at = data.get("last_fetch_at", 0.0)
        state.cooldown_until = data.get("cooldown_until", 0.0)
        state.consecutive_failures = data.get("consecutive_failures", 0)
        state._prune()
        return state


# ---------------------------------------------------------------------------
# Quota Manager Singleton
# ---------------------------------------------------------------------------

class QuotaManager:
    """
    Thread-safe quota manager for all ingestion sources.

    Usage:
        qm = get_quota_manager()
        if qm.can_fetch("newsapi"):
            # ... do fetch ...
            qm.record_request("newsapi")
        else:
            qm.log_skip("newsapi")
    """

    _instance: Optional["QuotaManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._sources: dict[str, _SourceQuotaState] = {}
        self._global_lock = threading.Lock()
        self._initialized = False

    def initialize(self) -> None:
        """Load persisted state and set up sources."""
        if self._initialized:
            return

        # Initialize source states
        for source, limit in DEFAULT_QUOTAS.items():
            self._sources[source] = _SourceQuotaState(source, limit)

        # RSS has no quota limit
        self._sources["rss"] = _SourceQuotaState("rss", daily_limit=999999)

        # Load persisted state
        self._load_state()
        self._initialized = True
        logger.info("QuotaManager initialized: %s", self.status_summary())

    def can_fetch(self, source: str) -> bool:
        """
        Check if a source can be fetched right now.

        Considers:
          1. Quota remaining
          2. Cooldown state
          3. Minimum interval since last fetch
        """
        with self._global_lock:
            state = self._sources.get(source)
            if state is None:
                return True  # Unknown source — allow

            # Check cooldown
            if state.is_in_cooldown:
                return False

            # Check quota
            if state.is_exhausted:
                return False

            # Check minimum interval
            min_interval = SOURCE_INTERVALS.get(source, 0)
            if min_interval > 0 and state.last_fetch_at > 0:
                elapsed = time.time() - state.last_fetch_at
                if elapsed < min_interval:
                    return False

            return True

    def record_request(self, source: str, count: int = 1) -> None:
        """Record API request(s) and persist state."""
        with self._global_lock:
            state = self._sources.get(source)
            if state:
                state.record_request(count)
                state.record_success()
                logger.info(
                    "[%s] quota used: %d/%d",
                    source, state.used_today, state.daily_limit,
                )
                self._persist_state()

    def record_failure(self, source: str) -> None:
        """Record a fetch failure."""
        with self._global_lock:
            state = self._sources.get(source)
            if state:
                state.record_failure()
                # Auto-cooldown after repeated failures
                if state.consecutive_failures >= 3:
                    cooldown = min(
                        QUOTA_EXHAUSTION_COOLDOWN,
                        60 * (2 ** state.consecutive_failures),
                    )
                    state.set_cooldown(cooldown)
                self._persist_state()

    def record_exhaustion(self, source: str) -> None:
        """Record that a source's quota is exhausted."""
        with self._global_lock:
            state = self._sources.get(source)
            if state:
                state.set_cooldown(QUOTA_EXHAUSTION_COOLDOWN)
                logger.warning(
                    "[%s] quota exhausted, skipping until reset (%d/%d used)",
                    source, state.used_today, state.daily_limit,
                )
                self._persist_state()

    def log_skip(self, source: str, reason: str = "") -> None:
        """Log a source skip with quota info."""
        state = self._sources.get(source)
        if state is None:
            logger.info("[%s] Skipped: %s", source, reason or "unknown source")
            return

        if state.is_exhausted:
            next_reset = state.next_reset_at
            reset_str = next_reset.strftime("%H:%M UTC") if next_reset else "unknown"
            logger.info(
                "[%s] quota exhausted, skipping until reset at %s (%d/%d used)",
                source, reset_str, state.used_today, state.daily_limit,
            )
        elif state.is_in_cooldown:
            remaining = max(0, state.cooldown_until - time.time())
            logger.info(
                "[%s] in cooldown for %.0fs more (%d failures)",
                source, remaining, state.consecutive_failures,
            )
        else:
            min_interval = SOURCE_INTERVALS.get(source, 0)
            elapsed = time.time() - state.last_fetch_at if state.last_fetch_at > 0 else float("inf")
            logger.debug(
                "[%s] Skipped: interval not reached (%.0f/%.0fs)",
                source, elapsed, min_interval,
            )

    def get_source_status(self, source: str) -> dict:
        """Get status for a specific source."""
        state = self._sources.get(source)
        if state is None:
            return {"source": source, "status": "unknown"}

        return {
            "source": source,
            "used_today": state.used_today,
            "daily_limit": state.daily_limit,
            "remaining": state.remaining,
            "exhausted": state.is_exhausted,
            "in_cooldown": state.is_in_cooldown,
            "consecutive_failures": state.consecutive_failures,
            "last_fetch_at": (
                datetime.fromtimestamp(state.last_fetch_at, tz=timezone.utc).isoformat()
                if state.last_fetch_at > 0 else None
            ),
        }

    def status_summary(self) -> str:
        """One-line status summary for logging."""
        parts = []
        for source in ("newsapi", "gnews", "gdelt", "rss"):
            state = self._sources.get(source)
            if state:
                status = "exhausted" if state.is_exhausted else (
                    "cooldown" if state.is_in_cooldown else "ok"
                )
                parts.append(f"{source}={state.used_today}/{state.daily_limit}({status})")
        return " | ".join(parts)

    def get_all_quotas(self) -> dict:
        """Get all quota states for metrics endpoint."""
        return {
            source: self.get_source_status(source)
            for source in self._sources
        }

    # --- Persistence ---

    def _persist_state(self) -> None:
        """Save quota state to disk."""
        try:
            os.makedirs(_QUOTA_STATE_DIR, exist_ok=True)
            data = {
                source: state.to_dict()
                for source, state in self._sources.items()
            }
            with open(_QUOTA_STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.debug("Failed to persist quota state: %s", exc)

    def _load_state(self) -> None:
        """Load quota state from disk."""
        try:
            if os.path.exists(_QUOTA_STATE_FILE):
                with open(_QUOTA_STATE_FILE) as f:
                    data = json.load(f)
                for source, state_data in data.items():
                    if source in self._sources:
                        loaded = _SourceQuotaState.from_dict(state_data)
                        # Preserve configured daily limit
                        loaded.daily_limit = self._sources[source].daily_limit
                        self._sources[source] = loaded
                logger.info("Loaded persisted quota state from %s", _QUOTA_STATE_FILE)
        except Exception as exc:
            logger.debug("Failed to load quota state: %s", exc)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_manager: Optional[QuotaManager] = None


def get_quota_manager() -> QuotaManager:
    """Get or create the global QuotaManager singleton."""
    global _manager
    if _manager is None:
        _manager = QuotaManager()
        _manager.initialize()
    return _manager
