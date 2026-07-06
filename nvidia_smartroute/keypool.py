# @spec[SECURITY_AND_KEYS.md#Requirements]
"""
API key pool with per-key rolling-window budgeting and cooldown.

NIM free-tier models cap at ~40 requests/minute per key. Rotating across a pool
of keys raises aggregate throughput (e.g. 5 keys -> ~200 req/min). The pool
picks the key with the most remaining budget, tracks usage in a rolling window,
and supports cooling a key down after an upstream 429.
"""

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


# @spec[SECURITY_AND_KEYS.md#Requirements]
class KeyPoolExhaustedError(Exception):
    """Raised when no key has remaining budget within the allowed wait."""


def _mask(key: str) -> str:
    """Mask a key for logs/metrics, e.g. 'nvapi-abc...xyz'."""
    if len(key) <= 14:
        return "***"
    return f"{key[:9]}...{key[-3:]}"


# @spec[SECURITY_AND_KEYS.md#Requirements]
class KeyPool:
    """Thread-safe rotating pool of API keys with per-key budgets."""

    def __init__(self, keys: List[str], per_key_limit: int = 40, window: int = 60):
        self._keys: List[str] = list(keys)
        self._limit = per_key_limit
        self._window = window
        self._usage: Dict[str, Deque[float]] = {k: deque() for k in self._keys}
        self._cooldown_until: Dict[str, float] = {k: 0.0 for k in self._keys}
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._keys)

    def has_keys(self) -> bool:
        return bool(self._keys)

    def _prune(self, key: str, now: float) -> None:
        dq = self._usage[key]
        cutoff = now - self._window
        while dq and dq[0] < cutoff:
            dq.popleft()

    # @spec[SECURITY_AND_KEYS.md#Requirements]
    def acquire(self) -> Tuple[Optional[str], float]:
        """
        Reserve a request slot on the best available key.

        Returns (key, 0.0) and records the usage when a key has budget, or
        (None, wait_seconds) when every key is saturated or cooling down —
        where wait_seconds is how long until the soonest key frees up.
        """
        now = time.time()
        with self._lock:
            if not self._keys:
                return None, 0.0
            best: Optional[str] = None
            best_remaining = -1
            soonest: Optional[float] = None
            for key in self._keys:
                self._prune(key, now)
                cooldown = self._cooldown_until[key]
                if cooldown > now:
                    soonest = cooldown if soonest is None else min(soonest, cooldown)
                    continue
                remaining = self._limit - len(self._usage[key])
                if remaining > 0:
                    if remaining > best_remaining:
                        best_remaining = remaining
                        best = key
                else:
                    free_at = self._usage[key][0] + self._window
                    soonest = free_at if soonest is None else min(soonest, free_at)
            if best is not None:
                self._usage[best].append(now)
                return best, 0.0
            wait = max(0.0, soonest - now) if soonest is not None else float(self._window)
            return None, wait

    # @spec[SECURITY_AND_KEYS.md#Requirements]
    def record_cooldown(self, key: str, seconds: float) -> None:
        """Cool a key down for `seconds` (e.g. after an upstream 429)."""
        with self._lock:
            if key in self._cooldown_until:
                self._cooldown_until[key] = max(
                    self._cooldown_until[key], time.time() + max(0.0, seconds)
                )

    # @spec[SECURITY_AND_KEYS.md#Requirements]
    def snapshot(self) -> List[Dict[str, Any]]:
        """Per-key usage/budget snapshot with masked keys (for /metrics)."""
        now = time.time()
        with self._lock:
            rows = []
            for key in self._keys:
                self._prune(key, now)
                used = len(self._usage[key])
                cd = max(0.0, self._cooldown_until[key] - now)
                rows.append(
                    {
                        "key": _mask(key),
                        "used": used,
                        "limit": self._limit,
                        "remaining": max(0, self._limit - used),
                        "cooldown_remaining": round(cd, 1),
                    }
                )
            return rows

    def reset(self) -> None:
        with self._lock:
            for key in self._keys:
                self._usage[key].clear()
                self._cooldown_until[key] = 0.0


# @spec[SECURITY_AND_KEYS.md#Requirements]
def build_default_pool() -> KeyPool:
    """Construct the process-wide key pool from settings."""
    from .config import settings

    return KeyPool(
        keys=settings.api_keys,
        per_key_limit=settings.rate_limit_per_key,
        window=settings.per_key_rate_window,
    )


# Process-wide pool.
key_pool = build_default_pool()
