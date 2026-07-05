# @spec[PROJECT_PROFILE.md#Requirements]
"""
Per-model circuit breaker.

Repeatedly-failing models are taken out of rotation ("open") for a cooldown,
then allowed a single probe ("half-open"). A successful probe closes the
circuit; a failed probe re-opens it. This keeps traffic away from a NIM model
that is down or degraded, complementing the model fallback chain.
"""

import threading
import time
from typing import Any, Dict, List


# @spec[PROJECT_PROFILE.md#Requirements]
class CircuitBreaker:
    """Thread-safe per-model failure circuit breaker."""

    def __init__(self, failure_threshold: int = 3, reset_seconds: int = 30):
        self._threshold = failure_threshold
        self._reset = reset_seconds
        self._failures: Dict[str, int] = {}
        self._opened_at: Dict[str, float] = {}
        self._half_open: Dict[str, bool] = {}
        self._lock = threading.Lock()

    # @spec[PROJECT_PROFILE.md#Requirements]
    def allow(self, model_id: str) -> bool:
        """Whether a request may be sent to this model right now.

        Closed -> allow. Open -> deny until the cooldown elapses, then allow a
        single half-open probe.
        """
        now = time.time()
        with self._lock:
            failures = self._failures.get(model_id, 0)
            if failures < self._threshold:
                return True
            # Circuit is open; allow one probe once the cooldown has passed.
            if now - self._opened_at.get(model_id, 0.0) >= self._reset:
                self._half_open[model_id] = True
                return True
            return False

    # @spec[PROJECT_PROFILE.md#Requirements]
    def record_success(self, model_id: str) -> None:
        with self._lock:
            self._failures[model_id] = 0
            self._half_open.pop(model_id, None)
            self._opened_at.pop(model_id, None)

    # @spec[PROJECT_PROFILE.md#Requirements]
    def record_failure(self, model_id: str) -> None:
        now = time.time()
        with self._lock:
            # A failed half-open probe re-opens immediately.
            if self._half_open.pop(model_id, False):
                self._failures[model_id] = self._threshold
                self._opened_at[model_id] = now
                return
            self._failures[model_id] = self._failures.get(model_id, 0) + 1
            if self._failures[model_id] >= self._threshold:
                self._opened_at.setdefault(model_id, now)

    def state(self, model_id: str) -> str:
        with self._lock:
            failures = self._failures.get(model_id, 0)
            if failures < self._threshold:
                return "closed"
            if self._half_open.get(model_id):
                return "half_open"
            if time.time() - self._opened_at.get(model_id, 0.0) >= self._reset:
                return "half_open"
            return "open"

    def open_models(self) -> List[str]:
        return [m for m in list(self._failures) if self.state(m) == "open"]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            models = list(self._failures)
        return {m: self.state(m) for m in models if self.state(m) != "closed"}

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()
            self._opened_at.clear()
            self._half_open.clear()


# @spec[PROJECT_PROFILE.md#Requirements]
def build_default_breaker() -> CircuitBreaker:
    from .config import settings

    return CircuitBreaker(
        failure_threshold=settings.circuit_failure_threshold,
        reset_seconds=settings.circuit_reset_seconds,
    )


# Process-wide breaker.
breaker = build_default_breaker()
