# @spec[GATEWAY_API.md#Requirements]
"""
Concurrency gate with bounded queueing (backpressure).

Caps the number of simultaneous upstream requests and lets a bounded number of
additional requests wait for a slot. This smooths bursts so they don't all hit
NIM at once (avoiding a wall of 429s). When the queue is full or a request waits
too long, the gate sheds load by raising ``QueueFullError`` (surfaced as 503).
"""

import asyncio
from typing import Any, Dict, Optional


# @spec[GATEWAY_API.md#Requirements]
class QueueFullError(Exception):
    """Raised when the concurrency queue is full or the wait times out."""


# @spec[GATEWAY_API.md#Requirements]
class ConcurrencyGate:
    """Bounded-concurrency admission gate with a wait queue."""

    def __init__(self, max_inflight: int, max_queued: int, timeout: float):
        self.max_inflight = max_inflight
        self.max_queued = max_queued
        self.timeout = timeout
        self._sem: Optional[asyncio.Semaphore] = None
        self._loop = None
        self._queued = 0
        self.rejected = 0

    def _get_sem(self) -> asyncio.Semaphore:
        # Bind the semaphore to the running loop, recreating it if the loop
        # changed (e.g. across test cases) — asyncio primitives are loop-bound.
        loop = asyncio.get_running_loop()
        if self._sem is None or self._loop is not loop:
            self._sem = asyncio.Semaphore(self.max_inflight)
            self._loop = loop
        return self._sem

    async def acquire(self) -> None:
        """Reserve a slot, waiting in the bounded queue if necessary."""
        sem = self._get_sem()
        # No await between this check and the increment, so it's atomic on the
        # event loop: reject immediately when at capacity and the queue is full.
        if sem.locked() and self._queued >= self.max_queued:
            self.rejected += 1
            raise QueueFullError("concurrency queue is full")

        self._queued += 1
        try:
            await asyncio.wait_for(sem.acquire(), timeout=self.timeout)
        except asyncio.TimeoutError:
            self.rejected += 1
            raise QueueFullError("timed out waiting for a slot")
        finally:
            self._queued -= 1

    def release(self) -> None:
        if self._sem is not None:
            self._sem.release()

    @property
    def inflight(self) -> int:
        if self._sem is None:
            return 0
        # Semaphore._value is the number of free permits.
        return self.max_inflight - self._sem._value  # noqa: SLF001

    def snapshot(self) -> Dict[str, Any]:
        return {
            "inflight": self.inflight,
            "queued": self._queued,
            "max_inflight": self.max_inflight,
            "max_queued": self.max_queued,
            "rejected": self.rejected,
        }


# @spec[GATEWAY_API.md#Requirements]
def build_default_gate() -> ConcurrencyGate:
    from .config import settings

    return ConcurrencyGate(
        max_inflight=settings.max_inflight_requests,
        max_queued=settings.max_queued_requests,
        timeout=settings.queue_timeout,
    )


# Process-wide gate.
gate = build_default_gate()
