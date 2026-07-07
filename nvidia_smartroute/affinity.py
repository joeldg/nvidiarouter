# @spec[ROUTING.md#Requirements]
"""
Session affinity store.

Maps a stable session key (X-Session-Id header or the OpenAI `user` field) to the
model id previously chosen for that conversation, so an opt-in caller keeps a
conversation on one model across turns (ROUTING.md req.9-11). Bounded by TTL and
a maximum entry count with LRU eviction, and never persisted across restarts.

This is deliberately NOT a conversation/memory store — it holds a key -> model_id
string only, entirely separate from the response cache so it does not perturb
cache hit/miss metrics.
"""

import threading
import time
from collections import OrderedDict
from typing import Optional


# @spec[ROUTING.md#Requirements]
class SessionAffinity:
    """Thread-safe TTL + LRU map of session key -> pinned model id."""

    def __init__(self, max_entries: int = 10000, ttl: int = 900):
        self._max = max_entries
        self._ttl = ttl
        self._store: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        """Return the pinned model id for `key`, or None if unset/expired."""
        if not key:
            return None
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, model_id = item
            if expires_at < now:
                del self._store[key]
                return None
            # Refresh recency so active sessions survive LRU eviction.
            self._store.move_to_end(key)
            return model_id

    def set(self, key: str, model_id: str) -> None:
        """Pin `key` to `model_id`, refreshing its TTL and recency."""
        if not key or not model_id:
            return
        with self._lock:
            expires_at = time.time() + self._ttl
            self._store[key] = (expires_at, model_id)
            self._store.move_to_end(key)
            # Evict least-recently-used entries beyond the cap.
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def reset(self) -> None:
        with self._lock:
            self._store.clear()


# @spec[ROUTING.md#Requirements]
def build_default_affinity() -> SessionAffinity:
    """Construct the process-wide affinity store from settings."""
    from .config import settings

    return SessionAffinity(
        max_entries=settings.session_affinity_max,
        ttl=settings.session_affinity_ttl,
    )


# Module-level singleton used by the router.
session_affinity = build_default_affinity()
