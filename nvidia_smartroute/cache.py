# @spec[GATEWAY_API.md#Requirements]
"""
In-memory TTL + LRU response cache.

Caches identical non-streaming chat responses so repeated prompts skip the
upstream NIM call entirely — cutting latency and conserving per-key rate budget.
Keys are a hash of the canonicalised request, so only exact matches hit.
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional


# @spec[GATEWAY_API.md#Requirements]
def make_key(payload: Dict[str, Any]) -> str:
    """Stable cache key from a request payload (order-independent)."""
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# @spec[GATEWAY_API.md#Requirements]
class ResponseCache:
    """Thread-safe TTL + LRU cache with hit/miss counters."""

    def __init__(self, max_entries: int = 1000, ttl: int = 300):
        self._max = max_entries
        self._ttl = ttl
        self._store: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self.misses += 1
                return None
            expires_at, value = item
            if expires_at < now:
                # Expired — drop it.
                del self._store[key]
                self.misses += 1
                return None
            # Refresh LRU order and serve.
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            expires_at = time.time() + (ttl if ttl is not None else self._ttl)
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)
            # Evict least-recently-used entries beyond the cap.
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._store),
                "max_entries": self._max,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
            }


# @spec[GATEWAY_API.md#Requirements]
def build_default_cache() -> ResponseCache:
    from .config import settings

    return ResponseCache(max_entries=settings.cache_max_entries, ttl=settings.cache_ttl)


# Process-wide response cache.
response_cache = build_default_cache()
