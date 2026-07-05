# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
"""
Live metrics tracking for NVIDIA-SmartRoute-CLI.

Provides a process-wide singleton that records per-model latency and
throughput, active connection counts, and a rolling routing-event log.
Consumed by:
  * the router  -> live latency feeds best-model scoring
  * the gateway -> records latency/tokens and connection gauges
  * the TUI     -> polls a JSON snapshot for the dashboard
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@dataclass
class ModelMetrics:
    """Rolling performance metrics for a single model."""

    model_id: str
    request_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    # Rolling window of the most recent latency samples (milliseconds).
    latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    last_latency_ms: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_used: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average of the rolling latency window (0.0 if no samples)."""
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def throughput_tps(self) -> float:
        """Approximate tokens-per-second based on cumulative tokens/time."""
        elapsed = max(time.time() - self.first_seen, 1e-6)
        return self.total_tokens / elapsed


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
class MetricsTracker:
    """Thread-safe, process-wide metrics registry."""

    def __init__(self, log_capacity: int = 100) -> None:
        self._lock = threading.Lock()
        self._models: Dict[str, ModelMetrics] = {}
        self._active_connections = 0
        self._total_requests = 0
        self._routing_log: Deque[Dict[str, Any]] = deque(maxlen=log_capacity)
        self._started_at = time.time()

    # -- connection gauges ------------------------------------------------
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def connection_opened(self) -> None:
        with self._lock:
            self._active_connections += 1
            self._total_requests += 1

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def connection_closed(self) -> None:
        with self._lock:
            self._active_connections = max(0, self._active_connections - 1)

    @property
    def active_connections(self) -> int:
        with self._lock:
            return self._active_connections

    # -- per-model recording ---------------------------------------------
    def _get_model(self, model_id: str) -> ModelMetrics:
        model = self._models.get(model_id)
        if model is None:
            model = ModelMetrics(model_id=model_id)
            self._models[model_id] = model
        return model

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def record_latency(self, model_id: str, latency_ms: float) -> None:
        """Record a completed request's latency for a model."""
        with self._lock:
            model = self._get_model(model_id)
            model.request_count += 1
            model.last_latency_ms = latency_ms
            model.last_used = time.time()
            model.latencies_ms.append(latency_ms)

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def record_tokens(self, model_id: str, tokens: int) -> None:
        with self._lock:
            self._get_model(model_id).total_tokens += tokens

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def record_error(self, model_id: str) -> None:
        with self._lock:
            self._get_model(model_id).error_count += 1

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def get_avg_latency_ms(self, model_id: str) -> Optional[float]:
        """Live average latency for a model, or None if never observed."""
        with self._lock:
            model = self._models.get(model_id)
            if model is None or not model.latencies_ms:
                return None
            return model.avg_latency_ms

    # -- routing log ------------------------------------------------------
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def log_routing(
        self,
        request_id: str,
        task_type: str,
        model_id: Optional[str],
        confidence: float,
    ) -> None:
        with self._lock:
            self._routing_log.append(
                {
                    "timestamp": time.time(),
                    "request_id": request_id,
                    "task_type": task_type,
                    "model": model_id or "none",
                    "confidence": round(confidence, 2),
                }
            )

    # -- snapshot ---------------------------------------------------------
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of all live metrics."""
        with self._lock:
            models = [
                {
                    "model_id": m.model_id,
                    "request_count": m.request_count,
                    "error_count": m.error_count,
                    "total_tokens": m.total_tokens,
                    "avg_latency_ms": round(m.avg_latency_ms, 2),
                    "last_latency_ms": round(m.last_latency_ms, 2),
                    "throughput_tps": round(m.throughput_tps, 2),
                    "last_used": m.last_used,
                }
                for m in self._models.values()
            ]
            return {
                "uptime_seconds": round(time.time() - self._started_at, 2),
                "active_connections": self._active_connections,
                "total_requests": self._total_requests,
                "models": models,
                "routing_log": list(self._routing_log),
            }

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def reset(self) -> None:
        """Clear all recorded metrics (primarily for tests)."""
        with self._lock:
            self._models.clear()
            self._active_connections = 0
            self._total_requests = 0
            self._routing_log.clear()
            self._started_at = time.time()


# Process-wide singleton.
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
metrics = MetricsTracker()
