# @spec[COST.md#Requirements]
"""
Cost accounting and daily budget guardrail.

Computes the USD cost of a request from token usage and per-model pricing, and
tracks spend against an optional daily budget. When the budget is exhausted the
gateway can shed requests (503) until the day rolls over.
"""

import threading
import time
from typing import Any, Dict, Optional


# @spec[COST.md#Requirements]
def compute_cost(model: Any, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost for a request given a model's per-1k pricing."""
    in_rate = getattr(model, "input_cost_per_1k", 0.0) or 0.0
    out_rate = getattr(model, "output_cost_per_1k", 0.0) or 0.0
    return (prompt_tokens / 1000.0) * in_rate + (completion_tokens / 1000.0) * out_rate


# @spec[COST.md#Requirements]
class BudgetTracker:
    """Tracks spend within a rolling 24h window against a daily cap."""

    def __init__(self, daily_budget_usd: float = 0.0, window_seconds: int = 86400):
        self.daily_budget_usd = daily_budget_usd
        self._window = window_seconds
        self._spend = 0.0
        self._window_start = time.time()
        self._lock = threading.Lock()

    def _roll(self, now: float) -> None:
        if now - self._window_start >= self._window:
            self._spend = 0.0
            self._window_start = now

    # @spec[COST.md#Requirements]
    def allow(self) -> bool:
        """Whether spend is still under the daily budget (unlimited if <= 0)."""
        if self.daily_budget_usd <= 0:
            return True
        with self._lock:
            self._roll(time.time())
            return self._spend < self.daily_budget_usd

    # @spec[COST.md#Requirements]
    def record(self, usd: float) -> None:
        with self._lock:
            self._roll(time.time())
            self._spend += max(0.0, usd)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._roll(time.time())
            remaining: Optional[float] = (
                round(max(0.0, self.daily_budget_usd - self._spend), 6)
                if self.daily_budget_usd > 0
                else None
            )
            return {
                "spend_usd": round(self._spend, 6),
                "daily_budget_usd": self.daily_budget_usd or None,
                "remaining_usd": remaining,
                "window_resets_in_s": round(
                    max(0.0, self._window - (time.time() - self._window_start)), 1
                ),
            }

    def reset(self) -> None:
        with self._lock:
            self._spend = 0.0
            self._window_start = time.time()


# @spec[COST.md#Requirements]
def build_default_budget() -> BudgetTracker:
    from .config import settings

    return BudgetTracker(daily_budget_usd=settings.daily_budget_usd)


# Process-wide budget tracker.
budget = build_default_budget()
