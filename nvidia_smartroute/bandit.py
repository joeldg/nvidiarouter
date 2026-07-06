# @spec[ROUTING.md#Requirements]
"""
Adaptive routing via an epsilon-greedy multi-armed bandit.

Learns which model performs best for each task type from real outcomes
(success and latency), and shifts traffic toward the better performers over
time. Unseen (task, model) pairs start optimistic so they get explored, and a
configurable epsilon keeps exploring so the router adapts to change.
"""

import random
import threading
from typing import Any, Dict, List, Optional, Tuple


# @spec[ROUTING.md#Requirements]
def reward_from(success: bool, latency_ms: float) -> float:
    """Map an outcome to a reward in [0, 1]: success minus a latency penalty."""
    if not success:
        return 0.0
    latency_penalty = min(latency_ms / 2000.0, 1.0)
    return 1.0 - 0.5 * latency_penalty


# @spec[ROUTING.md#Requirements]
class AdaptiveRouter:
    """Epsilon-greedy bandit over models, keyed by task type."""

    def __init__(self, epsilon: float = 0.1, optimistic: float = 1.0):
        self._epsilon = epsilon
        self._optimistic = optimistic
        # (task, model_id) -> (count, mean_reward)
        self._stats: Dict[Tuple[str, str], Tuple[int, float]] = {}
        self._lock = threading.Lock()

    def _value(self, task: str, model_id: str) -> float:
        entry = self._stats.get((task, model_id))
        # Optimistic initialisation encourages trying unseen models.
        return entry[1] if entry else self._optimistic

    # @spec[ROUTING.md#Requirements]
    def record(self, task: str, model_id: str, reward: float) -> None:
        """Update the running mean reward for a (task, model) arm."""
        with self._lock:
            count, mean = self._stats.get((task, model_id), (0, 0.0))
            count += 1
            mean += (reward - mean) / count
            self._stats[(task, model_id)] = (count, mean)

    # @spec[ROUTING.md#Requirements]
    def select(self, task: str, candidates: List[str]) -> Optional[str]:
        """Pick a model: explore with prob epsilon, else exploit best reward."""
        if not candidates:
            return None
        if random.random() < self._epsilon:
            return random.choice(candidates)
        with self._lock:
            return max(candidates, key=lambda m: self._value(task, m))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                f"{task}:{model}": {"count": c, "mean_reward": round(r, 3)}
                for (task, model), (c, r) in self._stats.items()
            }

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()


# @spec[ROUTING.md#Requirements]
def build_default_bandit() -> AdaptiveRouter:
    from .config import settings

    return AdaptiveRouter(epsilon=settings.bandit_epsilon)


# Process-wide adaptive router.
adaptive_router = build_default_bandit()
