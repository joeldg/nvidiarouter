# @spec[PARKOUR.md#Requirements]
"""Bounded in-process telemetry for PARKOUR runs."""

from collections import deque
from typing import Any, Dict


class ParkourTelemetry:
    def __init__(self):
        self.runs = self.failures = self.partial_runs = self.limit_stops = 0
        self.active_runs = self.total_nodes = self.total_calls = 0
        self.peak_concurrency = 0
        self.truncations = self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.role_tokens = {"conductor": 0, "worker": 0, "synthesizer": 0}
        self.role_cost_usd = {"conductor": 0.0, "worker": 0.0, "synthesizer": 0.0}
        self.recent = deque(maxlen=20)

    def start(self):
        self.active_runs += 1

    def finish(self, run_id: str, result, duration_ms: float):
        self.active_runs = max(0, self.active_runs - 1)
        self.runs += 1
        self.partial_runs += int(result.partial)
        self.total_nodes += len(result.scheduler.nodes)
        self.total_calls += result.scheduler.total_calls
        self.total_tokens += result.total_tokens
        self.total_cost_usd += result.total_cost_usd
        self.peak_concurrency = max(
            self.peak_concurrency, result.scheduler.peak_concurrency
        )
        conductor = result.conductor
        self.role_tokens["conductor"] += conductor.tokens if conductor else 0
        self.role_cost_usd["conductor"] += conductor.cost_usd if conductor else 0.0
        self.role_tokens["worker"] += result.scheduler.total_tokens
        self.role_cost_usd["worker"] += result.scheduler.total_cost_usd
        self.role_tokens["synthesizer"] += result.synthesis.tokens
        self.role_cost_usd["synthesizer"] += result.synthesis.cost_usd
        self.truncations += sum(
            int(node.context_truncated) for node in result.scheduler.nodes.values()
        )
        self.recent.append({
            "run_id": run_id, "outcome": "partial" if result.partial else "success",
            "duration_ms": round(duration_ms, 1),
            "nodes": len(result.scheduler.nodes),
            "calls": result.scheduler.total_calls,
            "tokens": result.total_tokens,
            "cost_usd": result.total_cost_usd,
            "nodes_summary": [
                {"id": node.node_id, "status": node.status, "model": node.model_id}
                for node in result.scheduler.nodes.values()
            ],
        })

    def fail(self, limit: bool = False):
        self.active_runs = max(0, self.active_runs - 1)
        self.runs += 1
        self.failures += 1
        self.limit_stops += int(limit)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "runs": self.runs, "failures": self.failures,
            "partial_runs": self.partial_runs, "limit_stops": self.limit_stops,
            "active_runs": self.active_runs, "total_nodes": self.total_nodes,
            "total_calls": self.total_calls, "truncations": self.truncations,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "peak_concurrency": self.peak_concurrency,
            "role_tokens": dict(self.role_tokens),
            "role_cost_usd": dict(self.role_cost_usd),
            "recent_runs": list(self.recent),
        }


parkour_telemetry = ParkourTelemetry()
