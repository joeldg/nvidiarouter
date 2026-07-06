# @spec[PROJECT_PROFILE.md#Requirements]
"""
Render the gateway metrics snapshot as Prometheus text exposition format.

Kept separate from the JSON ``/metrics`` payload (which the TUI/web dashboards
consume) and served at ``/metrics/prometheus`` for Prometheus scraping.
"""

from typing import Any, Dict, List


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _metric(lines: List[str], name: str, mtype: str, help_text: str, value) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {mtype}")
    lines.append(f"{name} {value}")


# @spec[PROJECT_PROFILE.md#Requirements]
def render_prometheus(snapshot: Dict[str, Any]) -> str:
    """Convert a /metrics snapshot dict into Prometheus exposition text."""
    lines: List[str] = []

    _metric(lines, "nsr_total_requests", "counter",
            "Total /v1 API requests", int(snapshot.get("total_requests", 0)))
    _metric(lines, "nsr_active_connections", "gauge",
            "Active in-flight connections", int(snapshot.get("active_connections", 0)))
    _metric(lines, "nsr_uptime_seconds", "gauge",
            "Gateway uptime in seconds", float(snapshot.get("uptime_seconds", 0)))
    _metric(lines, "nsr_total_cost_usd", "gauge",
            "Total spend in USD", float(snapshot.get("total_cost_usd", 0.0)))

    cache = snapshot.get("cache") or {}
    _metric(lines, "nsr_cache_hits", "counter", "Response cache hits",
            int(cache.get("hits", 0)))
    _metric(lines, "nsr_cache_misses", "counter", "Response cache misses",
            int(cache.get("misses", 0)))

    conc = snapshot.get("concurrency") or {}
    _metric(lines, "nsr_inflight_requests", "gauge", "Upstream requests in flight",
            int(conc.get("inflight", 0)))
    _metric(lines, "nsr_queued_requests", "gauge", "Requests waiting for a slot",
            int(conc.get("queued", 0)))
    _metric(lines, "nsr_rejected_requests", "counter", "Requests shed by backpressure",
            int(conc.get("rejected", 0)))

    budget = snapshot.get("budget") or {}
    _metric(lines, "nsr_budget_spend_usd", "gauge", "Spend in the current budget window",
            float(budget.get("spend_usd", 0.0)))

    # Per-model gauges/counters (labelled by model).
    models = snapshot.get("models") or []
    for family, mtype, help_text, key in [
        ("nsr_model_requests", "counter", "Requests per model", "request_count"),
        ("nsr_model_errors", "counter", "Errors per model", "error_count"),
        ("nsr_model_tokens", "counter", "Tokens per model", "total_tokens"),
        ("nsr_model_avg_latency_ms", "gauge", "Average latency (ms) per model", "avg_latency_ms"),
        ("nsr_model_max_tps", "gauge", "Peak tokens/sec per model", "max_tps"),
        ("nsr_model_cost_usd", "gauge", "Cost (USD) per model", "total_cost_usd"),
    ]:
        lines.append(f"# HELP {family} {help_text}")
        lines.append(f"# TYPE {family} {mtype}")
        for m in models:
            model = _escape(str(m.get("model_id", "unknown")))
            lines.append(f'{family}{{model="{model}"}} {m.get(key, 0)}')

    return "\n".join(lines) + "\n"
