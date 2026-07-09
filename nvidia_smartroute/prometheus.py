# @spec[OBSERVABILITY.md#Requirements]
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


# @spec[OBSERVABILITY.md#Requirements]
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

    parkour = snapshot.get("parkour") or {}
    for name, mtype, help_text, key in [
        ("nsr_parkour_runs", "counter", "PARKOUR runs", "runs"),
        ("nsr_parkour_failures", "counter", "PARKOUR failed runs", "failures"),
        ("nsr_parkour_partial_runs", "counter", "PARKOUR partial runs", "partial_runs"),
        ("nsr_parkour_limit_stops", "counter", "PARKOUR limit stops", "limit_stops"),
        ("nsr_parkour_active_runs", "gauge", "Active PARKOUR runs", "active_runs"),
        ("nsr_parkour_calls", "counter", "PARKOUR upstream calls", "total_calls"),
        ("nsr_parkour_tokens", "counter", "PARKOUR aggregate tokens", "total_tokens"),
        ("nsr_parkour_cost_usd", "gauge", "PARKOUR aggregate cost", "total_cost_usd"),
    ]:
        _metric(lines, name, mtype, help_text, parkour.get(key, 0))

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
