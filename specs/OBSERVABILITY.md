# Observability

## Scope
The gateway's runtime telemetry: the in-process metrics tracker, the JSON
`/metrics` snapshot, the Prometheus `/metrics/prometheus` exposition, structured
logging, optional metrics persistence, the routing log, and the TUI + web
dashboards. Project-scoped; narrows global `TRACEABILITY_AND_OBSERVABILITY.md`
for this repo (application telemetry, not SDD telemetry).

## Intent
Operators must be able to see, live, how the gateway is behaving — throughput,
per-model latency/tokens/cost/errors, cache, backpressure, budget, and circuit
state — through both a scrapable endpoint and human dashboards.

## Requirements
1. The metrics tracker MUST record, per model, request count, error count,
   tokens, cost, rolling average latency, and peak throughput; and globally,
   total requests, active connections, and uptime.
2. `total_requests` MUST count only real API traffic (`/v1/*`); health/metrics
   polling MUST NOT inflate it.
3. `GET /metrics` MUST return a JSON snapshot combining model metrics with
   routing stats, key-pool budgets (masked), cache, circuits, concurrency,
   budget, and adaptive-routing state.
4. `GET /metrics/prometheus` MUST expose the same signals in Prometheus text
   exposition format (global counters/gauges and per-model series labelled by
   model). (Satisfies TRACEABILITY_AND_OBSERVABILITY req.6.)
5. Logging MUST be structured (structlog) and consistent across the router,
   agent, and gateway layers; secrets MUST NOT be logged.
6. When enabled, metrics MUST persist to disk on a timer and on shutdown and be
   restored on startup.
7. A rolling routing log MUST record recent decisions (task, model, confidence)
   for the dashboards; the TUI MUST show a requests/sec chart and per-model
   table; the web dashboard MUST render live tiles, chart, model table, and the
   routing log.

## Non-Goals
This spec does not define metric-driven model selection (`ROUTING.md`) or cost
computation (`COST.md`). It does not require developer-behavior surveillance.

## Acceptance Evidence
- `tests/test_features.py` covers the metrics tracker, prometheus exposition
  and endpoint, the request-count fix, and metrics persistence round-trip.
- `docker compose` provisions Prometheus scraping `/metrics/prometheus` and a
  Grafana dashboard.

## Token Budget Class
Project contract. Load for metrics, logging, or dashboard work.

## Related Specs
- `TRACEABILITY_AND_OBSERVABILITY.md`
- `GATEWAY_API.md`
- `COST.md`

## AI Agent Directives
Expose new signals in both `/metrics` and the Prometheus endpoint. Never log
secrets. Keep the JSON and Prometheus views consistent.
