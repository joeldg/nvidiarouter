# NVIDIA-SmartRoute-CLI — TODO

Living checklist of what's done, what's in progress, and what's deferred.

## Done (implemented & live-verified)

- [x] Fix chat endpoint 500 (messages passed twice to router)
- [x] Vision task type + multimodal (list) content handling
- [x] Latency-aware model scoring (was returning first candidate)
- [x] Live metrics tracker + `/metrics` endpoint
- [x] Agent Autoscale Engine (writer/tester/reviewer, real NIM calls)
- [x] Textual TUI dashboard + `dashboard` command
- [x] Dashboard auto-starts the gateway if not running
- [x] Config env aliases (NVIDIA_API_KEY / NVIDIA_BASE_URL) + correct base URL
- [x] Configurable upstream read timeout (cold-start models)
- [x] Port 9000 default; verified-servable NIM model IDs
- [x] Replace garbled ASCII banner
- [x] Untrack `.env` (real key never committed)

## Done — recommended batch (2nd pass)

- [x] Remove dead `code_score` computation in the classifier
- [x] Add Dockerfile + .dockerignore (containerization)
- [x] Migrate FastAPI `on_event` startup/shutdown → `lifespan` handler
- [x] Record live metrics for the streaming path (latency)
- [x] Fix broken streaming (`await` on an async generator — never worked)
- [x] Inbound rate limiting on `/v1/*` (sliding window, 429, Retry-After)
- [x] Upstream 429/5xx retry with exponential backoff (honor Retry-After)
- [x] Auto-inline remote image URLs → base64 for vision requests
- [x] Tests for the above (26 passing, no deprecation warnings)

## Done — multi-key rotation (throughput)

- [x] **Multi-key rotation / failover to scale past 40 req/min per key.**
  - Config: `NVIDIA_API_KEYS=k1,k2,...` pool + single `NVIDIA_API_KEY`, merged
    and de-duplicated via `settings.api_keys`.
  - `KeyPool`: per-key rolling-window budget, picks the key with most remaining
    budget (spreads load evenly), cooldown on 429.
  - `NIMClient` uses the pool (per-request auth header); `_post_with_retries`
    fails over to another key on 429 and returns 503 + Retry-After when all keys
    are exhausted. Streaming path rotates keys on connect-time 429.
  - Per-key usage/remaining surfaced in `/metrics`; keys masked in logs/metrics.
  - Verified live: budget exhaustion -> [200,200,503]; failover key1->key2.
  - Follow-up: optional `NVIDIA_API_KEY_1..5` numbered vars (comma list covers it).

## Done — backlog batch (3rd pass)

- [x] `/v1/models`: return the router's registry (with `?source=upstream` for the
      NVIDIA catalog). Fixed a `models()` regression (removed `self.headers`).
- [x] Autoscale free-tier reliability: `autoscale_sequential` (default on) runs
      tester/reviewer one at a time so they don't compete on a slow model.
- [x] Embeddings hardening: default model (`nv-embedqa-e5-v5`), 400 on missing
      input, `input_type`/`truncate` defaults, live metrics, 502/503 error codes.

## Done — backlog batch (4th pass)

- [x] `stop` command: real process control via PID file + SIGTERM; `start`
      writes/cleans the PID file (configurable `PID_FILE`).
- [x] Token usage accounting for autoscaled responses (summed across sub-agents).
- [x] Unify logging on structlog (`logging_config`); gateway fields now render;
      `LOG_JSON` for JSON output.

## Done — backlog batch (5th pass)

- [x] Inbound API-key auth on `/v1/*` (`REQUIRE_API_KEY` + `GATEWAY_API_KEYS`);
      accepts `X-API-Key` header or `Authorization: Bearer`; health/metrics exempt.
- [x] Streaming token accounting: record usage from the final chunk when the
      client sets `stream_options.include_usage` (fixed cumulative-overcount bug).

## Done — classifier rewrite (6th pass)

- [x] Replace brittle keyword classifier with a weighted, word-boundary scorer:
      per-task weighted rules, code-domain boost, arithmetic + vision structural
      signals, deterministic tie-break priority, and a real confidence score
      (surfaced via `classify()` and used in `route_request`). Fixes false
      positives like "meaning"->maths and "calculate factorial"->maths.

## Done — wrap-up (7th pass)

- [x] Comprehensive README refresh (features, endpoints, config table, scaling,
      Docker, dev).

## Done — backpressure + persistent metrics (10th pass)

- [x] Concurrency gate with bounded queue (backpressure): caps simultaneous
      upstream requests, queues bursts, sheds load (503 "overloaded") when full
      or the wait times out. Loop-safe lazy semaphore. Snapshot in `/metrics`.
- [x] Persistent metrics: optional dump/load of per-model counters to disk on a
      timer + on shutdown, restored on startup (`PERSIST_METRICS`).

## Done — circuit breaker (9th pass)

- [x] Per-model circuit breaker: trip open after N consecutive hard failures,
      deny for a cooldown, then half-open probe; close on success / re-open on a
      failed probe. Integrated into the fallback chain (open models skipped) and
      surfaced in `/ready` (unhealthy_models) and `/metrics` (circuits).

## Done — throughput / reliability / capabilities (8th pass)

- [x] Response cache (TTL + LRU) — identical non-streaming requests skip the
      upstream call. Live: 6.4s -> 0.003s on hit. Stats in `/metrics`.
- [x] Model fallback chains — on 404/5xx/timeout, fail over to the next-best
      model for the task (`rank_models`); no fan-out on 4xx client errors.
- [x] Tool / function calling — `tools`/`tool_choice` forwarded, `tool_calls`
      returned unchanged (verified live: real `get_weather` tool call).

## Done — model discovery + benchmarking (11th pass)

- [x] `discover` command: fetch the NIM catalog, probe servability per account,
      enrich with capability profiles (params, tasks, vision/fc), persist to
      `discovered_models.json`; the router loads them on top of the defaults.
- [x] `model_catalog.py`: curated metadata for notable free models (Kimi 1T,
      GLM, DeepSeek-R1, Llama-405B, …) + pattern-based inference for the rest.
- [x] `benchmark` command: per-model leaderboard (params, p50 latency, tok/s)
      so you can pick the biggest/fastest models that work for your key.
- [x] `/v1/models` exposes `parameters_b`.

## Stellar — stretch ideas (make it a standout app)

Differentiators:
- [x] Cost & budget intelligence: per-request cost, spend in /metrics,
      cost-aware routing, daily budget cap (503 when exhausted).
- [x] Adaptive routing (epsilon-greedy bandit): learns best model per task from
      real outcomes (success + latency) and shifts traffic over time.
- [x] Web dashboard + prompt playground at /dashboard: live chart, model table,
      routing log, and a "why did this route here" explainer (/explain endpoint).
      TUI gains a requests/sec sparkline; fixed total_requests counting polling.
- [ ] Horizontal scale via Redis: shared cache / rate-limit / key-budget /
      circuit state so N replicas share one budget pool behind a load balancer.

Capabilities:
- [ ] RAG + semantic (near-match) cache using the embedding model already wired.
- [ ] Tool-using autoscale: real write -> run-tests -> fix loop in a sandbox.
- [ ] Multi-provider: Anthropic /v1/messages compat + provider abstraction.
- [ ] Multi-tenancy: per-client keys with quotas / rate limits / model allowlists
      + audit log (with PII redaction).

Foundation:
- [x] CI (GitHub Actions): ruff + pytest on push/PR to main (3.9/3.11/3.12).
- [x] `doctor` command: validate keys, connectivity, and which models the
      account can actually serve.
- [ ] Publish to PyPI + changelog + semver.
- [ ] docker-compose with Prometheus /metrics + Grafana dashboards.
- [ ] Load-test harness proving the 40 -> 200 rpm multi-key scaling with numbers.

## Backlog — deferred (with rationale)

- [ ] Regenerate `.spec/code-map.json` & `code-trace.json` — generated by the
      external SpecRegistry `specreg` tool against the governance server
      (`10.0.0.142:4000`); run `specreg` when that environment is available
      rather than hand-editing.
- [ ] Autoscale over the streaming path — deferred: autoscale composes a
      multi-section answer from several agents, which doesn't map cleanly onto
      token streaming; low value versus complexity.
- [ ] Embeddings query/passage routing — N/A with a single embedding model;
      `input_type` is already client-overridable. Revisit if more embed models
      are added.
- [ ] Optional embedding-based routing strategy — intentionally not default: a
      per-request embedding call adds latency and consumes key budget. Worth it
      only if max routing accuracy is preferred over throughput.
