# NVIDIA-SmartRoute-CLI

[![CI](https://github.com/joeldg/nvidiarouter/actions/workflows/ci.yml/badge.svg)](https://github.com/joeldg/nvidiarouter/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.12-blue)](https://github.com/joeldg/nvidiarouter)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

An OpenAI-compatible API gateway for [build.nvidia.com](https://build.nvidia.com)
(NVIDIA NIM) models, with intelligent request routing, a dynamic agent autoscale
engine, multi-key rotation for throughput, and a rich terminal dashboard.

## Features

- **OpenAI-compatible gateway** on `0.0.0.0:9000` — `/v1/chat/completions`,
  `/v1/embeddings`, `/v1/models` (streaming supported).
- **Intelligent routing** — a weighted, word-boundary classifier detects the
  task (code, maths, vision, reasoning, translation, summarization, …) and picks
  the best-suited NIM model, with latency-aware scoring and a confidence signal.
- **Adaptive routing** (optional) — an epsilon-greedy bandit that learns the
  best model per task from real outcomes and shifts traffic over time.
- **Cost intelligence** — per-request cost tracking, a daily budget guardrail,
  and optional cost-aware routing.
- **Vision** — remote image URLs are auto-fetched and inlined as base64 (NVIDIA's
  vision NIM requires inline images).
- **Agent autoscale engine** — complex multi-step code requests are fanned out to
  writer / tester / reviewer sub-agents (each a real NIM call) and composed.
- **Multi-key rotation** — pool several API keys to scale past the ~40 req/min
  per-key free-tier cap, with per-key budgeting and 429 failover.
- **Response caching** — identical non-streaming requests are served from an
  in-memory TTL+LRU cache, skipping the upstream call (big latency/budget win).
- **Model fallback chains** — if a model fails (404/5xx/timeout), the request
  fails over to the next-best model for the task.
- **Circuit breaker** — a repeatedly-failing model is taken out of rotation,
  then probed back in after a cooldown.
- **Backpressure** — a concurrency gate bounds simultaneous upstream requests
  and queues bursts, shedding load (503) when the queue is full.
- **Persistent metrics** — counters optionally survive restarts.
- **Tool / function calling** — `tools`/`tool_choice` pass through and
  `tool_calls` responses are returned unchanged.
- **Resilience** — inbound rate limiting, upstream retry/backoff, configurable
  timeouts, optional inbound API-key auth.
- **Dashboards** — a rich terminal TUI (with a requests/sec sparkline) *and* a
  browser dashboard at `/dashboard` with a live chart, model table, routing log,
  and a **prompt playground** that explains *why* each request routed where it did.
- **Observable** — structured logging (structlog), `/metrics`, health/readiness.

## Installation

```bash
pip install -e ".[dev]"      # from a clone
cp .env.example .env         # then add your NVIDIA API key(s)
```

Get a free API key at <https://build.nvidia.com>. Free models are rate-limited to
~40 requests/minute per key.

## CLI

```bash
nvidia-smartroute start          # start the gateway on 0.0.0.0:9000
nvidia-smartroute dashboard      # launch the TUI (auto-starts the gateway if down)
nvidia-smartroute status         # check whether the gateway is running
nvidia-smartroute config         # print the effective configuration (secrets redacted)
nvidia-smartroute doctor         # diagnose config, connectivity, model availability
nvidia-smartroute discover       # find which NIM models your account can serve
nvidia-smartroute benchmark      # rank registered models by speed/success/throughput
nvidia-smartroute stop           # stop the running gateway (via its PID file)
nvidia-smartroute version
```

`dashboard` starts the gateway automatically if it isn't running and stops it on
exit; pass `--no-start-gateway` to disable that.

Prefer a browser? With the gateway running, open **`http://localhost:9000/dashboard`**
for a live web view (charts, model table, routing log) plus a **playground** that
shows the routing decision — task scores, confidence, selected model, cache/
fallback, latency, tokens, and cost — for any prompt you type.

### Discover and evaluate models

The router only routes to models it knows about. Out of the box it ships a small
built-in set; discover the full set your account can serve and let the router use
them:

```bash
nvidia-smartroute discover              # probe the catalog, save servable models
# (restart the gateway to pick them up)
nvidia-smartroute benchmark             # leaderboard: params, latency, tok/s per model
```

`discover` fetches the NIM catalog, probes each model for servability (a real
1-token request), enriches it with a capability profile (parameter count, tasks,
vision/function-calling), and writes `discovered_models.json`. The router loads
that on top of the built-in defaults. Use `--limit N` to sample, or `--no-probe`
to enrich the whole catalog without servability checks.

`benchmark` is standalone (no gateway needed): it reads the registry, calls the
top-N largest models directly, and ranks them by success, p50 latency, and
generation tok/s — throttled to respect your rate limit. Use `--top N` and
`--per-model K`. Add **`--save`** to write the measured latency/throughput back
into the model profiles so the router prefers benchmarked-fast models (models
also get a size-based latency prior at discovery time, refined by live traffic).
Example output:

```
Model leaderboard (fastest first)
  nvidia/nemotron-3-super-120b   120B   683ms   70.3 tok/s
  mistralai/mistral-large-3-675b 675B   740ms   64.8 tok/s
  z-ai/glm-5.2                   355B  1420ms   32.7 tok/s
  moonshotai/kimi-k2.6          1000B  1255ms   31.5 tok/s
Fastest reliable model: nvidia/nemotron-3-super-120b (120B, 70.3 tok/s)
```

### Watch it under load

Run the dashboard in one terminal and drive traffic in another to watch routing,
model selection, cache hits, backpressure, and cost update live:

```bash
# terminal 1 — dashboard (also starts the gateway)
nvidia-smartroute dashboard

# terminal 2 — load generator
nvidia-smartroute stress -n 200 -c 20            # 200 requests, 20 concurrent
nvidia-smartroute stress -n 500 -c 50 --rps 30   # throttled to 30 req/s
```

`stress` uses a varied prompt mix (maths, code, chat, translation, …) so
different tasks and models light up, repeats some prompts to exercise the cache,
and prints a latency/throughput/routing summary at the end.

## API

```bash
# Chat (routed automatically to the best model for the task)
curl http://localhost:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17 * 3?"}]}'

# Streaming
curl -N http://localhost:9000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hi"}],"stream":true}'

# Vision (a remote image URL is auto-inlined as base64)
curl http://localhost:9000/v1/chat/completions -d '{"messages":[{"role":"user",
  "content":[{"type":"text","text":"What is this?"},
  {"type":"image_url","image_url":{"url":"https://example.com/cat.jpg"}}]}]}'

# Embeddings
curl http://localhost:9000/v1/embeddings -d '{"input":["hello world"]}'
```

Response headers expose routing decisions: `X-Selected-Model`, `X-Task-Type`,
`X-Routing-Confidence`, and (for autoscaled requests) `X-Autoscaled` /
`X-Agent-Count`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Routed chat (streaming + autoscale) |
| `POST /v1/embeddings` | Embeddings (defaults to `nv-embedqa-e5-v5`) |
| `GET /v1/models` | Router registry (`?source=upstream` for NVIDIA's catalog) |
| `GET /metrics` | Live metrics, routing stats, per-key budgets (JSON) |
| `GET /metrics/prometheus` | Prometheus text exposition for scraping |
| `GET /dashboard` | Web dashboard + prompt playground |
| `GET /health`, `GET /ready` | Liveness / readiness |

## Configuration

All settings are environment variables (see [.env.example](./.env.example)):

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | – | Your NIM API key |
| `NVIDIA_API_KEYS` | – | Comma-separated key pool for rotation |
| `PORT` | `9000` | Gateway port |
| `REQUEST_TIMEOUT` | `120` | Upstream read timeout (s); NIM cold-starts are slow |
| `RATE_LIMIT_PER_KEY` | `40` | Per-key outbound budget (NIM free-tier cap) |
| `ENABLE_RATE_LIMIT` / `RATE_LIMIT_REQUESTS` | `True` / `100` | Inbound per-client limit on `/v1/*` |
| `UPSTREAM_MAX_RETRIES` | `3` | Retries on 429/5xx with backoff |
| `ENABLE_CACHE` / `MODEL_CACHE_TTL` | `True` / `300` | Response cache + TTL (s) |
| `ENABLE_MODEL_FALLBACK` / `MAX_MODEL_FALLBACKS` | `True` / `2` | Fail over to next-best model |
| `CIRCUIT_BREAKER_ENABLED` / `CIRCUIT_FAILURE_THRESHOLD` | `True` / `3` | Take failing models out of rotation |
| `MAX_INFLIGHT_REQUESTS` / `MAX_QUEUED_REQUESTS` | `32` / `64` | Concurrency gate + queue depth |
| `PERSIST_METRICS` / `METRICS_FILE` | `False` / … | Persist metrics across restarts |
| `ROUTING_STRATEGY` / `BANDIT_EPSILON` | `static` / `0.1` | `adaptive` = learning bandit |
| `DAILY_BUDGET_USD` / `COST_WEIGHT` | `0` / `0` | Daily spend cap / cost-aware routing |
| `REQUIRE_API_KEY` / `GATEWAY_API_KEYS` | `False` / – | Optional inbound client auth |
| `AUTOSCALE_SEQUENTIAL` | `True` | Run sub-agents one at a time (free-tier safe) |
| `DEFAULT_EMBEDDING_MODEL` | `nvidia/nv-embedqa-e5-v5` | Embeddings model |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `False` | Logging level / JSON output |

### Scaling throughput

NIM free models cap at ~40 requests/minute per key. Add a pool to multiply that:

```bash
NVIDIA_API_KEYS=nvapi-key1,nvapi-key2,nvapi-key3   # ~120 req/min across 3 keys
```

The gateway picks the key with the most remaining budget and fails over to
another key on a 429.

## Docker

```bash
docker build -t nvidia-smartroute .
docker run -p 9000:9000 -e NVIDIA_API_KEY=nvapi-... nvidia-smartroute
```

### Full observability stack (Prometheus + Grafana)

```bash
NVIDIA_API_KEY=nvapi-... docker compose up --build
```

- Gateway → <http://localhost:9000> (web dashboard at `/dashboard`)
- Prometheus → <http://localhost:9090> (scrapes `/metrics/prometheus`)
- Grafana → <http://localhost:3000> (anonymous access; a "NVIDIA SmartRoute"
  dashboard is auto-provisioned)

The gateway exposes Prometheus-format metrics at `GET /metrics/prometheus`
(`nsr_total_requests`, per-model latency/requests/errors/cost, cache, budget,
backpressure) alongside the richer JSON `GET /metrics` used by the dashboards.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## Governance

Governed by SpecRegistry. The application is covered by six project-scoped specs
in [`specs/`](./specs) — `GATEWAY_API`, `ROUTING`, `MODEL_DISCOVERY`,
`SECURITY_AND_KEYS`, `OBSERVABILITY`, `COST` — plus the global SDD spec set. Code
entities carry `@spec[FILE#section]` annotations tracing to the governing
sections. Latest `specreg comply`: **PASS (97% coverage, 3% drift)**. See
[SPECREGISTRY.md](./SPECREGISTRY.md) and [AGENTS.md](./AGENTS.md); outstanding
work is in [TODO.md](./TODO.md).
