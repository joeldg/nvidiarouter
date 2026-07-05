# NVIDIA-SmartRoute-CLI

An OpenAI-compatible API gateway for [build.nvidia.com](https://build.nvidia.com)
(NVIDIA NIM) models, with intelligent request routing, a dynamic agent autoscale
engine, multi-key rotation for throughput, and a rich terminal dashboard.

## Features

- **OpenAI-compatible gateway** on `0.0.0.0:9000` — `/v1/chat/completions`,
  `/v1/embeddings`, `/v1/models` (streaming supported).
- **Intelligent routing** — a weighted, word-boundary classifier detects the
  task (code, maths, vision, reasoning, translation, summarization, …) and picks
  the best-suited NIM model, with latency-aware scoring and a confidence signal.
- **Vision** — remote image URLs are auto-fetched and inlined as base64 (NVIDIA's
  vision NIM requires inline images).
- **Agent autoscale engine** — complex multi-step code requests are fanned out to
  writer / tester / reviewer sub-agents (each a real NIM call) and composed.
- **Multi-key rotation** — pool several API keys to scale past the ~40 req/min
  per-key free-tier cap, with per-key budgeting and 429 failover.
- **Resilience** — inbound rate limiting, upstream retry/backoff, configurable
  timeouts, optional inbound API-key auth.
- **Rich TUI dashboard** — live throughput, active connections, per-model
  performance, per-key budget, and routing logs. Can auto-start the gateway.
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
nvidia-smartroute stop           # stop the running gateway (via its PID file)
nvidia-smartroute version
```

`dashboard` starts the gateway automatically if it isn't running and stops it on
exit; pass `--no-start-gateway` to disable that.

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
| `GET /metrics` | Live metrics, routing stats, per-key budgets |
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

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## Governance

Developed under SpecRegistry governance. See [SPECREGISTRY.md](./SPECREGISTRY.md)
and [AGENTS.md](./AGENTS.md). Outstanding work is tracked in [TODO.md](./TODO.md).
