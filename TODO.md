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

## Backlog — future improvements

- [ ] Embeddings: routing + metrics + guard for missing `model`
- [ ] `stop` command: real process control (PID file / signal)
- [ ] Inbound API-key auth on the gateway (`api_key_header`)
- [ ] Token usage accounting for autoscaled responses (currently 0/0/0)
- [ ] Autoscale on free tier: sequential option / concurrency cap to avoid timeouts
- [ ] Replace brittle keyword classifier with weighted / embedding-based routing
- [ ] Unify logging (gateway stdlib `logging` vs router `structlog`)
- [ ] Regenerate `.spec/code-map.json` & `code-trace.json` for new modules
- [ ] Streaming path: token accounting + autoscale support
- [ ] `/v1/models`: return the router's registry, not just the upstream catalog
