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

## Backlog — future improvements

### High priority — throughput

- [ ] **Multi-key rotation / failover to scale past 40 req/min per key.**
  NIM free models cap at ~40 requests/minute per API key. Support a pool of up
  to ~5 keys and rotate across them to raise aggregate throughput (~200/min).
  - Config: accept multiple keys, e.g. `NVIDIA_API_KEYS=key1,key2,...` and/or
    `NVIDIA_API_KEY_1..5`; keep single-key `NVIDIA_API_KEY` working.
  - Track per-key request counts in a rolling 60s window (reuse the sliding-window
    approach from the inbound rate limiter).
  - Selection: pick the key with most remaining budget (or round-robin); when a
    key is at/near its cap, skip it. If all keys are saturated, either queue the
    request or return 429 with `Retry-After` (make it configurable).
  - Failover: on upstream 429 for a key, mark it cooled-down (honor `Retry-After`)
    and retry the request on the next available key before surfacing an error.
    Tie into the existing `_post_with_retries` backoff.
  - Per-key auth is per request (rebuild the `Authorization` header), so the
    shared `http_client` can stay; the `NIMClient` needs a key provider instead
    of a fixed key.
  - Surface per-key usage/remaining-budget in `/metrics` for the TUI.
  - Never log or echo full keys (mask like `nvapi-...abc`); keep them out of git.

### Other

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
