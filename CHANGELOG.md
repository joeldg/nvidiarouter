# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Model discovery** (`discover`): probe the NIM catalog for models your
  account can serve, enrich each with a capability profile (parameters, tasks,
  vision/function-calling), and register them for routing. Throttled to respect
  the per-key rate limit.
- **Benchmark** (`benchmark`): standalone, throttled leaderboard of the largest
  models by latency and generation throughput; `--save` feeds measured latency
  back into routing.
- **Intelligent routing**: weighted, word-boundary task classifier with
  confidence; latency/size-aware model scoring; optional cost-aware routing;
  adaptive (epsilon-greedy bandit) routing that learns per task from traffic.
- **Throughput**: multi-key rotation with per-key budgeting and 429 failover;
  TTL+LRU response cache; concurrency gate with bounded-queue backpressure.
- **Reliability**: upstream retry/backoff, model fallback chains, per-model
  circuit breaker, persistent metrics.
- **Cost**: per-request cost accounting, daily budget guardrail.
- **Interfaces**: OpenAI-compatible `/v1/chat/completions` (+ streaming, tools),
  `/v1/embeddings`, `/v1/models`; a Textual TUI dashboard with a request
  sparkline; a web dashboard + prompt "explain" playground; a `stress` load
  generator.
- **Ops**: optional inbound API-key auth, inbound rate limiting, structured
  logging, `/metrics`, health/readiness, `start`/`stop` (PID file), `doctor`
  diagnostics, Dockerfile, CI (ruff + pytest on 3.9/3.11/3.12).

- **Observability stack**: Prometheus text exposition at `/metrics/prometheus`,
  plus a `docker compose` stack (gateway + Prometheus + Grafana with an
  auto-provisioned dashboard).
- **Packaging**: PyPI-ready metadata, MIT `LICENSE`, and a tag-triggered release
  workflow (Trusted Publishing).

### Notes
- Verified live against build.nvidia.com throughout development.

[Unreleased]: https://github.com/joeldg/nvidiarouter/commits/main
