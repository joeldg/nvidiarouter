# Gateway API

## Scope
The OpenAI-compatible HTTP surface exposed by the NVIDIA-SmartRoute gateway:
`/v1/chat/completions` (streaming and non-streaming), `/v1/embeddings`,
`/v1/models`, `/explain`, and the operational endpoints `/health`, `/ready`,
`/metrics`, `/metrics/prometheus`, `/dashboard`. Project-scoped; narrows the
CLI/Developer-Tooling project type for `github.com/joeldg/nvidiarouter`.

## Intent
Clients that speak the OpenAI API must be able to use this gateway unchanged,
while the gateway transparently routes to the best NVIDIA NIM model and remains
observable and resilient. Responses and errors must be predictable and
OpenAI-shaped.

## Requirements
1. `POST /v1/chat/completions` MUST accept the OpenAI chat schema (`messages`,
   optional `model`, `stream`, `max_tokens`, `temperature`, `tools`,
   `tool_choice`) and return an OpenAI-format completion or, when `stream:true`,
   a `text/event-stream` of `data:` chunks terminated by `data: [DONE]`.
2. Unspecified `model` MUST be resolved by the router (see `ROUTING.md`); an
   explicit `model` MUST be honored when registered.
3. Responses MUST carry tracing headers `X-Request-ID`, `X-Selected-Model`,
   `X-Task-Type`, `X-Routing-Confidence`, and `X-Cache`; autoscaled and
   fallback responses add `X-Autoscaled`/`X-Agent-Count` and `X-Model-Fallback`.
4. `tools`/`tool_choice` and other unknown OpenAI parameters MUST pass through to
   NIM unchanged, and `tool_calls` responses MUST be returned unmodified.
5. Errors MUST use the OpenAI error envelope `{"error": {message, type, code}}`
   with HTTP status matching `code`; upstream failures surface as 502, key
   exhaustion and budget/backpressure limits as 503 (with `Retry-After`).
6. `GET /v1/models` MUST return the router's registry by default and the NIM
   catalog when `?source=upstream`.
7. `POST /v1/embeddings` MUST default the model when omitted, require non-empty
   `input` (else 400), and record usage/cost.
8. `POST /explain` MUST return the answer plus the routing decision detail
   (task, confidence, per-task scores, selected model, fallback flag, latency,
   usage, cost) for the dashboard playground.
9. `/health` MUST be liveness-only; `/ready` MUST report HTTP-client and
   API-key readiness plus any unhealthy (open-circuit) models.

## Non-Goals
This spec does not define model-selection scoring (`ROUTING.md`), key handling
(`SECURITY_AND_KEYS.md`), metrics content (`OBSERVABILITY.md`), or cost
accounting (`COST.md`).

## Acceptance Evidence
- `tests/test_basic.py` and `tests/test_features.py` exercise the endpoints,
  streaming, tool passthrough, error codes, and the models/explain surfaces.
- Live verification against build.nvidia.com is recorded in PR summaries.
- OpenAI clients can call `/v1/chat/completions` without modification.

## Token Budget Class
Project contract. Load for work on the gateway HTTP surface.

## Related Specs
- `ROUTING.md`
- `SECURITY_AND_KEYS.md`
- `OBSERVABILITY.md`

## AI Agent Directives
When changing endpoints, preserve OpenAI compatibility and the tracing-header and
error-envelope contracts. Add or update tests for any new endpoint or parameter.
