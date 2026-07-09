# Gateway API

## Scope
The OpenAI-compatible HTTP surface exposed by the NVIDIA-SmartRoute gateway:
`/v1/chat/completions` (streaming and non-streaming), `/v1/embeddings`,
`/v1/models`, `/explain`, and the operational endpoints `/health`, `/ready`,
`/metrics`, `/metrics/prometheus`, `/dashboard`. This includes explicit
selection of the project-scoped `parkour` virtual model. Project-scoped; narrows
the CLI/Developer-Tooling project type for `github.com/joeldg/nvidiarouter`.

## Intent
Clients that speak the OpenAI API must be able to use this gateway unchanged,
while the gateway transparently routes to the best NVIDIA NIM model and remains
observable and resilient. Clients may explicitly opt into PARKOUR multi-agent
execution through the standard `model` field. Responses and errors must be
predictable and OpenAI-shaped.

## Requirements
1. `POST /v1/chat/completions` MUST accept the OpenAI chat schema (`messages`,
   optional `model`, `stream`, `max_tokens`, `temperature`, `tools`,
   `tool_choice`) and return an OpenAI-format completion or, when `stream:true`,
   a `text/event-stream` of `data:` chunks terminated by `data: [DONE]`.
   Virtual models MAY stream bounded, namespaced progress metadata when their
   governed contract permits it.
2. Unspecified `model` MUST be resolved by the router (see `ROUTING.md`); an
   explicit `model` MUST be honored when registered. An enabled virtual model
   MUST be dispatched to its governed execution strategy rather than sent
   upstream as a NIM model ID.
3. Responses MUST carry tracing headers `X-Request-ID`, `X-Selected-Model`,
   `X-Task-Type`, `X-Routing-Confidence`, and `X-Cache`; autoscaled and
   fallback responses add `X-Autoscaled`/`X-Agent-Count` and `X-Model-Fallback`.
   Virtual-model responses MAY add a compact execution type and opaque run ID.
   Full execution graphs and unbounded JSON MUST NOT be placed in headers.
4. `tools`/`tool_choice` and other unknown OpenAI parameters MUST pass through to
   NIM unchanged, and `tool_calls` responses MUST be returned unmodified, except
   when an explicitly selected virtual model has a stricter governed tool
   policy that is validated before execution.
5. Errors MUST use the OpenAI error envelope `{"error": {message, type, code}}`
   with HTTP status matching `code`; upstream failures surface as 502, key
   exhaustion and budget/backpressure limits as 503 (with `Retry-After`).
6. `GET /v1/models` MUST return the router's registry, including enabled virtual
   models, by default and only the NIM catalog when `?source=upstream`. Virtual
   models MUST be distinguishable from upstream models in returned metadata.
7. `POST /v1/embeddings` MUST default the model when omitted, require non-empty
   `input` (else 400), and record usage/cost.
8. `POST /explain` MUST return the answer plus the routing decision detail
   (task, confidence, per-task scores, selected model, fallback flag, latency,
   usage, cost) for the dashboard playground. For virtual-model runs it MAY
   include a bounded, redacted execution summary.
9. `/health` MUST be liveness-only; `/ready` MUST report HTTP-client and
   API-key readiness plus any unhealthy (open-circuit) models.
10. When the client explicitly selects `model: "parkour"`, the gateway MUST
    follow `PARKOUR.md`. The response MUST report `model: "parkour"` while
    internal telemetry retains the actual conductor, worker, and synthesizer
    models.
11. PARKOUR requests with `stream: true` MUST follow `PARKOUR.md`: they MAY
    emit bounded `parkour_event` progress chunks and final answer
    `choices[].delta.content` chunks, MUST use `text/event-stream`, and MUST
    terminate with `data: [DONE]`. The gateway MUST NOT imitate true upstream
    token streaming by rapidly replaying buffered intermediate worker output.
12. Virtual-model graph metadata MUST be opt-in, bounded, namespaced, and
    redacted. Default completion responses MUST remain usable by ordinary
    OpenAI clients that ignore unknown response fields.

## Non-Goals
This spec does not define model-selection scoring (`ROUTING.md`), key handling
(`SECURITY_AND_KEYS.md`), metrics content (`OBSERVABILITY.md`), cost accounting
(`COST.md`), or PARKOUR graph execution semantics (`PARKOUR.md`).

## Acceptance Evidence
- `tests/test_basic.py` and `tests/test_features.py` exercise the endpoints,
  streaming, tool passthrough, error codes, and the models/explain surfaces.
- PARKOUR gateway tests cover explicit selection, enabled/disabled discovery,
  virtual versus upstream model listing, public model identity, compact headers,
  opt-in bounded metadata, tools policy, and progress-event streaming.
- Live verification against build.nvidia.com is recorded in PR summaries.
- OpenAI clients can call ordinary `/v1/chat/completions` without modification
  and can select enabled PARKOUR through the standard `model` field.

## Token Budget Class
Project contract. Load for work on the gateway HTTP surface.

## Related Specs
- `ROUTING.md`
- `SECURITY_AND_KEYS.md`
- `OBSERVABILITY.md`
- `PARKOUR.md`

## AI Agent Directives
When changing endpoints, preserve OpenAI compatibility and the tracing-header and
error-envelope contracts. Dispatch virtual models locally, keep metadata bounded
and redacted, and add or update tests for every new endpoint or parameter.
