# Routing

## Scope
Request classification and model selection: the capability analyzer, the model
registry and scoring, model fallback chains, the per-model circuit breaker, and
the optional adaptive (bandit) routing strategy. Project-scoped for
`github.com/joeldg/nvidiarouter`.

## Intent
Each request should be served by the most suitable registered model for its
task, balancing capability, latency, and (optionally) cost, and degrading
gracefully when a model fails — without the client specifying a model.

## Requirements
1. The analyzer MUST classify a request into a `TaskType` using weighted,
   word-boundary keyword scoring plus structural signals (image content →
   vision; arithmetic → mathematics) and return a confidence in [0, 1].
2. Word-boundary matching MUST be used for single-word signals so substrings do
   not cause false positives (e.g. "sum" must not match "summarize").
3. `select_best_model(task)` MUST choose among models supporting the task,
   scoring on quality, reliability, and latency, with an optional cost penalty
   (`cost_weight`) and a small size preference for deterministic tie-breaks.
4. Unknown latency MUST be estimated from model size, never treated as zero
   (which would read as "instant" and out-rank measured models).
5. When `routing_strategy = "adaptive"`, model choice MUST use an epsilon-greedy
   bandit over suitable models, learning from per-request outcomes
   (success and latency) recorded after each call.
6. On a hard upstream failure (404 / 5xx / transport), the request MUST fail
   over to the next-best model for the task; other 4xx MUST NOT fan out.
7. A model that fails repeatedly MUST be taken out of rotation by the circuit
   breaker and probed back in after a cooldown; 4xx client errors MUST NOT trip
   it.
8. `RoutingDecision` MUST carry `request_id`, `task_type`, `selected_model`,
   `confidence`, and human-readable `reasoning`.

## Non-Goals
This spec does not define the HTTP surface (`GATEWAY_API.md`), how models are
discovered/registered (`MODEL_DISCOVERY.md`), or cost math (`COST.md`).

## Acceptance Evidence
- `tests/test_routing.py` covers classifier outcomes (incl. word-boundary and
  vision/arithmetic cases) and model selection per task.
- `tests/test_features.py` covers latency/cost-aware scoring, fallback,
  circuit-breaker trip/recover, and adaptive selection.

## Token Budget Class
Project contract. Load for work on classification or model selection.

## Related Specs
- `GATEWAY_API.md`
- `MODEL_DISCOVERY.md`
- `COST.md`

## AI Agent Directives
Keep classification deterministic and testable. Do not add routing signals
without tests. Prefer measured latency (live/benchmark) over priors when present.
