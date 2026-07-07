# Routing

## Scope
Request classification and model selection: the capability analyzer, the model
registry and scoring, model fallback chains, the per-model circuit breaker, the
optional adaptive (bandit) routing strategy, and optional session affinity.
Project-scoped for `github.com/joeldg/nvidiarouter`.

## Intent
Each request should be served by the most suitable registered model for its
task, balancing capability, latency, and (optionally) cost, and degrading
gracefully when a model fails — without the client specifying a model. Callers
that want conversational continuity should be able to opt into keeping a
conversation on one model, without changing the stateless default for everyone
else.

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
9. Session affinity is OPTIONAL and OFF by default. When `session_affinity` is
   enabled AND a request carries a stable session key — the `X-Session-Id`
   header, or the OpenAI `user` field when the header is absent — the router
   MUST reuse the model previously selected for that key instead of
   re-classifying, until the affinity entry expires (`session_affinity_ttl`) or
   the pinned model is unavailable. When affinity is disabled OR no session key
   is present, routing MUST remain fully stateless and per-request; this is the
   default and MUST be byte-for-byte the pre-affinity behavior.
10. Affinity MUST fail safe. If the pinned model is circuit-broken (req.7) or no
   longer registered, the router MUST re-route by normal scoring (req.3 / req.5),
   re-pin the session to the model that actually served, and MUST NOT fail the
   request for want of the original pin. An explicit `model` override in the
   request MUST take precedence over an affinity pin and MUST NOT be recorded as
   the session's pin.
11. Affinity state MUST be bounded (a TTL plus a maximum entry count with LRU
   eviction) and MUST NOT persist across process restarts. `RoutingDecision`
   MUST record whether the selected model came from an affinity pin
   (`from_session`) so the decision remains auditable and observable.

## Non-Goals
This spec does not define the HTTP surface (`GATEWAY_API.md`), how models are
discovered/registered (`MODEL_DISCOVERY.md`), or cost math (`COST.md`). Session
affinity maps a session key to a chosen model id only — it is NOT a conversation
store, memory, or context cache, and it does not alter classification of the
latest turn when a session is not pinned.

## Acceptance Evidence
- `tests/test_routing.py` covers classifier outcomes (incl. word-boundary and
  vision/arithmetic cases) and model selection per task.
- `tests/test_features.py` covers latency/cost-aware scoring, fallback,
  circuit-breaker trip/recover, and adaptive selection.
- Affinity tests assert: with affinity off (default) routing is unchanged and
  stateless; with affinity on, a second request bearing the same session key
  reuses the first pick; a circuit-broken/deregistered pin re-routes and re-pins
  without failing the request; an explicit `model` override beats the pin and is
  not stored; and the affinity store honors TTL expiry and the LRU max-entry
  bound. `RoutingDecision.from_session` is asserted for a pinned vs unpinned pick.

## Token Budget Class
Project contract. Load for work on classification or model selection.

## Related Specs
- `GATEWAY_API.md`
- `MODEL_DISCOVERY.md`
- `COST.md`
- `OBSERVABILITY.md`

## AI Agent Directives
Keep classification deterministic and testable. Do not add routing signals
without tests. Prefer measured latency (live/benchmark) over priors when present.
Keep session affinity opt-in and default-off; never let a pin defeat the circuit
breaker or an explicit model override, and keep the affinity store bounded and
non-persistent. Surface `from_session` so a pinned decision is auditable.
