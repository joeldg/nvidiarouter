# Model Recommendation Advisor

## Scope
The model-recommendation advisor: the `recommend` CLI command and the
`GET /v1/recommend` endpoint, and the pure advisor function that selects, for
each task type, the best registered model with an explainable rationale.
Project-scoped for `github.com/joeldg/nvidiarouter`.

## Intent
Operators should get an explainable "best model per task" answer derived from the
same data and scoring the gateway actually routes with — so model selection is a
transparent, reproducible recommendation rather than guesswork. The advisor makes
routing decisions inspectable without sending any traffic.

## Requirements
1. For each `TaskType`, the advisor MUST recommend exactly one registered model
   that supports that task, or explicitly report that no registered model
   supports it — it MUST NOT recommend a model whose `supported_tasks` excludes
   the task.
2. Candidate ranking MUST use the router's own model scoring (see `ROUTING.md`)
   so the recommendation is consistent with how requests would actually route.
3. Each recommendation MUST include a rationale exposing the chosen model's
   parameter size, latency basis, throughput, and cost, plus the next-best
   alternative(s), so a reader can see why it won.
4. When live metrics exist for a model, the advisor MUST prefer observed latency
   over the size-based estimate and MUST record which basis was used
   (`measured` vs `estimated`), per `OBSERVABILITY.md`.
5. The advisor MUST be read-only and pure: it MUST NOT make any upstream NIM
   call, mutate registry/metrics state, or require network access.
6. `GET /v1/recommend` MUST return JSON keyed by task type with `{model,
   basis, rationale, alternatives}` for every task; an optional `?task=<name>`
   MUST return only that task (400 for an unknown task name).
7. The `recommend` CLI command MUST render the same recommendations as a table
   and MUST NOT require the gateway to be running.
8. Recommendations MUST be deterministic for a fixed registry + metrics state.

## Non-Goals
This spec does not change routing behavior (`ROUTING.md`), does not benchmark or
discover models (`MODEL_DISCOVERY.md`), and does not persist recommendations. It
does not guarantee the recommended model is globally optimal — only best under
the current registry, metrics, and configured weights.

## Acceptance Evidence
- Unit tests assert the recommendation for a task equals the top-scored model
  whose `supported_tasks` includes that task, and that unsupported tasks report
  "no model".
- A test asserts the advisor performs zero upstream calls (pure function).
- A test covers the `?task=` filter and the unknown-task 400.
- `GET /v1/recommend` returns all task types; `nvidia-smartroute recommend`
  renders a table without a running gateway.

## Token Budget Class
Project contract. Load for work on the recommendation advisor, CLI, or endpoint.

## Related Specs
- `ROUTING.md`
- `MODEL_DISCOVERY.md`
- `OBSERVABILITY.md`
- `COST.md`
- `GATEWAY_API.md`

## AI Agent Directives
Keep the advisor's ranking identical to the router's scoring so recommendations
never contradict real routing. Keep it read-only and side-effect free. Always
surface the basis (measured vs estimated) and the losing alternatives so the
recommendation is auditable.
