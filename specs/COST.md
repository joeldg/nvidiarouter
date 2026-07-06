# Cost and Budget

## Scope
Monetary cost accounting and spend control for upstream model usage: per-model
pricing, per-request USD cost from token usage, the daily budget guardrail, and
optional cost-aware routing. Project-scoped for `github.com/joeldg/nvidiarouter`.
(Distinct from `TOKENOMICS.md`, which governs spec-context token budget, not USD.)

## Intent
Operators should see what inference costs and be able to cap daily spend, and
optionally bias routing toward cheaper models — without changing default
behavior when cost controls are off.

## Requirements
1. Each registered model MUST carry input/output price per 1k tokens (0 on the
   free tier); prices are representative metadata, not billing truth.
2. On each successful completion (streaming and non-streaming), request cost
   MUST be computed from prompt/completion token usage and the model's pricing,
   and recorded per model and globally.
3. A daily budget (`daily_budget_usd`, 0 = unlimited) MUST track spend in a
   rolling window; when exceeded, new upstream requests MUST be refused with 503
   `budget_exceeded` until the window resets.
4. Cost-aware routing (`cost_weight`, default 0 = off) MUST, when enabled,
   penalize pricier models in scoring without otherwise altering selection.
5. Autoscaled responses MUST report aggregated token usage across sub-agents,
   and cost MUST be recorded for the model used.
6. Cost and remaining budget MUST be surfaced via the metrics snapshot and the
   Prometheus exposition (see `OBSERVABILITY.md`).

## Non-Goals
This spec does not define provider billing reconciliation or invoicing, nor the
routing algorithm itself (`ROUTING.md`).

## Acceptance Evidence
- `tests/test_features.py` covers `compute_cost`, the budget guardrail
  (allow/exceed/unlimited), per-model cost recording, and cost-aware routing
  preferring the cheaper model when `cost_weight > 0`.

## Token Budget Class
Project contract. Load for work on pricing, cost accounting, or budgets.

## Related Specs
- `ROUTING.md`
- `OBSERVABILITY.md`
- `TOKENOMICS.md`

## AI Agent Directives
Keep pricing as metadata, not a billing source of truth. Record cost on every
successful call. Do not enable cost-aware routing or budgets by default.
