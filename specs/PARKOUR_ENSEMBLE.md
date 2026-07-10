# PARKOUR Multi-Model Ensemble Panel

## Scope

The optional multi-model diversity panel for PARKOUR: a graph node may fan a
single prompt across an explicit, configured set of two or more distinct models
in parallel, then combine their responses into one node result. Covers the panel
opt-in and membership, concurrent execution under existing gateway controls,
distinctness and bounds, partial-failure tolerance, combination policy,
accounting, and telemetry. Project-scoped for `github.com/joeldg/nvidiarouter`.
This spec narrows `PARKOUR.md` and `ROUTING.md`; it does not relax their safety,
resilience, cost, or observability requirements.

## Intent

PARKOUR normally selects one model per node by task type, so parallel sibling
nodes of the same type tend to route to the same model — there is no genuine
model diversity. This capability lets a node run one prompt across a diverse
panel of distinct models concurrently and combine their answers, moving PARKOUR
toward a diverse-pool ensemble that can outperform any single model. It must add
diversity without bypassing routing resilience, cost controls, or PARKOUR's
bounds, and without any change when disabled.

## Requirements

1. The ensemble panel MUST be disabled by default behind a dedicated setting
   (`ENABLE_PARKOUR_ENSEMBLE`, default `false`) that is independent of the
   PARKOUR, research, and refinement flags. When disabled, PARKOUR MUST behave
   exactly as specified in `PARKOUR.md`, and node execution MUST route a single
   model as today.
2. Panel membership MUST come only from server-side configuration (an explicit
   list of model identifiers), never from the client request. A node's use of
   the panel is a server/conductor-controlled flag; the client MUST NOT be able
   to specify, extend, or reorder panel members.
3. A panel MUST contain at least two distinct model identifiers after
   de-duplication, and MUST NOT include `parkour`. If fewer than two distinct
   valid members are configured, the node MUST fall back to ordinary
   single-model routing rather than failing.
4. Panel members MUST execute concurrently as bounded asyncio tasks under the
   existing PARKOUR concurrency semaphore, and each member call MUST route
   through the existing gateway paths for the key pool, retries, model fallback,
   circuit breaker, backpressure, daily budget, cost recording, and metrics.
   The panel MUST NOT open a parallel path that bypasses those controls, and no
   member MUST recurse into `parkour`.
5. Every panel member call MUST count against the run's total upstream call,
   concurrency, token, and cost limits defined in `PARKOUR.md`. A configurable
   maximum panel size MUST bound how many members may run for one node, and the
   effective panel MUST be truncated deterministically to that maximum.
6. Panel execution MUST tolerate partial failure: the node MUST proceed using
   the successful members' responses and record which members failed. The node
   MUST fail only if every panel member fails; a single member failure MUST NOT
   fail the node.
7. Panel responses MUST be combined into one node result by an explicit,
   documented policy. When the refinement loop (`PARKOUR_REFINEMENT.md`) is
   enabled, the combination MAY use the verifier to select or rank candidates;
   otherwise the existing synthesizer MUST combine them. The combination step is
   itself a routed call subject to the same controls and MUST NOT select
   `parkour`.
8. Node accounting MUST aggregate tokens and cost across all panel members and
   the combination call without double counting, while retaining the actual
   model identifier and outcome for each member internally.
9. When `stream: true`, the panel MAY emit bounded, namespaced progress events
   (for example `panel_started`, `panel_member_completed`, `panel_combined`)
   consistent with `PARKOUR.md` req.11. Events MUST NOT include full prompts,
   full member outputs, API keys, or authorization data, and MUST bound member
   identifiers and counts.
10. Ensemble telemetry MUST record panel invocations, configured and effective
    panel size, member successes and failures, distinct models used, and added
    tokens and cost, separated from ordinary worker figures while still rolling
    into the PARKOUR total. JSON metrics and Prometheus exposition MUST remain
    equivalent, consistent with `OBSERVABILITY.md`.
11. Documentation MUST disclose the cost and latency multiplication of running a
    panel (roughly proportional to panel size) and its tuning knobs.
    `.env.example` and `README.md` MUST document the flag, the member list, and
    the maximum panel size.

## Non-Goals

This spec does not introduce a learned, evolved, or reinforcement-learned
coordinator that selects panels; membership is static configuration. It does not
add cross-vendor credential management beyond the existing key handling in
`SECURITY_AND_KEYS.md`, nor per-member routing policies beyond distinctness and
bounds. It does not add weighted voting, embedding-based clustering, or
best-of-N search beyond the documented combination policy, and it does not
change PARKOUR, research, or refinement behavior when the panel is disabled.

## Acceptance Evidence

- Config tests cover default-off, member parsing/de-duplication, `parkour`
  exclusion, and the single-model fallback when fewer than two distinct members
  exist.
- Concurrency tests prove panel members run concurrently under the semaphore
  using deterministic barriers, and that per-run call/concurrency/token/cost
  limits are enforced including the panel calls.
- Partial-failure tests prove the node proceeds on surviving members, records
  member failures, and fails only when all members fail.
- Combination tests prove synthesizer combination by default and verifier-based
  selection when refinement is enabled, with no recursion into `parkour`.
- Accounting tests reconcile per-member plus combination tokens/cost with the
  node and PARKOUR aggregates.
- Observability tests assert JSON/Prometheus parity for ensemble metrics and
  bounded streaming events, with no secrets or full prompts in events or logs.
- Regression tests prove PARKOUR with the panel disabled, and ordinary
  non-PARKOUR routing, are byte-for-byte unchanged.

## Token Budget Class

Project contract. Load for any work on the PARKOUR ensemble panel, panel
membership/bounds, panel combination, or ensemble telemetry/UI.

## Related Specs

- `PARKOUR.md`
- `ROUTING.md`
- `PARKOUR_REFINEMENT.md`
- `COST.md`
- `OBSERVABILITY.md`
- `SECURITY_AND_KEYS.md`

## AI Agent Directives

Keep the panel opt-in and disabled by default, membership server-configured and
distinct, and every member routed through existing controls with no recursion
into `parkour`. Tolerate partial member failure, bound panel size, and roll all
added tokens and cost into the PARKOUR aggregate. Combine deterministically via
the synthesizer, or the verifier when refinement is enabled. Do not begin
implementation until this spec is reviewed, published, and synced; propose
changes for review and never self-approve.
