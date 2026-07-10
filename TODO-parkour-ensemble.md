# PARKOUR Multi-Model Ensemble Panel Implementation Plan

## Goal

Give PARKOUR genuine **model diversity** — the third Fugu pillar. A graph node
may fan one prompt across an explicit, server-configured set of two or more
**distinct** models in parallel and combine their answers into one node result,
instead of routing a single model per task type. This approximates Fugu's
"diverse pool of powerful models" while staying bounded and routed through
existing controls.

Opt-in and disabled by default. Ordinary PARKOUR, research, and refinement runs,
and non-PARKOUR routing, must retain their current behavior and cost.

## Where this sits vs. Fugu

Fugu orchestrates a **diverse multi-vendor pool** under a **learned** coordinator.
This plan delivers the diversity substrate — parallel distinct-model panels with
combination — as **static configuration**, not a trained selector. The learned/RL
coordinator remains deferred (see the refinement plan's Deferred section).

## Governing Contracts

- `PARKOUR_ENSEMBLE.md` — panel opt-in/membership, concurrency, bounds,
  partial-failure tolerance, combination policy, accounting, telemetry.
- `PARKOUR.md` — the panel narrows this; core bounds and no-recursion still apply.
- `ROUTING.md` — panel members bypass single-model selection but still use the
  key pool, retries, fallback, circuit breaker, and metrics.
- `PARKOUR_REFINEMENT.md` — optional verifier-based combination/selection.
- `COST.md` / `OBSERVABILITY.md` — aggregate accounting and JSON/Prometheus parity.

The ensemble spec must be approved and published before implementation begins.

## Architectural Contract

```text
panel node (server-flagged)
        |
        v
 explicit distinct model set  {model_a, model_b, model_c}  (config, deduped, <= max)
        |            |            |
     routed       routed       routed      members run concurrently under the
     member       member       member      existing PARKOUR semaphore + controls
        \            |            /
          surviving member results (partial-failure tolerant)
                     |
                     v
        combine: synthesizer (default) OR verifier-select (if refinement on)
                     |
                     v
              one NodeResult (+ per-member internal accounting)
```

Panel members and the combiner MUST NOT select `parkour` and MUST NOT recurse.

## Phase 0 — Spec Review and Interface Freeze

- [ ] Submit `PARKOUR_ENSEMBLE.md` as a project-scoped draft.
- [ ] Obtain human approval and publication; `specreg sync` the bundle.
- [ ] Freeze settings/flag names and the panel-membership + max-size semantics.
- [ ] Confirm disabled-by-default is byte-for-byte identical to current PARKOUR.

Acceptance evidence: registry review ID + published version; `specreg check`
reports the local bundle current.

## Phase 1 — Configuration and Membership

- [ ] Add `ENABLE_PARKOUR_ENSEMBLE` (default off), independent of the other
      PARKOUR flags.
- [ ] Add `PARKOUR_ENSEMBLE_MODELS` (comma-separated) and
      `PARKOUR_ENSEMBLE_MAX_SIZE`.
- [ ] Parse, de-duplicate, and exclude `parkour`; expose the effective panel via
      a settings property. Require >= 2 distinct members or fall back to single
      routing.
- [ ] Document settings and the cost/latency multiplication in `.env.example`
      and `README.md`.

Verification: default-off, parsing/de-dup, `parkour`-exclusion, and
fewer-than-two fallback tests.

## Phase 2 — Panel Node and Concurrent Execution

- [ ] Add a server/conductor-controlled `panel` opt-in on `SubtaskSpec`
      (ignored unless the panel is enabled; never client-parameterizable).
- [ ] Execute panel members concurrently under the existing scheduler semaphore,
      each via the routed worker path (key pool, retries, fallback, breaker,
      backpressure, budget, metrics); prohibit `parkour` selection/recursion.
- [ ] Enforce max panel size (deterministic truncation) and count every member
      against the run's call/concurrency/token/cost limits.

Verification: concurrency test with deterministic barriers; call/limit
enforcement including panel calls; recursion-prevention test.

## Phase 3 — Partial Failure and Combination

- [ ] Tolerate partial failure: proceed on surviving members, record failures,
      fail the node only when all members fail.
- [ ] Combine surviving responses: synthesizer by default; when refinement is
      enabled, the verifier MAY select/rank a best candidate.
- [ ] Ensure the combination call is routed, bounded, and never selects
      `parkour`.

Verification: partial-failure matrix; synthesizer-default vs. verifier-select
tests; all-fail node-failure test.

## Phase 4 — Accounting, Telemetry, and Metadata

- [ ] Aggregate member + combination tokens/cost into the node and PARKOUR
      totals without double counting; retain per-member model/outcome internally.
- [ ] Record panel invocations, configured/effective size, member
      successes/failures, distinct models used, added tokens/cost.
- [ ] Expose equivalent JSON and Prometheus signals (`nsr_parkour_ensemble_*`).
- [ ] Add bounded streaming events (`panel_started`, `panel_member_completed`,
      `panel_combined`) and a compact panel summary in the opt-in `parkour_trace`.

Verification: cost-reconciliation (per-member sum == aggregate); JSON/Prometheus
parity; streaming event-bound and redaction tests.

## Phase 5 — Hardening and Release

- [ ] Run unit, integration, lint, type, and full regression suites.
- [ ] Adversarial tests: duplicate/invalid members, `parkour` in the list,
      oversized panels, all-members-fail, and budget pressure (no bypass).
- [ ] Load-test panel concurrency; prove the semaphore and key budgets hold.
- [ ] Document expected cost/latency multiplication and tuning.
- [ ] Ship disabled by default; enable in an opt-in preview.
- [ ] Capture SpecRegistry traceability; run `finish_task` / `specreg comply`
      before commit.

Release gates: no implementation before published specs are synced; no unbounded
panel, no `parkour` recursion, no secret leakage, no budget bypass; disabled-path
behavior and cost unchanged; full suite and objective compliance pass.

## Deferred Beyond This Spec

- Learned / evolved / RL-trained panel selection (Fugu's coordinator).
- Weighted voting, embedding-based clustering, or best-of-N search over members.
- Per-member routing policies or dynamic panel membership.
- Cross-vendor credential management beyond existing key handling.
