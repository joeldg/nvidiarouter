# PARKOUR Verifier + Iterative Refinement Implementation Plan

## Goal

Add an opt-in, bounded **verify-and-refine loop** to PARKOUR so a candidate
answer is scored by a server-owned verifier and, when it falls short and budget
remains, revised with targeted feedback and re-verified before returning. This
moves PARKOUR toward the verify-and-refine behavior of Sakana Fugu's
Thinker/Worker/Verifier coordination — without adopting Fugu's learned/evolved
coordinator or multi-vendor model panel, which stay out of scope (see Deferred).

The loop is opt-in and disabled by default. Ordinary PARKOUR runs, research
runs, and non-PARKOUR routing must retain their current behavior and cost.

## What this is (and is not) vs. Fugu

Fugu = an **evolved/RL-trained** coordinator running a **sequential, multi-turn**
Thinker/Worker/**Verifier** loop over a **diverse multi-vendor** pool. This plan
adopts the highest-leverage, tractable slice — the **Verifier + bounded
iterative refinement** — as a deterministic control structure over PARKOUR's
existing routed workers. Learned coordination and vendor diversity are tracked
separately as deferred research.

## Governing Contracts

- `PARKOUR_REFINEMENT.md` — verifier role, refinement loop, termination
  guarantees, bounds, accounting, failure policy, and acceptance evidence.
- `PARKOUR.md` — the loop narrows this; core orchestration, limits, and the
  no-recursion rule still apply.
- `ROUTING.md` — verifier/reviser model selection through existing controls.
- `COST.md` — refinement spend rolls into the PARKOUR aggregate and daily budget.
- `OBSERVABILITY.md` — refinement metrics, JSON/Prometheus parity, redacted logs.
- `GATEWAY_API.md` — streaming refinement events and response metadata.

The refinement spec must be approved and published before implementation begins.

## Architectural Contract

```text
validated ExecutionPlan -> workers -> initial synthesis (candidate v0)
        |
        v
   [ refinement loop, bounded ]
        |
   verifier scores candidate  --- accept? --> return best candidate
        |  (revise)
        v
   reviser (routed worker) revises using bounded verifier feedback
        |
        v
   re-verify (score) --> stop on: accept | max-iterations | resource limit |
                                   no-improvement margin
        |
        v
return best-scored candidate + compact refinement trace metadata
```

The verifier and reviser are ordinary routed calls; they MUST NOT select
`parkour` and MUST NOT open a nested refinement loop.

## Phase 0 — Spec Review and Interface Freeze

- [ ] Submit `PARKOUR_REFINEMENT.md` as a project-scoped draft.
- [ ] Obtain human approval and publication; `specreg sync` the bundle.
- [ ] Freeze the verdict schema (score range, accept/revise, bounded feedback)
      and the settings/flag names.
- [ ] Confirm disabled-by-default is byte-for-byte identical to current PARKOUR.

Acceptance evidence: registry review ID + published version; `specreg check`
reports the local bundle current.

## Phase 1 — Configuration and Roles

- [ ] Add `ENABLE_PARKOUR_REFINEMENT` (default off), independent of
      `ENABLE_PARKOUR` and `ENABLE_PARKOUR_RESEARCH`.
- [ ] Add `PARKOUR_VERIFIER_MODEL` (defaulting to the synthesizer/conductor
      model) and an acceptance threshold, no-improvement margin, and hard limits:
      max iterations, max verifier calls, max revision calls, added wall-clock,
      added tokens, and added cost.
- [ ] Document every setting in `.env.example` and `README.md`, including the
      latency/cost amplification disclosure.

Verification: default-off and validation tests; disabled-path regression proving
normal PARKOUR is unchanged.

## Phase 2 — Verifier Verdict Schema and Call

- [ ] Add a Pydantic `Verdict` model (score in fixed range, accept/revise,
      bounded structured feedback) with `extra="forbid"`.
- [ ] Parse native structured output first; fenced-JSON fallback only, matching
      the conductor-plan parsing approach.
- [ ] Treat malformed/unparseable/out-of-range verdicts as a verifier failure —
      never an implicit accept.
- [ ] Implement a server-owned verifier call routed through existing controls
      that cannot select `parkour`.

Verification: valid/malformed/out-of-range verdict tests; recursion-prevention
test.

## Phase 3 — Bounded Refinement Loop

- [ ] Implement the loop over the synthesized candidate: verify -> (revise ->
      re-verify)\* under all Phase 1 limits, enforced during execution.
- [ ] Reviser is a routed worker call receiving only the prior candidate and
      bounded, deterministically truncated verifier feedback (record truncation).
- [ ] Track every candidate's score; return the best observed candidate, ties to
      the earlier one.
- [ ] Record the terminating condition (accept | max-iterations | resource-limit
      | no-improvement).

Verification: termination tests for each stop condition; best-candidate
selection test; deterministic-truncation test; limit-enforcement tests using
barriers, not timing alone.

## Phase 4 — Accounting and Failure Semantics

- [ ] Roll verifier/reviser tokens and cost into the PARKOUR aggregate and daily
      budget without double counting; retain per-role internal accounting.
- [ ] On verifier or reviser failure, return the best prior candidate marked
      unverified/partially-verified — never an error, never inappropriate
      fan-out.
- [ ] Ensure cancellation/deadline releases any held concurrency and gauges.

Verification: cost-reconciliation test (per-call sum == aggregate); failure
matrix; cancellation cleanup test.

## Phase 5 — Gateway Contract and Metadata

- [ ] Add bounded, namespaced streaming events (`verification_started`,
      `verification_completed`, `revision_started`, `revision_completed`,
      `refinement_stopped`) with no full prompts/candidates/secrets.
- [ ] Extend the opt-in `parkour_trace` block with a compact refinement summary
      (iterations, returned score, stop reason, verified flag); keep default
      responses OpenAI-compatible and headers compact.

Verification: OpenAI response-shape and header-size tests; streaming event-bound
tests; disabled-path parity test.

## Phase 6 — Observability, Cost, and Operator UI

- [ ] Record loop invocations, total iterations, accepts, rejects,
      no-improvement stops, limit stops, verifier failures, returned score, and
      added verifier/reviser token/cost figures.
- [ ] Expose equivalent JSON and Prometheus signals (`nsr_parkour_refine_*`).
- [ ] Emit redacted structured run events; add a bounded refinement summary to
      `/explain`, the web dashboard, and the TUI.

Verification: metrics snapshot/Prometheus parity tests; cost reconciliation; log
redaction tests; dashboard tests for accept, no-improvement, and limit-stop.

## Phase 7 — Hardening and Release

- [ ] Run unit, integration, lint, type, and full regression suites.
- [ ] Add adversarial verifier-output tests (reward-hacking scores, always-accept,
      always-revise) and prove bounded, terminating behavior.
- [ ] Load-test added concurrency; prove no key-budget bypass and no
      non-termination.
- [ ] Document expected latency/cost amplification and tuning guidance.
- [ ] Ship disabled by default; enable in an opt-in preview.
- [ ] Capture SpecRegistry traceability; run `finish_task` / `specreg comply`
      before commit.

Release gates: no implementation before published specs are synced; no unbounded
or non-terminating loop, no recursive PARKOUR, no secret leakage, no budget
bypass; disabled-path behavior and cost unchanged; full suite and objective
SpecRegistry compliance pass.

## Deferred Beyond This Spec

- **Multi-vendor model diversity** — fan a node/verifier across an explicit set
  of distinct models (Fugu's diverse pool). Smaller, routing-level change; its
  own minor `ROUTING.md`/spec update.
- **Learned / evolved / RL-trained coordinator** — Fugu's TRINITY-style trained
  Conductor. Genuinely different, research-grade engine; separate contract.
- **Tree search / best-of-N beyond the sequential loop** (e.g. AB-MCTS-style
  widen/deepen search over candidates).
- **Per-node (not just final-answer) verification and refinement.**
- **Client-supplied or pluggable verifiers.**
