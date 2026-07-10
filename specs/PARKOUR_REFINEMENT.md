# PARKOUR Verifier and Iterative Refinement

## Scope

The optional verify-and-refine loop for PARKOUR runs: a server-owned verifier
role that scores a candidate answer against the request, and a bounded
iterative refinement loop that revises the answer using verifier feedback before
returning it. Covers the verifier verdict schema, the refinement loop and its
termination guarantees, resource bounds and accounting, streaming events,
telemetry, and failure behavior. Project-scoped for
`github.com/joeldg/nvidiarouter`. This spec narrows `PARKOUR.md`; it does not
relax any of its safety, routing, cost, or observability requirements.

## Intent

PARKOUR today plans once, runs workers in parallel, and synthesizes a single
answer with no evaluation of that answer's quality. This capability adds a
Thinker/Worker/Verifier-style loop: a candidate answer is verified, and if it
falls short and budget remains, it is revised with targeted feedback and
re-verified. The goal is higher answer quality on hard tasks without unbounded
cost, non-termination, recursion, or any change to ordinary PARKOUR or
non-PARKOUR behavior when the loop is disabled.

## Requirements

1. The refinement loop MUST be disabled by default behind a dedicated setting
   (`ENABLE_PARKOUR_REFINEMENT`, default `false`) that is independent of the
   PARKOUR and research enable flags. When disabled, PARKOUR MUST behave exactly
   as specified in `PARKOUR.md` with no extra verifier or revision calls.
2. The verifier MUST be a server-owned role, never a client-supplied tool or
   client-parameterizable behavior. It MUST route through the same gateway
   controls as other PARKOUR calls (routing, key pool, retries, fallback,
   circuit breaker, backpressure, budget, cost, metrics) and MUST NOT select
   `parkour`. Verifier and revision calls MUST NOT recurse into a nested loop.
3. The verifier MUST return a bounded, schema-validated verdict containing at
   least a numeric quality score in a fixed range, an accept/revise decision,
   and bounded, structured feedback usable to guide a revision. Malformed or
   unparseable verdicts MUST be treated as a verifier failure under req.9, not
   as an implicit accept.
4. Each run MUST enforce hard limits for maximum refinement iterations, maximum
   verifier calls, maximum revision (worker) calls, added wall-clock time, and
   added tokens and estimated cost. Every limit MUST be configurable and
   enforced during execution. Refinement tokens and cost MUST roll into the
   existing PARKOUR aggregate accounting and daily budget defined in
   `PARKOUR.md` and `COST.md`; the loop MUST NOT create a parallel budget path.
5. The loop MUST be guaranteed to terminate. It MUST stop on the first of:
   verifier acceptance at or above the configured threshold, exhaustion of the
   iteration limit, exhaustion of any resource limit, or a no-improvement stop
   when a revision does not raise the verifier score by a configured minimum
   margin. The terminating condition MUST be recorded.
6. Revision feedback substitution MUST be typed and bounded like PARKOUR
   dependency context: it MUST NOT evaluate arbitrary expressions, MUST be
   deterministically truncated, and truncation MUST be recorded. The prior
   candidate and the verifier feedback are the only refinement inputs a revision
   receives beyond the original request context.
7. The run MUST return the best candidate observed by verifier score, not merely
   the last one produced, so a lower-quality final revision never replaces a
   better earlier answer. Ties MUST resolve to the earlier candidate.
8. If verification or revision fails, the loop MUST be non-fatal: PARKOUR MUST
   return the best answer produced so far and MUST mark the run as unverified or
   partially verified in trace metadata. Verifier failure MUST NOT convert a
   usable answer into an error.
9. When `stream: true`, the loop MAY emit bounded, namespaced progress events
   (for example `verification_started`, `verification_completed`,
   `revision_started`, `revision_completed`, `refinement_stopped`) consistent
   with `PARKOUR.md` req.11. These events MUST NOT include full prompts, full
   candidate text, verifier rationale beyond a bounded summary, API keys, or
   authorization data.
10. Refinement telemetry MUST record loop invocations, total iterations,
    accepts, rejects, no-improvement stops, limit stops, verifier failures, the
    score of the returned answer, and added verifier/revision token and cost
    figures separated from conductor/worker/synthesizer totals while still
    rolling into the PARKOUR total. JSON metrics and Prometheus exposition MUST
    remain equivalent, consistent with `OBSERVABILITY.md`.
11. Documentation MUST disclose the latency and cost amplification of the loop
    and its tuning knobs. `.env.example` and `README.md` MUST document the flag,
    the acceptance threshold, the no-improvement margin, and every refinement
    limit.

## Non-Goals

This spec does not introduce a learned, evolved, or reinforcement-learned
coordinator; the loop is a bounded, deterministic control structure, not a
trained policy. It does not add multi-vendor model diversity or a heterogeneous
frontier-model panel; worker and verifier model selection remain governed by
`ROUTING.md`. It does not add tree search, best-of-N sampling beyond the
sequential refine loop, persistent or resumable loops, recursive PARKOUR, or
client-supplied verifiers. It does not change the disabled-path behavior of
PARKOUR, research, or ordinary routing.

## Acceptance Evidence

- Schema tests cover valid verdicts, malformed/unparseable verdicts (treated as
  failure, never implicit accept), and score-range enforcement.
- Loop tests prove termination on acceptance, iteration-limit, resource-limit,
  and no-improvement conditions, and that the terminating condition is recorded.
- Best-candidate tests prove the returned answer is the highest-scored observed
  candidate, with ties resolving to the earlier one.
- Bounds tests prove verifier/revision call, iteration, wall-clock, token, and
  cost limits are enforced during execution and rolled into PARKOUR aggregates
  and the daily budget, with no recursion into `parkour`.
- Failure tests prove verifier/revision failure returns the best prior answer
  marked unverified, never an error, and never inappropriate fan-out.
- Gateway/observability tests prove disabled-by-default behavior, bounded
  streaming events, JSON/Prometheus parity, and that no secrets or full prompts
  appear in events or logs.
- Regression tests prove PARKOUR with the loop disabled, and ordinary
  non-PARKOUR routing, are byte-for-byte unchanged.

## Token Budget Class

Project contract. Load for any work on the PARKOUR verifier role, the refinement
loop, refinement limits/telemetry, or refinement UI.

## Related Specs

- `PARKOUR.md`
- `PARKOUR_RESEARCH.md`
- `ROUTING.md`
- `COST.md`
- `OBSERVABILITY.md`
- `GATEWAY_API.md`

## AI Agent Directives

Keep the loop opt-in and disabled by default, server-owned, bounded, and
guaranteed to terminate. Never treat a malformed verdict as an accept, never let
a worse revision replace a better answer, and never let verification failure
turn a usable answer into an error. Route every verifier and revision call
through existing controls, prohibit recursion into `parkour`, and roll all added
tokens and cost into the PARKOUR aggregate. Do not begin implementation until
this spec is reviewed, published, and synced; propose changes for review and
never self-approve.
