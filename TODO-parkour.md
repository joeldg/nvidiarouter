# PARKOUR Virtual Model Implementation Plan

## Goal

Add `parkour` as an explicitly selected virtual model in the OpenAI-compatible
gateway. Unlike a registered NVIDIA NIM model, PARKOUR is an execution strategy:
it decomposes a request into a bounded directed acyclic graph (DAG), runs ready
worker nodes concurrently through the gateway's existing model-routing and
resilience controls, and synthesizes their outputs into one completion.

PARKOUR is opt-in. Ordinary routed requests and explicit upstream model requests
must retain their current behavior and cost.

## Governing Contracts

- `PARKOUR.md` — virtual-model identity, graph lifecycle, safety bounds,
  failure policy, accounting, and acceptance evidence.
- `GATEWAY_API.md` — OpenAI request/response behavior, `/v1/models`, headers,
  errors, and initial streaming policy.
- `ROUTING.md` — worker model selection, explicit-model precedence, fallback,
  circuit breaker, and session-affinity behavior.
- `COST.md` — per-call and aggregate token/cost accounting and budget refusal.
- `OBSERVABILITY.md` — graph/run/node metrics, structured logs, dashboards, and
  Prometheus parity.
- `SECURITY_AND_KEYS.md` — authentication, key rotation, rate limiting, and
  secret masking for every internal upstream call.

The PARKOUR spec must be approved and published before implementation begins.

## Architectural Contract

```text
OpenAI request with model="parkour"
        |
        v
local request validation and PARKOUR limits
        |
        v
conductor call -> validated ExecutionPlan DAG
        |
        v
bounded async scheduler
   |          |          |
 routed     routed     routed       ready nodes run concurrently
 worker     worker     worker       through existing gateway controls
   \          |          /
        partial results
              |
              v
         synthesizer call
              |
              v
OpenAI-compatible response + compact PARKOUR trace metadata
```

PARKOUR workers must never select `parkour`; this prevents accidental recursive
orchestration. A future recursive mode would require a separate spec change.

## Phase 0 — Spec Review and Interface Freeze

- [x] Submit `PARKOUR.md` as a project-scoped draft.
- [x] Submit the `GATEWAY_API.md` minor-version change for review.
- [x] Obtain human approval and publication; sync the governed bundle.
- [x] Freeze the v1 API identifier as `parkour`, with display name `PARKOUR`.
- [x] Confirm initial `stream=true` behavior: reject with an OpenAI-shaped 400
      error rather than pretending buffered output is live token streaming.

Acceptance evidence:

- Registry review IDs and published versions.
- `specreg check` reports the local bundle is current.

## Phase 1 — Configuration and Virtual-Model Registration

- [x] Add a PARKOUR settings group, disabled by default.
- [x] Define conductor and synthesizer model settings.
- [x] Define hard limits for graph nodes, depth, width/concurrency, total
      upstream calls, wall-clock duration, per-node output, aggregate tokens,
      and estimated cost.
- [x] Register `parkour` as a virtual model in the router-facing model catalog
      without treating it as an upstream NIM capability profile.
- [x] Include `parkour` in the default `/v1/models` response only when enabled;
      never include it in `?source=upstream`.
- [x] Document all settings in `.env.example` and `README.md`.

Verification:

- Settings default-off and validation tests.
- `/v1/models` enabled/disabled and upstream-source tests.
- Regression test proving normal routing is byte-for-byte unchanged while
  PARKOUR is disabled.

## Phase 2 — Execution-Plan Schema and Validation

- [x] Add Pydantic models for `SubtaskSpec` and `ExecutionPlan`.
- [x] Require unique bounded node IDs, supported task types, explicit dependency
      lists, prompts, roles, and synthesis instructions.
- [x] Reject missing dependencies, self-dependencies, duplicate IDs, cycles,
      excessive depth/width/node count, and unsupported fields.
- [x] Parse native structured output first; support fenced JSON extraction only
      as a compatibility fallback.
- [x] Keep plan parsing pure and deterministic after conductor output is
      received.
- [x] Add a local complexity/direct-route rule so PARKOUR may execute a single
      worker without a synthesis call when decomposition adds no value.

Verification:

- Valid graph, malformed JSON, duplicate ID, missing dependency, cycle,
  excessive depth, excessive width, excessive node count, and direct-route
  tests.
- Property-style tests asserting every accepted plan is schedulable.

## Phase 3 — Bounded Async DAG Scheduler

- [x] Execute ready nodes with `asyncio` tasks under a semaphore; do not create
      OS threads for upstream I/O.
- [x] Schedule a node only after all required dependencies have terminal
      results.
- [x] Route every worker through existing model selection, key pool, retry,
      fallback, circuit breaker, concurrency, daily budget, and metrics paths.
- [x] Prevent workers and the synthesizer from selecting `parkour`.
- [x] Substitute dependency outputs through typed context fields rather than
      unrestricted string evaluation.
- [x] Truncate injected dependency content deterministically and record that
      truncation occurred.
- [x] Enforce total-call, deadline, cancellation, token, and cost limits during
      execution—not only during initial validation.

Verification:

- Sequential dependency and parallel-ready-node tests.
- Semaphore/concurrency test using deterministic barriers rather than timing
  alone.
- Deadline, cancellation, call-limit, token-limit, cost-limit, and recursion
  prevention tests.
- Integration tests showing workers use fallback and circuit-breaker behavior.

## Phase 4 — Failure Semantics and Synthesis

- [x] Classify nodes as required or optional in the validated schema.
- [x] Skip descendants of a failed required dependency.
- [x] Permit independent branches and optional-node descendants to continue
      when their inputs remain valid.
- [x] Synthesize from successful partial results when useful and clearly mark
      partial execution in trace metadata.
- [x] If conductor generation or validation fails, fall back to one ordinary
      routed completion.
- [x] If no useful result exists, return an OpenAI-shaped error with a stable
      PARKOUR error code.
- [x] Aggregate usage across conductor, workers, and synthesizer while retaining
      per-node internal accounting.

Verification:

- Required/optional failure matrix.
- Conductor fallback, partial synthesis, no-useful-result, and aggregate usage
  tests.
- Tests proving client errors do not trigger inappropriate worker fan-out.

## Phase 5 — Gateway Contract and Metadata

- [x] Intercept explicit `model: "parkour"` before ordinary upstream-model
      lookup while preserving normal explicit-model behavior.
- [x] Return `model: "parkour"` in the public completion; retain actual worker
      model IDs in PARKOUR trace data.
- [x] Add compact headers such as `X-Autoscale-Type: parkour`,
      `X-Agent-Count`, and a bounded opaque run/trace ID.
- [x] Do not serialize the full graph into HTTP headers.
- [x] Add an opt-in namespaced response extension for bounded graph summaries;
      default responses remain OpenAI-compatible.
- [x] Accept `stream=true` with bounded `parkour_event` progress chunks and a
      clean final answer stream; never describe buffered worker progress as
      true upstream token streaming.
- [x] Ensure tools and tool results have an explicit initial policy. Default:
      do not let workers execute tools; preserve ordinary non-PARKOUR tool
      behavior.

Verification:

- OpenAI response-shape and header-size tests.
- Unknown/disabled PARKOUR, progress-event streaming, explicit-model precedence,
  tools policy, and namespaced metadata tests.
- Compatibility smoke test using an OpenAI client.

## Phase 6 — Observability, Cost, and Operator UI

- [x] Record run count, outcome, duration, active runs, node count, worker-call
      count, concurrency, truncations, limit stops, partial runs, and failures.
- [x] Separate conductor, worker, synthesizer, and total token/cost figures.
- [x] Expose equivalent JSON and Prometheus signals.
- [x] Emit structured run/node events with request ID and opaque PARKOUR run ID;
      never log secrets or full user/worker prompts by default.
- [x] Add a bounded graph-summary view to `/explain`, the web dashboard, and the
      TUI using stored summary data—not oversized response headers.

Verification:

- Metrics snapshot/Prometheus parity tests.
- Cost reconciliation tests: sum of all internal calls equals PARKOUR total.
- Log redaction tests.
- Dashboard rendering tests for success, partial failure, and limit stop.

## Phase 7 — Hardening and Release

- [x] Run unit, integration, lint, type, and full regression suites.
- [x] Add adversarial conductor-output tests and fuzz graph validation.
- [x] Load-test bounded concurrency and prove no key-budget bypass.
- [x] Verify cancellation releases semaphores and active-run gauges.
- [x] Document expected latency/cost amplification and operational tuning.
- [x] Ship disabled by default, then enable in an opt-in preview release.
- [x] Capture SpecRegistry traceability evidence and run `finish_task` or
      `specreg comply` before commit.

Release gates:

- No implementation begins before published specs are synced.
- No unbounded graph, recursive PARKOUR selection, secret leakage, or budget
  bypass.
- Normal-model latency and behavior show no material regression while PARKOUR
  is disabled.
- Full test suite and objective SpecRegistry compliance pass.

## Phase 8 — Governed Research Lane and Built-In Web Search

Status: SHIPPED. Governed by `PARKOUR_RESEARCH.md` v1.0.0; implemented in
`nvidia_smartroute/parkour/research.py` (commit `9ea22b7`). Two follow-on
capabilities were subsequently specced and built toward Sakana Fugu parity:
the verify-and-refine loop (`PARKOUR_REFINEMENT.md`, `TODO-parkour-refinement.md`,
commit `8e71349`) and the multi-model ensemble panel (`PARKOUR_ENSEMBLE.md`,
`TODO-parkour-ensemble.md`).

Goal: restore PARKOUR's research usefulness without allowing arbitrary
client-supplied tool execution. PARKOUR should gain a server-owned, bounded web
research capability that workers can use under gateway policy, while continuing
to reject untrusted OpenAI `tools` payloads until a separate sandbox/security
contract exists.

- [x] Draft and submit a minor `PARKOUR.md` spec update, or a focused
      `PARKOUR_RESEARCH.md`, before implementation. The spec must distinguish
      built-in server-owned research tools from arbitrary client-provided tools.
- [x] Keep `tools` / `tool_choice` rejection for PARKOUR requests unless and
      until a reviewed arbitrary-tool sandbox contract is approved.
- [x] Add an opt-in setting such as `ENABLE_PARKOUR_RESEARCH=false` so web
      research remains disabled by default and can be rolled out separately
      from core PARKOUR.
- [x] Define a built-in `parkour_web_search` capability with a narrow interface:
      query string, optional domain filters, max result count, timeout, and
      citation/result snippets only.
- [x] Enforce hard limits for total searches per run, searches per node, query
      length, result count, fetched bytes, wall-clock time, and aggregate cost.
- [x] Route all research through a dedicated adapter that blocks local/private
      network targets, masks secrets, applies domain allow/block lists, and
      never exposes raw API keys or authorization headers to workers.
- [x] Require workers that use search to return source provenance/citations in
      a bounded structured field; synthesis must preserve cited claims where
      possible and mark uncited claims as model-derived.
- [x] Add streaming progress events for research activity, e.g.
      `research_query_started`, `research_query_completed`, and
      `research_query_failed`, without including full prompts or unbounded page
      text in the event payload.
- [x] Record telemetry for search count, failures, latency, bytes/results
      retained, domains contacted, truncations, and limit stops; expose JSON and
      Prometheus parity.
- [x] Add cache/deduplication for identical research queries within one run so
      parallel workers do not stampede the same provider.
- [x] Document privacy and freshness tradeoffs: research may send generated
      search queries to the configured provider, and output quality depends on
      provider coverage and recency.

Verification:

- Spec review and `specreg sync` before code changes.
- Unit tests for query validation, limit enforcement, redaction, provider
  errors, deduplication, and citation shaping.
- Gateway tests proving arbitrary client tools are still rejected while the
  built-in research lane works only when explicitly enabled.
- Security tests for SSRF/private-network blocking and secret/header exclusion.
- Observability tests for JSON/Prometheus parity and progress-event bounds.
- Regression tests proving PARKOUR without research, and ordinary non-PARKOUR
  routing, are unchanged.

## Deferred Beyond v1

- Recursive PARKOUR workers.
- Persistent/resumable graphs.
- Cross-process graph execution or Redis-backed scheduling.
- Arbitrary client-provided worker tool execution and sandboxed write/test/fix
  loops.
- Public retrieval of full graph traces.
