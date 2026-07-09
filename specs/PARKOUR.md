# PARKOUR Virtual Multi-Agent Model

## Scope

The `parkour` virtual model exposed by the NVIDIA-SmartRoute gateway: explicit
selection through the OpenAI chat-completions API, conductor-generated execution
plans, bounded asynchronous DAG scheduling, routed worker calls, final
synthesis, failure behavior, resource limits, accounting, and graph
observability. Project-scoped for `github.com/joeldg/nvidiarouter`.

## Intent

Clients should be able to opt into a multi-model execution strategy using the
same `model` field they use for ordinary models. PARKOUR should decompose
complex requests, parallelize independent work, and synthesize a useful answer
without bypassing the gateway's routing, resilience, security, cost, or
observability controls. Ordinary requests must not pay a conductor-call penalty.

## Requirements

1. The gateway MUST expose the stable API model identifier `parkour` (display
   name `PARKOUR`) only when PARKOUR is enabled. PARKOUR is a virtual execution
   strategy, not an upstream NIM model, and MUST appear in the router source of
   `GET /v1/models` but MUST NOT appear in `?source=upstream`.
2. PARKOUR MUST run only when a client explicitly supplies `model: "parkour"`;
   it MUST be disabled by default and MUST NOT be selected by ordinary task
   classification, adaptive routing, fallback, or session affinity.
3. PARKOUR MUST use asynchronous I/O concurrency. A conductor MAY produce a
   single-node direct plan, or a validated directed acyclic graph of worker
   nodes and synthesis instructions. Independent ready nodes MAY execute
   concurrently under a configured semaphore.
4. Every execution plan MUST be schema-validated before scheduling. Validation
   MUST reject malformed plans, duplicate or invalid node IDs, missing or
   self-dependencies, cycles, unsupported task types, and graphs exceeding any
   configured node, depth, or width limit.
5. Every run MUST enforce hard limits for graph nodes, graph depth, concurrent
   workers, total upstream calls, wall-clock duration, per-node injected/output
   context, aggregate tokens, and estimated cost. Hitting a limit MUST stop or
   cancel affected work predictably and MUST be observable.
6. Every conductor, worker, and synthesizer upstream call MUST use the existing
   gateway paths for key rotation, outbound rate budgeting, retries, model
   fallback, circuit breaking, backpressure, daily spend control, cost
   recording, and metrics. PARKOUR MUST NOT maintain a parallel path that
   bypasses those controls.
7. PARKOUR workers and the synthesizer MUST NOT select `parkour`. Recursive
   PARKOUR execution is prohibited. Dependency output substitution MUST be
   typed and bounded; it MUST NOT evaluate arbitrary expressions, and
   truncation MUST be deterministic and recorded.
8. Execution MUST respect dependencies. A node MUST NOT start before all
   required dependencies reach a usable terminal result. Failure of a required
   dependency MUST skip its dependent branch; independent branches MAY
   continue. Optional failures MAY be omitted from synthesis.
9. If conductor generation or plan validation fails, PARKOUR MUST fall back to
   one ordinary routed completion. If workers partially fail but useful results
   remain, PARKOUR MAY synthesize a partial answer and MUST mark the run
   partial. If no useful result exists, it MUST return an OpenAI-shaped error
   with a stable PARKOUR error code.
10. Successful responses MUST use the normal OpenAI completion envelope and
    report the public model as `parkour`. Token usage and cost MUST aggregate
    conductor, worker, and synthesizer calls without double counting. Internal
    accounting MUST retain the actual model and role for every upstream call.
11. PARKOUR v1 MUST reject `stream: true` with an OpenAI-shaped 400 error.
    Buffered or synthetic output MUST NOT be represented as live upstream
    token streaming. Adding streaming requires a reviewed spec change.
12. Full execution graphs MUST NOT be serialized into HTTP headers. Responses
    MAY include an opt-in, bounded, namespaced graph-summary extension; headers
    MUST be limited to compact state such as autoscale type, node count, and an
    opaque run ID.
13. Runtime telemetry MUST include run count/outcome/duration, active runs,
    node and upstream-call counts, peak concurrency, partial runs, truncations,
    limit stops, failures, and conductor/worker/synthesizer/total token and cost
    figures. JSON metrics and Prometheus exposition MUST remain equivalent.
14. Structured logs and graph summaries MUST correlate request and opaque run
    IDs while excluding API keys, authorization data, and full prompt/output
    content by default. Existing inbound authentication and rate limiting apply
    to PARKOUR requests.
15. Worker tool execution is prohibited in PARKOUR v1. A request containing
    tools MUST either follow a documented no-execution passthrough policy or be
    rejected before graph execution; PARKOUR MUST never execute arbitrary
    client tools without a separate reviewed sandbox/security contract.

## Non-Goals

PARKOUR v1 does not provide true token streaming, recursive orchestration,
persistent or resumable graphs, cross-process scheduling, tool execution,
conversation storage, or public retrieval of full graph traces. It does not
replace the router's model-scoring algorithm or the gateway's existing
resilience and security controls.

## Acceptance Evidence

- Schema tests cover valid plans, malformed output, duplicate IDs, missing
  dependencies, cycles, unsupported task types, and every structural limit.
- Scheduler tests prove dependency ordering, concurrent execution of ready
  nodes, semaphore enforcement, cancellation cleanup, and recursion prevention.
- Integration tests prove worker calls use existing routing, key-pool, fallback,
  circuit-breaker, backpressure, budget, cost, and metrics paths.
- Failure tests cover conductor fallback, required and optional worker failure,
  partial synthesis, no-useful-result errors, deadlines, token/cost/call limits,
  and deterministic context truncation.
- Gateway tests cover explicit opt-in, disabled behavior, `/v1/models` virtual
  versus upstream sources, OpenAI response/error shapes, v1 streaming rejection,
  bounded metadata, and the tools policy.
- Accounting and observability tests reconcile per-call with aggregate tokens
  and cost, assert JSON/Prometheus parity, and verify secrets and full prompts
  are absent from logs.
- Regression and load tests show ordinary routing is unchanged while PARKOUR is
  disabled and PARKOUR cannot exceed configured concurrency or key budgets.

## Token Budget Class

Project contract. Load for any work on PARKOUR orchestration, virtual-model
registration, graph scheduling, synthesis, graph metadata, or PARKOUR UI.

## Related Specs

- `GATEWAY_API.md`
- `ROUTING.md`
- `COST.md`
- `OBSERVABILITY.md`
- `SECURITY_AND_KEYS.md`
- `MODEL_DISCOVERY.md`

## AI Agent Directives

Keep PARKOUR explicit, disabled by default, bounded, non-recursive, and routed
through existing controls. Validate the full graph before execution and enforce
limits again at runtime. Never describe buffered output as true streaming,
never place full graphs in headers, and never execute worker tools without a
separately reviewed security contract.

