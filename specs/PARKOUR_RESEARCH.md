# PARKOUR Governed Research Lane

## Scope

The optional, server-owned web-research capability available to PARKOUR worker
nodes: the `parkour_web_search` built-in tool, its enablement flag, request and
result schema, the dedicated research adapter and its network-egress controls,
per-run and per-node resource limits, citation/provenance requirements, research
streaming events, telemetry, and per-run query deduplication. Project-scoped for
`github.com/joeldg/nvidiarouter`. This spec narrows `PARKOUR.md`,
`SECURITY_AND_KEYS.md`, and the global `GLOBAL_SECURITY.md` for research egress;
it does not relax any of them.

## Intent

PARKOUR should be able to ground worker reasoning in current external sources
without opening the gateway to arbitrary client-supplied tool execution. The
research lane must be a narrow, server-owned capability under gateway policy:
disabled by default, bounded in cost and blast radius, incapable of reaching
private/internal network targets, and unable to expose upstream credentials to
workers. Research adds latency, spend, and a data-egress path, so it must remain
an explicit operator opt-in that is separable from core PARKOUR.

## Requirements

1. The research lane MUST be disabled by default behind a dedicated setting
   (`ENABLE_PARKOUR_RESEARCH`, default `false`) that is independent of the
   PARKOUR enable flag. Core PARKOUR MUST function unchanged when research is
   disabled, and disabling research MUST remove `parkour_web_search` from every
   worker's available capabilities.
2. The gateway MUST continue to reject client-supplied `tools` / `tool_choice`
   payloads on PARKOUR requests under the existing `PARKOUR.md` req.15 policy.
   Enabling the research lane MUST NOT permit arbitrary client tool execution;
   `parkour_web_search` is server-owned and MUST NOT be definable, overridable,
   or parameterizable by the client request beyond the normal prompt.
3. `parkour_web_search` MUST expose a narrow, typed interface only: a query
   string, optional domain filters, and a maximum result count; the server MUST
   own timeout and all other execution parameters. It MUST return bounded
   structured results limited to source URL, title, and a truncated snippet per
   result. It MUST NOT return raw page bodies, request/response headers, cookies,
   or redirect chains to workers.
4. Every run MUST enforce hard limits for total searches per run, searches per
   node, query length, result count per search, total fetched bytes retained,
   research wall-clock time, and estimated research cost. Each limit MUST be
   configurable, MUST be enforced during execution (not only at validation), and
   hitting a limit MUST stop the affected research predictably and MUST be
   observable. Research spend MUST count against the same PARKOUR aggregate cost
   and budget controls defined in `PARKOUR.md` and `COST.md`; it MUST NOT create
   a parallel budget path.
5. All research egress MUST pass through one dedicated adapter that blocks
   requests to private, loopback, link-local, and otherwise non-public network
   targets (SSRF protection), including after DNS resolution and across
   redirects. The adapter MUST enforce configured domain allow/block lists and
   MUST reject disallowed targets before any network call. (Narrows
   `GLOBAL_SECURITY.md` egress rules and `SECURITY_AND_KEYS.md`.)
6. The research adapter MUST NOT expose the provider API key, `Authorization`
   header, or any gateway secret to workers, to synthesis, to logs, or to
   streaming events. Provider credentials MUST be loaded from the environment,
   masked wherever surfaced, and never returned in results. (Narrows
   `SECURITY_AND_KEYS.md` req.1, req.2.)
7. Workers that use search MUST return source provenance in a bounded, structured
   citation field (source URL plus the claim or snippet supported). Synthesis
   MUST preserve cited claims where possible and MUST mark uncited assertions as
   model-derived rather than sourced. A run MUST NOT present unsourced content as
   if it carried a citation.
8. When `stream: true`, the research lane MAY emit bounded, namespaced progress
   events (`research_query_started`, `research_query_completed`,
   `research_query_failed`) consistent with `PARKOUR.md` req.11. These events
   MUST NOT include full prompts, full page text, provider keys, authorization
   data, or unbounded result payloads.
9. Identical research queries within a single run MUST be deduplicated/cached so
   parallel workers do not stampede the provider with the same query. Cache
   scope MUST be bounded to one run and MUST NOT persist results across runs or
   across processes in v1.
10. Research telemetry MUST record search count, failures, latency,
    bytes/results retained, distinct domains contacted, truncations, and limit
    stops, and MUST separate research cost from conductor/worker/synthesizer
    figures while still rolling into the PARKOUR total. JSON metrics and
    Prometheus exposition MUST remain equivalent, consistent with
    `OBSERVABILITY.md` and `PARKOUR.md` req.13.
11. Documentation MUST disclose the privacy and freshness tradeoffs: enabling
    research sends generated queries to the configured provider, and output
    quality depends on provider coverage and recency. `.env.example` and
    `README.md` MUST document the flag and every research limit.

## Non-Goals

This spec does not authorize arbitrary or client-defined worker tools, sandboxed
write/test/fix loops, page crawling beyond bounded snippet retrieval, persistent
or cross-run research caches, cross-process research scheduling, or storage/
public retrieval of full fetched documents. It does not change core PARKOUR
orchestration, the router's scoring, or the existing inbound auth and rate
limiting.

## Acceptance Evidence

- Unit tests cover query validation, every limit's enforcement, secret/header
  redaction, provider-error handling, per-run deduplication, and citation
  shaping.
- Security tests prove SSRF/private-network blocking (including post-DNS and
  redirect cases), domain allow/block enforcement, and that no provider key or
  authorization header reaches workers, logs, or events.
- Gateway tests prove arbitrary client `tools`/`tool_choice` are still rejected
  while the built-in research lane works only when `ENABLE_PARKOUR_RESEARCH` is
  true.
- Observability tests assert JSON/Prometheus parity for research metrics and
  bounded research streaming events.
- Cost tests reconcile research spend into the PARKOUR aggregate and prove the
  shared budget is enforced.
- Regression tests prove PARKOUR without research, and ordinary non-PARKOUR
  routing, are byte-for-byte unchanged with the flag disabled.

## Token Budget Class

Project contract. Load for any work on the PARKOUR research lane, the research
adapter, research limits/telemetry, or the `parkour_web_search` capability.

## Related Specs

- `PARKOUR.md`
- `SECURITY_AND_KEYS.md`
- `GLOBAL_SECURITY.md`
- `COST.md`
- `OBSERVABILITY.md`
- `GATEWAY_API.md`

## AI Agent Directives

Keep research opt-in and disabled by default, server-owned, bounded, and routed
through the single egress adapter. Never let workers define tools, reach private
networks, or see provider secrets. Always require citations for sourced claims
and mark uncited content as model-derived. Do not begin implementation until
this spec is reviewed, published, and synced; propose changes for review and
never self-approve.
