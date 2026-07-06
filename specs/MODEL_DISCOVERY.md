# Model Discovery

## Scope
Discovering which NIM models an account can serve and enriching them with
capability profiles: the `discover` and `benchmark` commands, catalog fetch,
servability probing, capability inference, the routability filter, persistence
to `discovered_models.json`, and the benchmark → routing feedback loop.
Project-scoped for `github.com/joeldg/nvidiarouter`.

## Intent
The router should be able to route across every model the key can actually
serve — with meaningful metadata (size, tasks, modality) — rather than a
hardcoded handful, and operators should be able to measure and prefer the
fastest capable models.

## Requirements
1. `discover` MUST fetch the NIM `/models` catalog and probe each model for
   servability with a minimal request, throttled to respect the per-key rate
   limit (default derived from `rate_limit_per_key`).
2. Discovery MUST exclude non-chat models from routing (embeddings, guardrails,
   safety/PII classifiers, reward models, image generators) via a routability
   filter.
3. Each servable model MUST be enriched into a capability profile: parameter
   count, supported tasks, vision/function-calling flags, context window, and a
   size-based latency prior. Curated metadata MAY override inference for named
   models; unknown IDs MUST fall back to pattern inference.
4. Capable general models (≥7B) MUST advertise code tasks so code requests have
   a home once discovery overrides the built-in defaults.
5. Discovered profiles MUST be persisted to `models_file` and loaded by the
   router on top of the built-in defaults without requiring a redeploy.
6. `benchmark` MUST measure the largest registered models directly (throttled,
   standalone) and rank them by success, p50 latency, and generation
   throughput; `--save` MUST write measured latency/throughput back into the
   model profiles so routing prefers benchmarked-fast models.
7. `discovered_models.json` is account-specific and MUST NOT be committed.

## Non-Goals
This spec does not define how models are scored/selected (`ROUTING.md`) or the
per-model metrics surface (`OBSERVABILITY.md`).

## Acceptance Evidence
- `tests/test_features.py` covers capability inference (curated + inferred),
  routability exclusion, serialize/deserialize round-trip, discovery filtering,
  registry loading of the discovered file, and benchmark feedback (`apply_benchmark`).
- Live discovery recorded 42 routable models (from 4) in PR evidence.

## Token Budget Class
Project contract. Load for discovery, capability inference, or benchmark work.

## Related Specs
- `ROUTING.md`
- `OBSERVABILITY.md`
- `SECURITY_AND_KEYS.md`

## AI Agent Directives
Probe with throttling; never mark a model unservable due to self-inflicted rate
limits. Keep account-specific artifacts out of version control.
