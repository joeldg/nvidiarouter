# @spec[PARKOUR.md#Requirements]
"""End-to-end PARKOUR orchestration using existing routed completion controls."""

import json
from typing import List

from ..config import settings
from ..gateway.completion import complete_with_fallback
from ..routing.router import router
from .engine import EngineResult, ParkourEngine
from .planning import PlanLimits, build_direct_plan, should_decompose
from .scheduler import (
    DagScheduler,
    NodeResult,
    SchedulerLimits,
    WorkerResult,
    routed_gateway_worker,
)

_CONDUCTOR_PROMPT = """Create a JSON execution plan with keys `tasks` and
`synthesis_instructions`. Each task requires: id, task_type, role,
system_prompt, user_prompt, dependencies, optional. Use lowercase safe IDs,
only known task types, an acyclic dependency graph, and the fewest useful
tasks. Return JSON only."""


async def _routed_explicit_completion(model_id: str, messages: list) -> WorkerResult:
    decision = await router.route_request(messages, model=model_id)
    if not decision.selected_model:
        raise RuntimeError("no model available")
    data, used_model, _ = await complete_with_fallback(
        decision.task_type, decision.selected_model, messages, None, None, {}
    )
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    return WorkerResult(
        data["choices"][0]["message"].get("content", ""),
        used_model.model_id,
        int(usage.get("total_tokens") or 0),
        0.0,  # complete_with_fallback records authoritative per-call cost.
    )


async def _synthesize(nodes: List[NodeResult], instructions: str) -> WorkerResult:
    evidence = "\n\n".join(
        f"[{node.node_id} / {node.model_id}]\n{node.output}" for node in nodes
    )
    return await _routed_explicit_completion(
        settings.parkour_synthesizer_model,
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": evidence},
        ],
    )


# @spec[PARKOUR.md#Requirements]
async def run_parkour(messages: list) -> EngineResult:
    """Run a PARKOUR request through planning, workers, and synthesis."""
    engine = ParkourEngine(
        DagScheduler(SchedulerLimits.from_settings(settings)),
        PlanLimits.from_settings(settings),
    )
    direct = build_direct_plan(messages)
    if not should_decompose(messages):
        return await engine.execute(direct, routed_gateway_worker, _synthesize)

    conductor = await _routed_explicit_completion(
        settings.parkour_conductor_model,
        [
            {"role": "system", "content": _CONDUCTOR_PROMPT},
            {"role": "user", "content": json.dumps(messages)},
        ],
    )
    return await engine.execute_conductor_output(
        conductor.output,
        direct,
        routed_gateway_worker,
        _synthesize,
        conductor,
    )
