# @spec[PARKOUR.md#Requirements]
"""End-to-end PARKOUR orchestration using existing routed completion controls."""

import json
from dataclasses import replace
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..config import settings
from ..gateway.completion import complete_with_fallback
from ..routing.router import router
from .engine import EngineResult, ParkourEngine
from .planning import PlanLimits, build_direct_plan, should_decompose
from .research import ResearchSession, build_research_session
from .scheduler import (
    DagScheduler,
    NodeResult,
    ProgressCallback,
    SchedulerLimits,
    WorkerResult,
    routed_gateway_worker,
)

_CONDUCTOR_PROMPT = """Create a JSON execution plan with keys `tasks` and
`synthesis_instructions`. Each task requires: id, task_type, role,
system_prompt, user_prompt, dependencies, optional. Use lowercase safe IDs,
only known task types, an acyclic dependency graph, and the fewest useful
tasks. Return JSON only."""


async def _emit(
    progress: Optional[ProgressCallback],
    event: Dict[str, Any],
) -> None:
    if progress is not None:
        await progress(event)


async def _routed_explicit_completion(
    model_id: str,
    messages: list,
    progress: Optional[ProgressCallback] = None,
    role: str = "model",
) -> WorkerResult:
    decision = await router.route_request(messages, model=model_id)
    if not decision.selected_model:
        raise RuntimeError("no model available")
    await _emit(progress, {
        "type": f"{role}_model_call",
        "model": decision.selected_model.model_id,
        "task_type": decision.task_type.value,
    })
    data, used_model, _ = await complete_with_fallback(
        decision.task_type, decision.selected_model, messages, None, None, {}
    )
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    result = WorkerResult(
        data["choices"][0]["message"].get("content", ""),
        used_model.model_id,
        int(usage.get("total_tokens") or 0),
        0.0,  # complete_with_fallback records authoritative per-call cost.
    )
    await _emit(progress, {
        "type": f"{role}_model_complete",
        "model": result.model_id,
        "tokens": result.tokens,
    })
    return result


async def _synthesize(
    nodes: List[NodeResult],
    instructions: str,
    progress: Optional[ProgressCallback] = None,
) -> WorkerResult:
    evidence = "\n\n".join(
        f"[{node.node_id} / {node.model_id}]\n{node.output}{_sources(node)}"
        for node in nodes
    )
    # @spec[PARKOUR_RESEARCH.md#Requirements]
    cited = any(node.citations for node in nodes)
    system = instructions
    if cited:
        system = (
            (instructions + "\n\n" if instructions else "")
            + "Preserve source citations where the evidence supports a claim, "
            "and clearly mark any claim not backed by a listed source as "
            "model-derived."
        )
    await _emit(progress, {"type": "synthesis_started", "node_count": len(nodes)})
    return await _routed_explicit_completion(
        settings.parkour_synthesizer_model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": evidence},
        ],
        progress,
        "synthesizer",
    )


# @spec[PARKOUR_RESEARCH.md#Requirements]
def _sources(node: NodeResult) -> str:
    """Render a node's bounded research citations for the synthesis prompt."""
    if not node.citations:
        return ""
    lines = "\n".join(
        f"- {c.get('title') or c.get('url')} ({c.get('url')})"
        for c in node.citations
    )
    return f"\nSources:\n{lines}"


# @spec[PARKOUR.md#Requirements]
async def run_parkour(
    messages: list,
    progress: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
) -> EngineResult:
    """Run a PARKOUR request through planning, workers, and synthesis."""
    await _emit(progress, {"type": "planning_started"})
    engine = ParkourEngine(
        DagScheduler(SchedulerLimits.from_settings(settings), progress),
        PlanLimits.from_settings(settings),
    )
    # @spec[PARKOUR_RESEARCH.md#Requirements]
    # None unless ENABLE_PARKOUR_RESEARCH is set and a provider is configured.
    research = build_research_session(settings)
    direct = build_direct_plan(messages)

    async def worker(task, contexts):
        return await routed_gateway_worker(task, contexts, progress, research)

    async def synthesizer(nodes, instructions):
        return await _synthesize(nodes, instructions, progress)

    if not should_decompose(messages):
        await _emit(progress, {"type": "direct_plan_selected"})
        result = await engine.execute(direct, worker, synthesizer)
        return _with_research(result, research)

    conductor = await _routed_explicit_completion(
        settings.parkour_conductor_model,
        [
            {"role": "system", "content": _CONDUCTOR_PROMPT},
            {"role": "user", "content": json.dumps(messages)},
        ],
        progress,
        "conductor",
    )
    await _emit(progress, {"type": "plan_generated"})

    result = await engine.execute_conductor_output(
        conductor.output,
        direct,
        worker,
        synthesizer,
        conductor,
    )
    return _with_research(result, research)


# @spec[PARKOUR_RESEARCH.md#Requirements]
def _with_research(
    result: EngineResult, research: Optional[ResearchSession]
) -> EngineResult:
    """Fold research spend into the PARKOUR total and attach a bounded summary."""
    if research is None or research.searches == 0:
        return result
    summary = {
        "searches": research.searches,
        "bytes_retained": research.bytes_retained,
        "cost_usd": round(research.cost_usd, 8),
    }
    return replace(
        result,
        total_cost_usd=result.total_cost_usd + research.cost_usd,
        research=summary,
    )
