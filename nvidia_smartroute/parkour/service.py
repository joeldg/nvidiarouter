# @spec[PARKOUR.md#Requirements]
"""End-to-end PARKOUR orchestration using existing routed completion controls."""

import json
from dataclasses import replace
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..config import settings
from ..cost import compute_cost
from ..gateway.completion import complete_with_fallback
from ..routing.router import router
from .engine import EngineResult, ParkourEngine
from .planning import PlanLimits, build_direct_plan, should_decompose
from .refinement import (
    RefinementLimits,
    ReviseOutcome,
    VerifyOutcome,
    parse_verdict,
    refinement_telemetry,
    run_refinement,
)
from .research import ResearchSession, build_research_session
from .scheduler import (
    DagScheduler,
    NodeResult,
    ProgressCallback,
    SchedulerLimits,
    WorkerResult,
    routed_gateway_worker,
)

_VERIFIER_PROMPT = """You are a strict answer verifier. Given the user's request
and a candidate answer, judge the candidate. Return JSON only with keys: `score`
(a number from 0 to 1 for overall quality and correctness), `accept` (boolean),
and `feedback` (short, concrete guidance to improve the answer). Return JSON
only."""

_REVISER_PROMPT = """Revise the candidate answer to address the verifier
feedback while staying faithful to the original request. Return only the
improved answer, with no preamble."""

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


# @spec[PARKOUR_REFINEMENT.md#Requirements]
async def _routed_completion_with_cost(model_id: str, messages: list):
    """Route an explicit-model call through existing controls with real cost.

    Returns (content, model_id, tokens, cost_usd). Refuses to select `parkour`.
    """
    decision = await router.route_request(messages, model=model_id)
    if not decision.selected_model or decision.selected_model.model_id == "parkour":
        raise RuntimeError("no non-PARKOUR model available for refinement")
    data, used_model, _ = await complete_with_fallback(
        decision.task_type, decision.selected_model, messages, None, None, {}
    )
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    tokens = int(usage.get("total_tokens") or 0)
    cost = compute_cost(
        used_model,
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )
    content = data["choices"][0]["message"].get("content", "")
    return content, used_model.model_id, tokens, cost


# @spec[PARKOUR_REFINEMENT.md#Requirements]
async def _refine(result: EngineResult, request_text: str, progress) -> EngineResult:
    """Run the bounded verify-and-refine loop over the synthesized answer."""
    if not settings.enable_parkour_refinement:
        return result

    async def verify(candidate: str) -> VerifyOutcome:
        content, _model, tokens, cost = await _routed_completion_with_cost(
            settings.parkour_verifier_model,
            [
                {"role": "system", "content": _VERIFIER_PROMPT},
                {"role": "user",
                 "content": f"REQUEST:\n{request_text}\n\nCANDIDATE:\n{candidate}"},
            ],
        )
        # parse_verdict raises on malformed output; the loop treats that as a
        # verifier failure rather than an implicit accept.
        return VerifyOutcome(parse_verdict(content), tokens, cost)

    async def revise(candidate: str, feedback: str) -> ReviseOutcome:
        content, model_id, tokens, cost = await _routed_completion_with_cost(
            settings.parkour_synthesizer_model,
            [
                {"role": "system", "content": _REVISER_PROMPT},
                {"role": "user",
                 "content": (f"REQUEST:\n{request_text}\n\nCANDIDATE:\n{candidate}"
                             f"\n\nFEEDBACK:\n{feedback}")},
            ],
        )
        return ReviseOutcome(content, model_id, tokens, cost)

    refined = await run_refinement(
        result.output, verify, revise,
        RefinementLimits.from_settings(settings), refinement_telemetry, progress,
    )
    return replace(
        result,
        output=refined.output,
        total_tokens=result.total_tokens + refined.added_tokens,
        total_cost_usd=result.total_cost_usd + refined.added_cost_usd,
        refinement=refined.summary(),
    )


def _request_text(messages: list) -> str:
    """Flatten inbound messages into a bounded plain-text request for the loop."""
    parts = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(p for p in parts if p)[:8000]


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
        result = _with_research(result, research)
        return await _refine(result, _request_text(messages), progress)

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
    result = _with_research(result, research)
    return await _refine(result, _request_text(messages), progress)


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
