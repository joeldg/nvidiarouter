# @spec[PARKOUR_ENSEMBLE.md#Requirements]
"""Bounded multi-model ensemble panel for PARKOUR nodes.

A panel node fans one prompt across an explicit set of distinct models that run
concurrently, tolerates partial member failure, and combines the surviving
responses into one node result. Member and combination LLM calls are injected as
callables so the panel is pure and testable without the gateway; every real call
routes through existing gateway controls and never selects `parkour`.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple


class EnsembleError(RuntimeError):
    """Raised when every panel member fails and the node cannot proceed."""


@dataclass(frozen=True)
class MemberResult:
    """One panel member's outcome."""

    model_id: str
    content: str = ""
    tokens: int = 0
    cost_usd: float = 0.0
    ok: bool = True
    error: Optional[str] = None


@dataclass(frozen=True)
class PanelResult:
    """The combined outcome of one panel node."""

    content: str
    model_id: str
    tokens: int
    cost_usd: float
    members: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    combined: bool = False

    @property
    def successes(self) -> int:
        return sum(1 for m in self.members if m.get("ok"))

    @property
    def failures(self) -> int:
        return sum(1 for m in self.members if not m.get("ok"))


# A member runner takes a model id and returns that member's result. A combiner
# takes the surviving members and returns the merged (content, model, tokens,
# cost).
MemberRunner = Callable[[str], Awaitable[MemberResult]]
Combiner = Callable[[List[MemberResult]], Awaitable[MemberResult]]
ProgressCallback = Callable[[Dict[str, Any]], Any]


async def _emit(progress: Optional[ProgressCallback], event: Dict[str, Any]) -> None:
    if progress is None:
        return
    outcome = progress(event)
    if hasattr(outcome, "__await__"):
        await outcome


# @spec[PARKOUR_ENSEMBLE.md#Requirements]
async def run_panel(
    members: Sequence[str],
    run_member: MemberRunner,
    combine: Combiner,
    telemetry: Optional["EnsembleTelemetry"] = None,
    progress: Optional[ProgressCallback] = None,
    configured_size: Optional[int] = None,
) -> PanelResult:
    """Run panel members concurrently and combine the survivors.

    Proceeds on the successful members; fails only when every member fails. A
    single surviving member is returned directly without a combination call.
    """
    await _emit(progress, {"type": "panel_started", "size": len(members)})
    tasks = [
        asyncio.create_task(_guarded_member(run_member, model_id, progress))
        for model_id in members
    ]
    try:
        outcomes: List[MemberResult] = await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    summaries = tuple(
        {"model": m.model_id, "ok": m.ok, "tokens": m.tokens} for m in outcomes
    )
    successful = [m for m in outcomes if m.ok]
    member_tokens = sum(m.tokens for m in successful)
    member_cost = sum(m.cost_usd for m in successful)

    if not successful:
        if telemetry:
            telemetry.record_panel(
                len(members),
                0,
                len(outcomes),
                0,
                0.0,
                configured_size=configured_size,
            )
        raise EnsembleError("all ensemble panel members failed")

    if len(successful) == 1:
        only = successful[0]
        result = PanelResult(
            only.content, only.model_id, member_tokens, member_cost,
            summaries, combined=False,
        )
    else:
        merged = await combine(successful)
        result = PanelResult(
            merged.content, merged.model_id,
            member_tokens + merged.tokens, member_cost + merged.cost_usd,
            summaries, combined=True,
        )

    await _emit(progress, {
        "type": "panel_combined",
        "successes": result.successes,
        "failures": result.failures,
        "combined": result.combined,
    })
    if telemetry:
        telemetry.record_panel(
            len(members),
            len(successful),
            len(outcomes),
            result.tokens,
            result.cost_usd,
            distinct={m.model_id for m in successful},
            configured_size=configured_size,
        )
    return result


async def _guarded_member(
    run_member: MemberRunner, model_id: str, progress: Optional[ProgressCallback]
) -> MemberResult:
    """Run one member, converting any failure into a non-fatal MemberResult."""
    try:
        result = await run_member(model_id)
    except Exception as exc:  # a member failure must not fail the whole node
        if getattr(exc, "is_parkour_limit", False):
            raise
        await _emit(progress, {
            "type": "panel_member_completed", "model": model_id, "ok": False,
        })
        return MemberResult(model_id, ok=False, error=str(exc) or repr(exc))
    await _emit(progress, {
        "type": "panel_member_completed",
        "model": result.model_id,
        "ok": result.ok,
    })
    return result


# @spec[PARKOUR_ENSEMBLE.md#Requirements]
class EnsembleTelemetry:
    """Bounded in-process telemetry for the ensemble panel."""

    def __init__(self) -> None:
        self.panels = 0
        self.member_successes = 0
        self.member_failures = 0
        self.all_failed = 0
        self.configured_members = 0
        self.effective_members = 0
        self.added_tokens = 0
        self.added_cost_usd = 0.0
        self.distinct_models: set = set()

    def record_panel(
        self,
        size: int,
        successes: int,
        total: int,
        tokens: int,
        cost_usd: float,
        distinct: Optional[set] = None,
        configured_size: Optional[int] = None,
    ) -> None:
        self.panels += 1
        self.configured_members += configured_size or size
        self.effective_members += size
        self.member_successes += successes
        self.member_failures += total - successes
        if successes == 0:
            self.all_failed += 1
        self.added_tokens += tokens
        self.added_cost_usd += cost_usd
        if distinct:
            self.distinct_models.update(distinct)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "panels": self.panels,
            "member_successes": self.member_successes,
            "member_failures": self.member_failures,
            "all_failed": self.all_failed,
            "configured_members": self.configured_members,
            "effective_members": self.effective_members,
            "distinct_models": len(self.distinct_models),
            "added_tokens": self.added_tokens,
            "added_cost_usd": round(self.added_cost_usd, 8),
        }


ensemble_telemetry = EnsembleTelemetry()
