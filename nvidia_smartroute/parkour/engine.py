# @spec[PARKOUR.md#Requirements]
"""PARKOUR failure policy, partial synthesis, and aggregate accounting."""

from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Union, Mapping, Any, Optional

from .planning import (
    ExecutionPlan,
    PlanLimits,
    PlanValidationError,
    parse_execution_plan,
)
from .scheduler import (
    DagScheduler,
    NodeResult,
    SchedulerResult,
    WorkerCallable,
    WorkerResult,
)


class ParkourExecutionError(RuntimeError):
    """Stable execution failure surfaced by the gateway in Phase 5."""

    code = "parkour_no_useful_result"

    def to_openai_error(self):
        return {
            "error": {
                "message": str(self),
                "type": "parkour_execution_error",
                "code": self.code,
            }
        }


SynthesizerCallable = Callable[
    [List[NodeResult], str], Awaitable[WorkerResult]
]


@dataclass(frozen=True)
class EngineResult:
    output: str
    scheduler: SchedulerResult
    synthesis: WorkerResult
    total_tokens: int
    total_cost_usd: float
    partial: bool
    conductor_fallback: bool
    conductor: Optional[WorkerResult] = None
    # Bounded research-lane summary; None when the research lane is inactive.
    # @spec[PARKOUR_RESEARCH.md#Requirements]
    research: Optional[Dict[str, Any]] = None


# @spec[PARKOUR.md#Requirements]
class ParkourEngine:
    """Apply PARKOUR's failure and synthesis policy to a validated graph."""

    def __init__(self, scheduler: DagScheduler, plan_limits: PlanLimits):
        self.scheduler = scheduler
        self.plan_limits = plan_limits

    async def execute(
        self,
        plan: ExecutionPlan,
        worker: WorkerCallable,
        synthesizer: SynthesizerCallable,
        conductor_fallback: bool = False,
        conductor: Optional[WorkerResult] = None,
    ) -> EngineResult:
        scheduled = await self.scheduler.execute(plan, worker)
        useful = [
            node for node in scheduled.nodes.values()
            if node.status == "succeeded" and bool(node.output)
        ]
        if not useful:
            raise ParkourExecutionError("PARKOUR produced no useful worker result")

        synthesis = await self._synthesize(plan, useful, synthesizer)
        return EngineResult(
            synthesis.output,
            scheduled,
            synthesis,
            scheduled.total_tokens + synthesis.tokens + (conductor.tokens if conductor else 0),
            scheduled.total_cost_usd + synthesis.cost_usd
            + (conductor.cost_usd if conductor else 0.0),
            scheduled.partial,
            conductor_fallback,
            conductor,
        )

    async def execute_conductor_output(
        self,
        conductor_output: Union[str, Mapping[str, Any], ExecutionPlan],
        fallback_plan: ExecutionPlan,
        worker: WorkerCallable,
        synthesizer: SynthesizerCallable,
        conductor: Optional[WorkerResult] = None,
    ) -> EngineResult:
        """Validate conductor output, falling back to one ordinary routed node."""
        fallback = False
        try:
            plan = parse_execution_plan(conductor_output, self.plan_limits)
        except PlanValidationError:
            plan = fallback_plan
            fallback = True
        return await self.execute(plan, worker, synthesizer, fallback, conductor)

    @staticmethod
    async def _synthesize(
        plan: ExecutionPlan,
        useful: List[NodeResult],
        synthesizer: SynthesizerCallable,
    ) -> WorkerResult:
        if plan.requires_synthesis:
            return await synthesizer(useful, plan.synthesis_instructions or "")
        first = useful[0]
        return WorkerResult(first.output, first.model_id)
