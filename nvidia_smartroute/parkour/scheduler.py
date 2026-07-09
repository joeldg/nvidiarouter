# @spec[PARKOUR.md#Requirements]
"""Bounded asyncio DAG scheduler for PARKOUR worker nodes."""

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .planning import ExecutionPlan, SubtaskSpec


class ParkourLimitError(RuntimeError):
    """Raised when a PARKOUR runtime limit is reached."""


@dataclass(frozen=True)
class DependencyContext:
    node_id: str
    output: str
    truncated: bool = False


@dataclass(frozen=True)
class WorkerResult:
    output: str
    model_id: str
    tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class NodeResult:
    node_id: str
    output: str
    model_id: str
    tokens: int
    cost_usd: float
    context_truncated: bool
    status: str = "succeeded"
    error: Optional[str] = None


@dataclass(frozen=True)
class SchedulerLimits:
    max_concurrency: int
    max_calls: int
    timeout_seconds: float
    max_context_chars: int
    max_output_chars: int
    max_tokens: int
    max_cost_usd: float

    @classmethod
    def from_settings(cls, settings):
        return cls(
            settings.parkour_max_concurrency,
            settings.parkour_max_calls,
            settings.parkour_timeout_seconds,
            settings.parkour_max_context_chars,
            settings.parkour_max_output_chars,
            settings.parkour_max_tokens,
            settings.parkour_max_cost_usd,
        )


@dataclass(frozen=True)
class SchedulerResult:
    nodes: Dict[str, NodeResult]
    total_calls: int
    total_tokens: int
    total_cost_usd: float
    peak_concurrency: int = 0

    @property
    def partial(self) -> bool:
        return any(node.status != "succeeded" for node in self.nodes.values())


WorkerCallable = Callable[
    [SubtaskSpec, List[DependencyContext]], Awaitable[WorkerResult]
]
ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]


# @spec[PARKOUR.md#Requirements]
class DagScheduler:
    """Execute validated DAG nodes when their dependencies are complete."""

    def __init__(
        self,
        limits: SchedulerLimits,
        progress: Optional[ProgressCallback] = None,
    ):
        self.limits = limits
        self._active = 0
        self._peak = 0
        self._progress = progress

    async def execute(
        self, plan: ExecutionPlan, worker: WorkerCallable
    ) -> SchedulerResult:
        self._active = self._peak = 0
        if len(plan.tasks) > self.limits.max_calls:
            raise ParkourLimitError("PARKOUR call limit is smaller than graph size")
        try:
            return await asyncio.wait_for(
                self._execute(plan, worker), timeout=self.limits.timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise ParkourLimitError("PARKOUR deadline exceeded") from exc

    async def _execute(
        self, plan: ExecutionPlan, worker: WorkerCallable
    ) -> SchedulerResult:
        pending = {task.id: task for task in plan.tasks}
        all_tasks = dict(pending)
        results: Dict[str, NodeResult] = {}
        semaphore = asyncio.Semaphore(self.limits.max_concurrency)
        total_tokens = 0
        total_cost = 0.0
        calls = 0

        while pending:
            ready = [
                task for task in pending.values()
                if all(dep in results for dep in task.dependencies)
            ]
            if not ready:
                raise RuntimeError("validated plan became unschedulable")
            if calls + len(ready) > self.limits.max_calls:
                raise ParkourLimitError("PARKOUR call limit exceeded")
            runnable = []
            for task in ready:
                blocked = any(
                    results[dep].status != "succeeded" and not all_tasks[dep].optional
                    for dep in task.dependencies
                )
                if blocked:
                    results[task.id] = NodeResult(
                        task.id, "", "", 0, 0.0, False, "skipped",
                        "required dependency failed",
                    )
                    await self._emit({
                        "type": "node_skipped",
                        "node_id": task.id,
                        "reason": "required_dependency_failed",
                    })
                    pending.pop(task.id)
                else:
                    runnable.append(task)
            calls += len(runnable)
            batch = await asyncio.gather(
                *(self._run_node(task, results, worker, semaphore) for task in runnable)
            )
            for result in batch:
                results[result.node_id] = result
                pending.pop(result.node_id)
                total_tokens += result.tokens
                total_cost += result.cost_usd
            if total_tokens > self.limits.max_tokens:
                raise ParkourLimitError("PARKOUR token limit exceeded")
            if total_cost > self.limits.max_cost_usd:
                raise ParkourLimitError("PARKOUR cost limit exceeded")

        return SchedulerResult(results, calls, total_tokens, total_cost, self._peak)

    async def _run_node(
        self,
        task: SubtaskSpec,
        results: Dict[str, NodeResult],
        worker: WorkerCallable,
        semaphore: asyncio.Semaphore,
    ) -> NodeResult:
        contexts = self._contexts(task, results)
        try:
            async with semaphore:
                self._active += 1
                self._peak = max(self._peak, self._active)
                try:
                    await self._emit({
                        "type": "node_started",
                        "node_id": task.id,
                        "task_type": task.task_type.value,
                    })
                    result = await worker(task, contexts)
                finally:
                    self._active -= 1
        except ParkourLimitError:
            raise
        except Exception as exc:
            failed = NodeResult(
                task.id, "", "", 0, 0.0,
                any(context.truncated for context in contexts),
                "failed", str(exc) or repr(exc),
            )
            await self._emit({
                "type": "node_failed",
                "node_id": task.id,
                "error": failed.error,
            })
            return failed
        if result.model_id == "parkour":
            raise ParkourLimitError("recursive PARKOUR worker selection is prohibited")
        output = result.output[: self.limits.max_output_chars]
        node = NodeResult(
            task.id,
            output,
            result.model_id,
            result.tokens,
            result.cost_usd,
            any(context.truncated for context in contexts),
        )
        await self._emit({
            "type": "node_completed",
            "node_id": task.id,
            "model": result.model_id,
            "tokens": result.tokens,
            "context_truncated": node.context_truncated,
            "output_truncated": len(output) < len(result.output),
        })
        return node

    async def _emit(self, event: Dict[str, Any]) -> None:
        if self._progress is not None:
            await self._progress(event)

    def _contexts(
        self, task: SubtaskSpec, results: Dict[str, NodeResult]
    ) -> List[DependencyContext]:
        remaining = self.limits.max_context_chars
        contexts = []
        for dependency in task.dependencies:
            if results[dependency].status != "succeeded":
                continue
            output = results[dependency].output
            retained = output[:remaining]
            contexts.append(
                DependencyContext(dependency, retained, len(retained) < len(output))
            )
            remaining = max(0, remaining - len(retained))
        return contexts


# @spec[PARKOUR.md#Requirements]
async def routed_gateway_worker(
    task: SubtaskSpec,
    contexts: List[DependencyContext],
    progress: Optional[ProgressCallback] = None,
) -> WorkerResult:
    """Run one node through the gateway's existing route/fallback/control path."""
    from ..cost import compute_cost
    from ..gateway.completion import complete_with_fallback
    from ..routing.router import router

    context_text = "\n\n".join(
        f"Dependency {item.node_id}:\n{item.output}" for item in contexts
    )
    messages = [{"role": "system", "content": task.system_prompt}]
    if context_text:
        messages.append({"role": "system", "content": context_text})
    messages.append({"role": "user", "content": task.user_prompt})
    decision = await router.route_request(messages)
    if not decision.selected_model or decision.selected_model.model_id == "parkour":
        raise ParkourLimitError("no non-PARKOUR worker model available")
    if progress is not None:
        await progress({
            "type": "worker_model_call",
            "node_id": task.id,
            "model": decision.selected_model.model_id,
            "task_type": decision.task_type.value,
        })
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
    return WorkerResult(content, used_model.model_id, tokens, cost)
