"""Bounded PARKOUR DAG scheduler tests."""

import asyncio

import pytest

from nvidia_smartroute.parkour import (
    DagScheduler,
    ExecutionPlan,
    ParkourLimitError,
    SchedulerLimits,
    SubtaskSpec,
    WorkerResult,
)
from nvidia_smartroute.routing.router import TaskType
from nvidia_smartroute.routing.router import ModelCapability, RoutingDecision


def _node(node_id, dependencies=None):
    return SubtaskSpec(
        id=node_id,
        task_type=TaskType.REASONING,
        role="worker",
        system_prompt="work",
        user_prompt=node_id,
        dependencies=dependencies or [],
    )


def _limits(**changes):
    values = dict(
        max_concurrency=2,
        max_calls=10,
        timeout_seconds=1,
        max_context_chars=8,
        max_output_chars=20,
        max_tokens=100,
        max_cost_usd=1,
    )
    values.update(changes)
    return SchedulerLimits(**values)


def test_dependencies_execute_in_order_and_context_is_typed():
    seen = []
    progress = []

    async def worker(task, contexts):
        seen.append((task.id, [item.node_id for item in contexts]))
        return WorkerResult(task.id * 5, "model", 2, 0.01)

    async def emit(event):
        progress.append(event)

    plan = ExecutionPlan(tasks=[_node("first"), _node("second", ["first"])])
    result = asyncio.run(DagScheduler(_limits(), emit).execute(plan, worker))
    assert seen == [("first", []), ("second", ["first"])]
    assert result.total_calls == 2 and result.total_tokens == 4
    assert result.nodes["second"].context_truncated is True
    assert {"type": "node_started", "node_id": "first",
            "task_type": "reasoning"} in progress


def test_ready_nodes_run_concurrently_under_semaphore():
    async def run():
        active = 0
        peak = 0
        entered = asyncio.Event()
        release = asyncio.Event()

        async def worker(task, contexts):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                entered.set()
            await release.wait()
            active -= 1
            return WorkerResult(task.id, "model")

        plan = ExecutionPlan(tasks=[_node("one"), _node("two"), _node("three")])
        task = asyncio.create_task(DagScheduler(_limits()).execute(plan, worker))
        await asyncio.wait_for(entered.wait(), 0.5)
        assert peak == 2
        release.set()
        await task

    asyncio.run(run())


@pytest.mark.parametrize(
    "limits,result,match",
    [
        (_limits(max_calls=1), WorkerResult("x", "model"), "call"),
        (_limits(max_tokens=1), WorkerResult("x", "model", tokens=2), "token"),
        (_limits(max_cost_usd=0.01), WorkerResult("x", "model", cost_usd=0.02), "cost"),
    ],
)
def test_runtime_limits(limits, result, match):
    async def worker(task, contexts):
        return result

    plan = ExecutionPlan(tasks=[_node("one"), _node("two", ["one"])])
    with pytest.raises(ParkourLimitError, match=match):
        asyncio.run(DagScheduler(limits).execute(plan, worker))


def test_deadline_cancels_worker():
    cancelled = False

    async def worker(task, contexts):
        nonlocal cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled = True
            raise

    with pytest.raises(ParkourLimitError, match="deadline"):
        asyncio.run(
            DagScheduler(_limits(timeout_seconds=0.01)).execute(
                ExecutionPlan(tasks=[_node("one")]), worker
            )
        )
    assert cancelled is True


def test_recursive_worker_model_is_rejected():
    async def worker(task, contexts):
        return WorkerResult("nope", "parkour")

    with pytest.raises(ParkourLimitError, match="recursive"):
        asyncio.run(
            DagScheduler(_limits()).execute(
                ExecutionPlan(tasks=[_node("one")]), worker
            )
        )


def test_gateway_worker_uses_router_and_existing_fallback_path(monkeypatch):
    import nvidia_smartroute.gateway.completion as completion
    import nvidia_smartroute.parkour.scheduler as scheduler
    import nvidia_smartroute.routing.router as routing

    model = ModelCapability(
        model_id="worker-model",
        name="Worker",
        provider="test",
        version="1",
        supported_tasks=[TaskType.REASONING],
    )
    calls = []

    async def route_request(messages):
        calls.append("route")
        return RoutingDecision("r", TaskType.REASONING, model, 1.0)

    async def complete(task_type, primary, messages, max_tokens, temperature, extra):
        calls.append("fallback")
        return (
            {
                "choices": [{"message": {"content": "done"}}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            },
            model,
            False,
        )

    monkeypatch.setattr(routing.router, "route_request", route_request)
    monkeypatch.setattr(completion, "complete_with_fallback", complete)
    result = asyncio.run(scheduler.routed_gateway_worker(_node("one"), []))
    assert calls == ["route", "fallback"]
    assert result.output == "done"
    assert result.model_id == "worker-model"
    assert result.tokens == 5
