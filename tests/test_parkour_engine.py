"""PARKOUR failure policy and synthesis tests."""

import asyncio

import pytest

from nvidia_smartroute.parkour import (
    DagScheduler,
    ExecutionPlan,
    ParkourEngine,
    ParkourExecutionError,
    PlanLimits,
    SchedulerLimits,
    SubtaskSpec,
    WorkerResult,
)
from nvidia_smartroute.routing.router import TaskType


def _node(node_id, deps=None, optional=False):
    return SubtaskSpec(
        id=node_id,
        task_type=TaskType.REASONING,
        role="worker",
        system_prompt="work",
        user_prompt=node_id,
        dependencies=deps or [],
        optional=optional,
    )


def _engine():
    scheduler = DagScheduler(SchedulerLimits(3, 10, 1, 100, 100, 100, 10))
    return ParkourEngine(scheduler, PlanLimits(10, 5, 5, 100))


async def _synth(nodes, instructions):
    return WorkerResult("|".join(node.output for node in nodes), "synth", 3, 0.3)


def test_required_failure_skips_descendants_but_independent_branch_continues():
    async def worker(task, contexts):
        if task.id == "bad":
            raise ValueError("boom")
        return WorkerResult(task.id, "model", 1, 0.1)

    plan = ExecutionPlan(
        tasks=[_node("bad"), _node("child", ["bad"]), _node("independent")],
        synthesis_instructions="combine",
    )
    result = asyncio.run(_engine().execute(plan, worker, _synth))
    assert result.scheduler.nodes["bad"].status == "failed"
    assert result.scheduler.nodes["child"].status == "skipped"
    assert result.scheduler.nodes["independent"].status == "succeeded"
    assert result.partial is True
    assert result.output == "independent"


def test_optional_failure_does_not_block_dependent():
    async def worker(task, contexts):
        if task.id == "optional":
            raise ValueError("optional failure")
        return WorkerResult(task.id, "model")

    plan = ExecutionPlan(
        tasks=[_node("optional", optional=True), _node("child", ["optional"])],
        synthesis_instructions="combine",
    )
    result = asyncio.run(_engine().execute(plan, worker, _synth))
    assert result.scheduler.nodes["child"].status == "succeeded"
    assert result.partial is True


def test_conductor_validation_failure_uses_direct_fallback():
    async def worker(task, contexts):
        return WorkerResult("fallback answer", "model", 2, 0.1)

    fallback = ExecutionPlan(tasks=[_node("direct")])
    result = asyncio.run(
        _engine().execute_conductor_output("not json", fallback, worker, _synth)
    )
    assert result.conductor_fallback is True
    assert result.output == "fallback answer"


def test_no_useful_result_has_stable_error_code():
    async def worker(task, contexts):
        raise ValueError("all failed")

    with pytest.raises(ParkourExecutionError) as caught:
        asyncio.run(
            _engine().execute(ExecutionPlan(tasks=[_node("bad")]), worker, _synth)
        )
    assert caught.value.code == "parkour_no_useful_result"
    assert caught.value.to_openai_error()["error"]["code"] == caught.value.code


def test_usage_aggregates_workers_and_synthesizer_without_losing_nodes():
    async def worker(task, contexts):
        return WorkerResult(task.id, f"model-{task.id}", 2, 0.2)

    plan = ExecutionPlan(
        tasks=[_node("one"), _node("two")],
        synthesis_instructions="combine",
    )
    conductor = WorkerResult("plan", "conductor", 4, 0.4)
    result = asyncio.run(_engine().execute(plan, worker, _synth, conductor=conductor))
    assert result.total_tokens == 11
    assert result.total_cost_usd == pytest.approx(1.1)
    assert result.conductor == conductor
    assert {node.model_id for node in result.scheduler.nodes.values()} == {
        "model-one", "model-two"
    }
