"""PARKOUR execution-plan schema and validation tests."""

import json

import pytest

from nvidia_smartroute.parkour import (
    PlanLimits,
    PlanValidationError,
    analyze_plan,
    build_direct_plan,
    parse_execution_plan,
    should_decompose,
)


@pytest.fixture
def limits():
    return PlanLimits(max_nodes=8, max_depth=4, max_width=3, max_prompt_chars=500)


def _task(node_id, dependencies=None, **overrides):
    task = {
        "id": node_id,
        "task_type": "reasoning",
        "role": "analyst",
        "system_prompt": "Analyze the assigned part.",
        "user_prompt": f"Work on {node_id}.",
        "dependencies": dependencies or [],
    }
    task.update(overrides)
    return task


def _plan(tasks, synthesis="Combine the results."):
    return {"tasks": tasks, "synthesis_instructions": synthesis}


def test_valid_plan_is_parsed_and_topologically_analyzed(limits):
    plan = parse_execution_plan(
        _plan([
            _task("research"),
            _task("review", ["research"]),
            _task("write", ["research"]),
            _task("final", ["review", "write"]),
        ]),
        limits,
    )
    analysis = analyze_plan(plan, limits)
    assert analysis.topological_order == ("research", "review", "write", "final")
    assert analysis.depth == 3
    assert analysis.width == 2
    assert plan.requires_synthesis is True


def test_native_json_and_fenced_json_have_identical_results(limits):
    payload = _plan([_task("one")])
    native = parse_execution_plan(payload, limits)
    encoded = parse_execution_plan(json.dumps(payload), limits)
    fenced = parse_execution_plan(f"planner notes\n```json\n{json.dumps(payload)}\n```", limits)
    assert native == encoded == fenced


@pytest.mark.parametrize("value", ["not json", "```json\n{broken}\n```", [], 3])
def test_malformed_plan_is_rejected(value, limits):
    with pytest.raises(PlanValidationError):
        parse_execution_plan(value, limits)


def test_duplicate_and_invalid_ids_are_rejected(limits):
    with pytest.raises(PlanValidationError, match="duplicate"):
        parse_execution_plan(_plan([_task("same"), _task("same")]), limits)
    with pytest.raises(PlanValidationError, match="invalid node ID"):
        parse_execution_plan(_plan([_task("Not Valid")]), limits)


def test_missing_self_and_repeated_dependencies_are_rejected(limits):
    with pytest.raises(PlanValidationError, match="missing"):
        parse_execution_plan(_plan([_task("one", ["absent"])]), limits)
    with pytest.raises(PlanValidationError, match="itself"):
        parse_execution_plan(_plan([_task("one", ["one"])]), limits)
    with pytest.raises(PlanValidationError, match="repeats"):
        parse_execution_plan(
            _plan([_task("one"), _task("two", ["one", "one"])]),
            limits,
        )


def test_cycle_is_rejected(limits):
    with pytest.raises(PlanValidationError, match="cycle"):
        parse_execution_plan(
            _plan([_task("one", ["two"]), _task("two", ["one"])]),
            limits,
        )


def test_node_depth_width_and_prompt_limits_are_rejected():
    with pytest.raises(PlanValidationError, match="nodes"):
        parse_execution_plan(
            _plan([_task("one"), _task("two")]),
            PlanLimits(max_nodes=1, max_depth=4, max_width=3, max_prompt_chars=500),
        )
    with pytest.raises(PlanValidationError, match="depth"):
        parse_execution_plan(
            _plan([_task("one"), _task("two", ["one"]), _task("three", ["two"])]),
            PlanLimits(max_nodes=8, max_depth=2, max_width=3, max_prompt_chars=500),
        )
    with pytest.raises(PlanValidationError, match="width"):
        parse_execution_plan(
            _plan([_task("one"), _task("two"), _task("three")]),
            PlanLimits(max_nodes=8, max_depth=4, max_width=2, max_prompt_chars=500),
        )
    with pytest.raises(PlanValidationError, match="characters"):
        parse_execution_plan(
            _plan([_task("one", user_prompt="x" * 500)]),
            PlanLimits(max_nodes=8, max_depth=4, max_width=3, max_prompt_chars=100),
        )


def test_unknown_task_type_and_extra_fields_are_rejected(limits):
    with pytest.raises(PlanValidationError, match="schema"):
        parse_execution_plan(
            _plan([_task("one", task_type="unsupported")]),
            limits,
        )
    with pytest.raises(PlanValidationError, match="schema"):
        parse_execution_plan(
            {**_plan([_task("one")]), "surprise": True},
            limits,
        )


def test_direct_route_is_single_node_without_synthesis(limits):
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    assert should_decompose(messages) is False
    plan = build_direct_plan(messages)
    analysis = analyze_plan(plan, limits)
    assert [task.id for task in plan.tasks] == ["direct"]
    assert plan.requires_synthesis is False
    assert analysis.depth == analysis.width == 1


def test_complexity_rule_is_deterministic():
    complex_request = [{"role": "user", "content": "Research and compare and then revise."}]
    long_request = [{"role": "user", "content": "x" * 601}]
    assert should_decompose(complex_request) is True
    assert should_decompose(complex_request) is True
    assert should_decompose(long_request) is True


def test_empty_direct_route_is_rejected():
    with pytest.raises(PlanValidationError, match="empty"):
        build_direct_plan([{"role": "user", "content": ""}])


@pytest.mark.parametrize("size", range(1, 8))
def test_every_accepted_chain_is_schedulable(size):
    tasks = [
        _task(f"node-{index}", [f"node-{index - 1}"] if index else [])
        for index in range(size)
    ]
    limits = PlanLimits(
        max_nodes=size,
        max_depth=size,
        max_width=1,
        max_prompt_chars=500,
    )
    plan = parse_execution_plan(_plan(tasks), limits)
    analysis = analyze_plan(plan, limits)
    assert len(analysis.topological_order) == size
    assert set(analysis.topological_order) == {task["id"] for task in tasks}
