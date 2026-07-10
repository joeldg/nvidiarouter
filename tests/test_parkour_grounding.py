"""Regression tests for PARKOUR conversation grounding."""

import json

import pytest

from nvidia_smartroute.parkour.scheduler import NodeResult, WorkerResult


def test_direct_plan_labels_prior_assistant_output():
    from nvidia_smartroute.parkour.planning import build_direct_plan

    plan = build_direct_plan([
        {"role": "user", "content": "What color is the widget?"},
        {"role": "assistant", "content": "The widget is purple."},
        {"role": "user", "content": "That is wrong; do not assume purple."},
    ])

    prompt = plan.tasks[0].user_prompt
    assert "[ASSISTANT]\nThe widget is purple." in prompt
    assert "[USER]\nThat is wrong; do not assume purple." in prompt


def test_request_context_retains_roles_and_bounds():
    from nvidia_smartroute.parkour.service import _request_context

    context = _request_context([
        {"role": "assistant", "content": "unsupported claim"},
        {"role": "user", "content": "correction"},
    ], max_chars=200)

    decoded = [json.loads(line) for line in context.splitlines()]
    assert decoded == [
        {"role": "assistant", "content": "unsupported claim"},
        {"role": "user", "content": "correction"},
    ]
    assert len(context) <= 200


@pytest.mark.asyncio
async def test_direct_run_passes_original_role_sequence(monkeypatch):
    import nvidia_smartroute.parkour.service as service

    captured = {}

    async def worker(
        task,
        contexts,
        progress=None,
        research=None,
        request_messages=None,
        request_context=None,
        upstream_budget=None,
    ):
        captured["messages"] = request_messages
        captured["context"] = request_context
        return WorkerResult("corrected answer", "worker-model", 3, 0.01)

    messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "What color is the widget?"},
        {"role": "assistant", "content": "The widget is purple."},
        {"role": "user", "content": "That is wrong; reassess it."},
    ]
    monkeypatch.setattr(service, "routed_gateway_worker", worker)
    monkeypatch.setattr(service.settings, "enable_parkour_refinement", False)
    monkeypatch.setattr(service.settings, "enable_parkour_research", False)

    result = await service.run_parkour(messages)

    assert result.output == "corrected answer"
    assert captured["messages"] == messages
    assert '"role": "assistant"' in captured["context"]
    assert '"role": "user"' in captured["context"]


@pytest.mark.asyncio
async def test_synthesis_is_anchored_to_original_conversation(monkeypatch):
    import nvidia_smartroute.parkour.service as service

    captured = {}

    async def complete(
        model_id,
        messages,
        progress=None,
        role="model",
        upstream_budget=None,
    ):
        captured["messages"] = messages
        return WorkerResult("final", model_id, 2, 0.0)

    monkeypatch.setattr(service, "_routed_explicit_completion", complete)
    context = (
        '{"role": "assistant", "content": "incorrect"}\n'
        '{"role": "user", "content": "correct that"}'
    )
    node = NodeResult("fact", "worker claim", "worker-model", 1, 0.0, False)

    result = await service._synthesize([node], "Combine carefully.", context)

    assert result.output == "final"
    assert "assistant messages are fallible prior output" in captured["messages"][0]["content"]
    synthesis_input = captured["messages"][1]["content"]
    assert context in synthesis_input
    assert "Combine carefully." in synthesis_input
    assert "worker claim" in synthesis_input


def test_conductor_advertises_only_available_capabilities():
    from nvidia_smartroute.parkour.service import _conductor_prompt

    basic = _conductor_prompt(False, False)
    enabled = _conductor_prompt(True, True)

    assert ", research" not in basic
    assert ", panel" not in basic
    assert ", research" in enabled
    assert ", panel" in enabled
    assert "prior assistant claims are not user-provided facts" in enabled
