# @spec[PARKOUR_ENSEMBLE.md#Requirements]
"""Tests for the PARKOUR multi-model ensemble panel."""

import pytest

from nvidia_smartroute.config import Settings
from nvidia_smartroute.parkour.ensemble import (
    EnsembleError,
    EnsembleTelemetry,
    MemberResult,
    run_panel,
)
from nvidia_smartroute.parkour.planning import SubtaskSpec
from nvidia_smartroute.parkour.scheduler import (
    ParkourLimitError,
    SchedulerLimits,
    UpstreamBudget,
    routed_gateway_worker,
)
from nvidia_smartroute.prometheus import render_prometheus
from nvidia_smartroute.routing.router import (
    ModelCapability,
    RoutingDecision,
    TaskType,
)


# --- config / membership -----------------------------------------------------

def test_panel_dedup_excludes_parkour_and_caps():
    s = Settings(parkour_ensemble_models="a, b, b ,parkour, c, d", parkour_ensemble_max_size=3)
    assert s.parkour_ensemble_configured_panel == ["a", "b", "c", "d"]
    assert s.parkour_ensemble_panel == ["a", "b", "c"]


def test_panel_empty_and_single():
    assert Settings().parkour_ensemble_panel == []
    assert Settings(parkour_ensemble_models="solo").parkour_ensemble_panel == ["solo"]


def test_panel_only_parkour_yields_empty():
    assert Settings(parkour_ensemble_models="parkour,parkour").parkour_ensemble_panel == []


def test_ensemble_disabled_by_default():
    assert Settings(_env_file=None).enable_parkour_ensemble is False


def test_subtaskspec_panel_defaults_false():
    task = SubtaskSpec(id="n", task_type=TaskType.CHAT, role="r",
                       system_prompt="s", user_prompt="u")
    assert task.panel is False


@pytest.mark.asyncio
async def test_disabled_panel_is_identical_to_single_model_path(monkeypatch):
    import nvidia_smartroute.gateway.completion as completion
    import nvidia_smartroute.parkour.scheduler as sched
    import nvidia_smartroute.routing.router as routing
    from nvidia_smartroute.config import settings as cfg

    model = ModelCapability("m", "M", "test", "1", [TaskType.CHAT])
    routed_messages = []
    completed_messages = []

    async def route_request(messages):
        routed_messages.append(messages)
        return RoutingDecision("r", TaskType.CHAT, model, 1.0)

    async def complete_call(
        task_type, primary, messages, max_tokens, temperature, extra
    ):
        completed_messages.append(messages)
        return (
            {
                "choices": [{"message": {"content": "same"}}],
                "usage": {"total_tokens": 1},
            },
            model,
            False,
        )

    monkeypatch.setattr(cfg, "enable_parkour_ensemble", False)
    monkeypatch.setattr(routing.router, "route_request", route_request)
    monkeypatch.setattr(completion, "complete_with_fallback", complete_call)
    ordinary = SubtaskSpec(
        id="n", task_type=TaskType.CHAT, role="r",
        system_prompt="s", user_prompt="u", panel=False,
    )
    panel_flagged = ordinary.model_copy(update={"panel": True})

    ordinary_result = await sched.routed_gateway_worker(ordinary, [])
    disabled_result = await sched.routed_gateway_worker(panel_flagged, [])

    assert disabled_result == ordinary_result
    assert routed_messages[1] == routed_messages[0]
    assert completed_messages[1] == completed_messages[0]


# --- run_panel behavior ------------------------------------------------------

async def _combine(successful):
    return MemberResult("synth-model", "COMBINED", 5, 0.005, ok=True)


def _ok_runner(content_prefix="ans"):
    async def run_member(model_id):
        return MemberResult(model_id, f"{content_prefix}-{model_id}", 10, 0.01, ok=True)
    return run_member


@pytest.mark.asyncio
async def test_all_succeed_combines():
    tel = EnsembleTelemetry()
    r = await run_panel(["a", "b", "c"], _ok_runner(), _combine, tel)
    assert r.content == "COMBINED"
    assert r.combined is True
    assert r.successes == 3 and r.failures == 0
    assert r.tokens == 35  # 10*3 members + 5 combine
    assert r.cost_usd == pytest.approx(0.035)


@pytest.mark.asyncio
async def test_partial_failure_proceeds():
    async def run_member(model_id):
        if model_id == "b":
            raise RuntimeError("member down")
        return MemberResult(model_id, f"ans-{model_id}", 10, 0.01, ok=True)

    r = await run_panel(["a", "b", "c"], run_member, _combine, EnsembleTelemetry())
    assert r.successes == 2 and r.failures == 1
    assert r.combined is True


@pytest.mark.asyncio
async def test_single_survivor_no_combine():
    async def run_member(model_id):
        if model_id != "a":
            raise RuntimeError("down")
        return MemberResult("a", "only-a", 10, 0.01, ok=True)

    r = await run_panel(["a", "b"], run_member, _combine, EnsembleTelemetry())
    assert r.combined is False
    assert r.content == "only-a"
    assert r.tokens == 10  # no combine call


@pytest.mark.asyncio
async def test_all_members_fail_raises():
    async def run_member(model_id):
        raise RuntimeError("down")

    tel = EnsembleTelemetry()
    with pytest.raises(EnsembleError):
        await run_panel(["a", "b"], run_member, _combine, tel)
    assert tel.all_failed == 1


@pytest.mark.asyncio
async def test_telemetry_snapshot_shape():
    tel = EnsembleTelemetry()
    await run_panel(
        ["a", "b", "c"],
        _ok_runner(),
        _combine,
        tel,
        configured_size=5,
    )
    snap = tel.snapshot()
    assert snap["panels"] == 1
    assert snap["member_successes"] == 3
    assert snap["distinct_models"] == 3
    assert snap["configured_members"] == 5
    assert snap["effective_members"] == 3
    assert snap["added_tokens"] == 35
    assert set(snap) >= {
        "panels", "member_successes", "member_failures", "all_failed",
        "configured_members", "effective_members", "distinct_models",
        "added_tokens", "added_cost_usd",
    }

    text = render_prometheus({"parkour": {"ensemble": snap}})
    assert "nsr_parkour_ensemble_configured_members 5" in text
    assert "nsr_parkour_ensemble_effective_members 3" in text
    assert "nsr_parkour_ensemble_added_tokens 35" in text


@pytest.mark.asyncio
async def test_progress_events_bounded():
    events = []

    async def progress(e):
        events.append(e)

    await run_panel(["a", "b"], _ok_runner(), _combine, EnsembleTelemetry(), progress)
    types_seen = {e["type"] for e in events}
    assert {"panel_started", "panel_member_completed", "panel_combined"} <= types_seen
    for e in events:
        assert "content" not in e
        assert "prompt" not in e


# --- scheduler integration ---------------------------------------------------

@pytest.mark.asyncio
async def test_panel_node_runs_ensemble(monkeypatch):
    import nvidia_smartroute.parkour.scheduler as sched
    from nvidia_smartroute.config import settings as cfg

    async def fake_routed(messages, model_id, upstream_budget=None):
        if model_id in ("m1", "m2", "m3"):
            return (f"ans-{model_id}", model_id, 10, 0.01)
        return ("COMBINED", model_id, 5, 0.005)  # synthesizer combine

    monkeypatch.setattr(sched, "_routed_call", fake_routed)
    monkeypatch.setattr(cfg, "enable_parkour_ensemble", True)
    monkeypatch.setattr(cfg, "parkour_ensemble_models", "m1,m2,m3")

    task = SubtaskSpec(id="p", task_type=TaskType.CHAT, role="r",
                       system_prompt="s", user_prompt="u", panel=True)
    result = await routed_gateway_worker(task, [])
    assert result.output == "COMBINED"
    assert result.tokens == 35  # 3 members * 10 + 5 combine


@pytest.mark.asyncio
async def test_panel_ignored_when_fewer_than_two_members(monkeypatch):
    import nvidia_smartroute.parkour.scheduler as sched
    from nvidia_smartroute.config import settings as cfg
    from nvidia_smartroute.routing.router import router as real_router

    called = {"panel": False, "single": False}

    async def fake_routed(
        messages, model_id, upstream_budget=None
    ):  # would only be hit by the panel
        called["panel"] = True
        return ("x", model_id, 1, 0.0)

    class _Decision:
        selected_model = None  # forces the single-model path to raise cleanly
        task_type = TaskType.CHAT

    async def fake_route(messages, **kw):
        called["single"] = True
        return _Decision()

    monkeypatch.setattr(sched, "_routed_call", fake_routed)
    monkeypatch.setattr(real_router, "route_request", fake_route)
    monkeypatch.setattr(cfg, "enable_parkour_ensemble", True)
    monkeypatch.setattr(cfg, "parkour_ensemble_models", "solo")  # only one member

    task = SubtaskSpec(id="p", task_type=TaskType.CHAT, role="r",
                       system_prompt="s", user_prompt="u", panel=True)
    with pytest.raises(Exception):
        await routed_gateway_worker(task, [])
    # The panel was skipped (fewer than two members); single-model path was taken.
    assert called["single"] is True
    assert called["panel"] is False


@pytest.mark.asyncio
async def test_panel_calls_share_concurrency_and_call_limits(monkeypatch):
    import asyncio

    import nvidia_smartroute.parkour.scheduler as sched
    from nvidia_smartroute.config import settings as cfg

    active = 0
    peak = 0
    two_entered = asyncio.Event()
    release = asyncio.Event()

    limits = SchedulerLimits(2, 4, 5, 100, 100, 100, 10.0)
    budget = UpstreamBudget(limits)

    async def fake_routed(messages, model_id, upstream_budget=None):
        async def operation():
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                two_entered.set()
            await release.wait()
            active -= 1
            return f"ans-{model_id}", model_id, 1, 0.01

        return await upstream_budget.execute(
            operation, lambda result: (result[2], result[3])
        )

    monkeypatch.setattr(sched, "_routed_call", fake_routed)
    monkeypatch.setattr(cfg, "enable_parkour_ensemble", True)
    monkeypatch.setattr(cfg, "parkour_ensemble_models", "m1,m2,m3")
    task = SubtaskSpec(
        id="p",
        task_type=TaskType.CHAT,
        role="r",
        system_prompt="s",
        user_prompt="u",
        panel=True,
    )

    running = asyncio.create_task(
        routed_gateway_worker(task, [], upstream_budget=budget)
    )
    await asyncio.wait_for(two_entered.wait(), 0.5)
    assert peak == 2
    release.set()
    result = await running

    assert result.tokens == 4
    assert budget.calls == 4  # three members plus the combiner
    assert budget.peak_concurrency == 2


@pytest.mark.asyncio
async def test_panel_combiner_cannot_bypass_call_limit(monkeypatch):
    import nvidia_smartroute.parkour.scheduler as sched
    from nvidia_smartroute.config import settings as cfg

    budget = UpstreamBudget(
        SchedulerLimits(2, 2, 5, 100, 100, 100, 10.0)
    )

    async def fake_routed(messages, model_id, upstream_budget=None):
        async def operation():
            return f"ans-{model_id}", model_id, 1, 0.01

        return await upstream_budget.execute(
            operation, lambda result: (result[2], result[3])
        )

    monkeypatch.setattr(sched, "_routed_call", fake_routed)
    monkeypatch.setattr(cfg, "enable_parkour_ensemble", True)
    monkeypatch.setattr(cfg, "parkour_ensemble_models", "m1,m2")
    task = SubtaskSpec(
        id="p",
        task_type=TaskType.CHAT,
        role="r",
        system_prompt="s",
        user_prompt="u",
        panel=True,
    )

    with pytest.raises(ParkourLimitError, match="call limit"):
        await routed_gateway_worker(task, [], upstream_budget=budget)
