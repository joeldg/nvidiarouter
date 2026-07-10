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
from nvidia_smartroute.parkour.scheduler import routed_gateway_worker
from nvidia_smartroute.routing.router import TaskType


# --- config / membership -----------------------------------------------------

def test_panel_dedup_excludes_parkour_and_caps():
    s = Settings(parkour_ensemble_models="a, b, b ,parkour, c, d", parkour_ensemble_max_size=3)
    assert s.parkour_ensemble_panel == ["a", "b", "c"]


def test_panel_empty_and_single():
    assert Settings().parkour_ensemble_panel == []
    assert Settings(parkour_ensemble_models="solo").parkour_ensemble_panel == ["solo"]


def test_panel_only_parkour_yields_empty():
    assert Settings(parkour_ensemble_models="parkour,parkour").parkour_ensemble_panel == []


def test_ensemble_disabled_by_default():
    assert Settings().enable_parkour_ensemble is False


def test_subtaskspec_panel_defaults_false():
    task = SubtaskSpec(id="n", task_type=TaskType.CHAT, role="r",
                       system_prompt="s", user_prompt="u")
    assert task.panel is False


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
    await run_panel(["a", "b", "c"], _ok_runner(), _combine, tel)
    snap = tel.snapshot()
    assert snap["panels"] == 1
    assert snap["member_successes"] == 3
    assert snap["distinct_models"] == 3
    assert set(snap) >= {
        "panels", "member_successes", "member_failures", "all_failed",
        "distinct_models", "added_cost_usd",
    }


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

    async def fake_routed(messages, model_id):
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

    async def fake_routed(messages, model_id):  # would only be hit by the panel
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
