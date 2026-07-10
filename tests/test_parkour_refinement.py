# @spec[PARKOUR_REFINEMENT.md#Requirements]
"""Tests for the PARKOUR verifier + iterative refinement loop."""

import pytest

from nvidia_smartroute.parkour.refinement import (
    RefinementError,
    RefinementLimits,
    RefinementTelemetry,
    ReviseOutcome,
    Verdict,
    VerifyOutcome,
    parse_verdict,
    run_refinement,
)


def _limits(**over):
    base = dict(
        max_iterations=3, max_verifier_calls=6, max_revision_calls=3,
        timeout_seconds=60.0, max_added_tokens=1_000_000, max_added_cost_usd=100.0,
        accept_threshold=0.8, min_improvement=0.02, feedback_chars=500,
    )
    base.update(over)
    return RefinementLimits(**base)


def _verifier(scores, accept_when=None):
    """Return a verify callable yielding the given scores in order."""
    it = iter(scores)

    async def verify(candidate):
        score = next(it)
        accept = (accept_when is not None and score >= accept_when)
        return VerifyOutcome(Verdict(score=score, accept=accept), tokens=10, cost_usd=0.01)

    return verify


async def _revise(candidate, feedback):
    return ReviseOutcome(candidate + "+", "reviser-model", tokens=5, cost_usd=0.005)


# --- verdict parsing ---------------------------------------------------------

def test_parse_verdict_valid_and_fenced():
    assert parse_verdict('{"score": 0.5, "accept": false}').score == 0.5
    v = parse_verdict('noise ```json\n{"score": 0.7, "accept": true}\n``` tail')
    assert v.accept is True and v.score == 0.7


@pytest.mark.parametrize("bad", [
    "not json at all",
    '{"accept": true}',              # missing score
    '{"score": 2.0, "accept": true}',  # out of range
    '{"score": -0.1, "accept": false}',
    '{"score": 0.5, "accept": true, "extra": 1}',  # extra=forbid
])
def test_parse_verdict_malformed_raises_not_accept(bad):
    with pytest.raises(RefinementError):
        parse_verdict(bad)


# --- termination conditions --------------------------------------------------

@pytest.mark.asyncio
async def test_accept_on_first_verification():
    r = await run_refinement("ans", _verifier([0.9], accept_when=0.8), _revise, _limits())
    assert r.stop_reason == "accepted"
    assert r.accepted and r.verified
    assert r.iterations == 0 and r.revision_calls == 0
    assert r.output == "ans"


@pytest.mark.asyncio
async def test_improve_then_accept_returns_revised():
    r = await run_refinement(
        "ans", _verifier([0.5, 0.85], accept_when=0.8), _revise, _limits()
    )
    assert r.stop_reason == "accepted"
    assert r.best_score == 0.85
    assert r.iterations == 1
    assert r.output == "ans+"


@pytest.mark.asyncio
async def test_accept_flag_below_threshold_cannot_reward_hack():
    r = await run_refinement(
        "ans",
        _verifier([0.2, 0.9], accept_when=0.0),
        _revise,
        _limits(),
    )

    assert r.accepted is True
    assert r.verifier_calls == 2
    assert r.output == "ans+"


@pytest.mark.asyncio
async def test_oscillating_scores_return_best_observed_candidate():
    r = await run_refinement(
        "initial",
        _verifier([0.4, 0.7, 0.5]),
        _revise,
        _limits(min_improvement=0.0),
    )

    assert r.stop_reason == "no_improvement"
    assert r.best_score == 0.7
    assert r.output == "initial+"


@pytest.mark.asyncio
async def test_no_improvement_stops():
    # 0.50 -> 0.505 is below the 0.02 margin.
    r = await run_refinement(
        "ans", _verifier([0.50, 0.505]), _revise, _limits(min_improvement=0.02)
    )
    assert r.stop_reason == "no_improvement"
    assert r.best_score == 0.505


@pytest.mark.asyncio
async def test_max_iterations_stop():
    r = await run_refinement(
        "ans", _verifier([0.1, 0.3, 0.5, 0.7]), _revise,
        _limits(max_iterations=2, min_improvement=0.0),
    )
    assert r.stop_reason == "max_iterations"
    assert r.iterations == 2


@pytest.mark.asyncio
async def test_verifier_call_limit_stop():
    r = await run_refinement(
        "ans", _verifier([0.1, 0.2, 0.3, 0.4, 0.5]), _revise,
        _limits(max_verifier_calls=2, max_iterations=10, min_improvement=0.0),
    )
    assert r.stop_reason == "resource_limit"
    assert r.verifier_calls == 2


@pytest.mark.asyncio
async def test_added_token_limit_stop():
    r = await run_refinement(
        "ans", _verifier([0.1, 0.2, 0.3, 0.4]), _revise,
        _limits(max_added_tokens=12, min_improvement=0.0, max_iterations=10),
    )
    assert r.stop_reason == "resource_limit"


@pytest.mark.asyncio
async def test_wall_clock_stop():
    # Clock jumps past the budget after the run starts, so the first loop check
    # trips the wall-clock guard before any verifier call.
    ticks = iter([0.0, 100.0, 100.0])

    r = await run_refinement(
        "ans", _verifier([0.1]), _revise, _limits(timeout_seconds=5.0),
        clock=lambda: next(ticks),
    )
    assert r.stop_reason == "resource_limit"
    assert r.verifier_calls == 0
    assert r.output == "ans"


# --- best candidate & failure ------------------------------------------------

@pytest.mark.asyncio
async def test_best_candidate_kept_when_revision_is_worse():
    # 0.7 then a worse 0.3; best (earlier) answer must win.
    r = await run_refinement(
        "good", _verifier([0.7, 0.3]), _revise, _limits(min_improvement=0.0)
    )
    assert r.output == "good"
    assert r.best_score == 0.7


@pytest.mark.asyncio
async def test_ties_resolve_to_earlier_candidate():
    r = await run_refinement(
        "first", _verifier([0.6, 0.6]), _revise, _limits(min_improvement=0.0)
    )
    # Equal score -> earlier candidate retained (strictly-greater replaces).
    assert r.output == "first"


@pytest.mark.asyncio
async def test_verifier_failure_is_non_fatal():
    async def boom(candidate):
        raise RuntimeError("verifier down")

    tel = RefinementTelemetry()
    r = await run_refinement("init", boom, _revise, _limits(), telemetry=tel)
    assert r.stop_reason == "verifier_failed"
    assert r.output == "init"
    assert r.verified is False
    assert tel.verifier_failures == 1


@pytest.mark.asyncio
async def test_malformed_verdict_is_verifier_failure_not_accept():
    async def bad_verify(candidate):
        # A verify callable that emits a malformed verdict via parse_verdict.
        return VerifyOutcome(parse_verdict('{"accept": true}'))

    r = await run_refinement("init", bad_verify, _revise, _limits())
    assert r.stop_reason == "verifier_failed"
    assert r.accepted is False
    assert r.output == "init"


@pytest.mark.asyncio
async def test_revision_failure_returns_best_so_far():
    async def revise_boom(candidate, feedback):
        raise RuntimeError("reviser down")

    # First verify scores 0.5 (not accepted), then revision fails.
    r = await run_refinement("ans", _verifier([0.5]), revise_boom, _limits())
    assert r.stop_reason == "revision_failed"
    assert r.output == "ans"
    assert r.best_score == 0.5


@pytest.mark.asyncio
async def test_shared_run_limit_stops_refinement_as_resource_limit():
    class SharedLimitError(RuntimeError):
        is_parkour_limit = True

    async def limited_verify(candidate):
        raise SharedLimitError("run call limit")

    r = await run_refinement("usable", limited_verify, _revise, _limits())

    assert r.stop_reason == "resource_limit"
    assert r.output == "usable"
    assert r.verified is False


# --- telemetry & accounting --------------------------------------------------

@pytest.mark.asyncio
async def test_added_tokens_and_cost_accumulate():
    r = await run_refinement(
        "ans", _verifier([0.5, 0.9], accept_when=0.8), _revise, _limits()
    )
    # verify(10)+revise(5)+verify(10) tokens; verify(0.01)*2 + revise(0.005) cost
    assert r.added_tokens == 25
    assert r.added_cost_usd == pytest.approx(0.025)


@pytest.mark.asyncio
async def test_telemetry_snapshot_shape():
    tel = RefinementTelemetry()
    await run_refinement("ans", _verifier([0.9], accept_when=0.8), _revise, _limits(), telemetry=tel)
    await run_refinement("ans", _verifier([0.5, 0.505]), _revise, _limits(), telemetry=tel)
    snap = tel.snapshot()
    assert snap["loops"] == 2
    assert snap["accepts"] == 1
    assert snap["no_improvement_stops"] == 1
    assert set(snap) >= {
        "loops", "iterations", "accepts", "rejects", "no_improvement_stops",
        "limit_stops", "verifier_failures", "avg_returned_score",
        "added_tokens", "added_cost_usd",
    }


# --- progress events ---------------------------------------------------------

@pytest.mark.asyncio
async def test_progress_events_bounded():
    events = []

    async def progress(e):
        events.append(e)

    await run_refinement(
        "ans", _verifier([0.5, 0.9], accept_when=0.8), _revise, _limits(),
        progress=progress,
    )
    types_seen = {e["type"] for e in events}
    assert {
        "verification_started",
        "verification_completed",
        "revision_started",
        "revision_completed",
        "refinement_stopped",
    } <= types_seen
    # No full candidate text leaks into events.
    for e in events:
        assert "candidate" not in e
        assert "ans" not in str(e.get("content", ""))


# --- gateway: disabled by default --------------------------------------------

def test_refinement_disabled_by_default_leaves_trace_absent(monkeypatch):
    from fastapi.testclient import TestClient
    import nvidia_smartroute.gateway.server as srv
    from nvidia_smartroute.parkour import (
        EngineResult, NodeResult, SchedulerResult, WorkerResult,
    )

    node = NodeResult("one", "answer", "worker-model", 5, 0.1, False)
    scheduled = SchedulerResult({"one": node}, 1, 5, 0.1)
    synthesis = WorkerResult("final", "synth-model", 3, 0.2)
    result = EngineResult("final", scheduled, synthesis, 8, 0.3, False, False)

    async def run(messages, progress=None):
        return result

    monkeypatch.setattr(srv.settings, "enable_parkour", True)
    monkeypatch.setattr(srv.settings, "enable_parkour_refinement", False)
    monkeypatch.setattr(srv, "run_parkour", run)
    client = TestClient(srv.app)
    response = client.post("/v1/chat/completions", json={
        "model": "parkour", "parkour_trace": True,
        "messages": [{"role": "user", "content": "do work"}],
    })
    assert response.status_code == 200
    assert "refinement" not in response.json()["parkour"]


@pytest.mark.asyncio
async def test_refinement_disabled_returns_identical_result(monkeypatch):
    import nvidia_smartroute.parkour.service as service
    from nvidia_smartroute.parkour import (
        EngineResult,
        NodeResult,
        SchedulerResult,
        WorkerResult,
    )

    node = NodeResult("one", "answer", "worker", 5, 0.1, False)
    scheduled = SchedulerResult({"one": node}, 1, 5, 0.1)
    synthesis = WorkerResult("answer", "worker")
    result = EngineResult(
        "answer", scheduled, synthesis, 5, 0.1, False, False
    )
    monkeypatch.setattr(service.settings, "enable_parkour_refinement", False)

    returned = await service._refine(result, "request", None)

    assert returned is result
