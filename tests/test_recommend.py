# @spec[RECOMMENDATION.md#Acceptance Evidence]
"""
Tests for the model recommendation advisor, mapped to RECOMMENDATION.md.
"""

import pytest
from fastapi.testclient import TestClient

import nvidia_smartroute.metrics as M
from nvidia_smartroute.recommend import recommend_all, recommend_for, is_task
from nvidia_smartroute.routing.router import ModelCapability, ModelRegistry, TaskType


def _model(mid, tasks, quality=0.85, latency=500, params=0.0, tps=0.0, cost=0.0):
    return ModelCapability(
        model_id=mid, name=mid, provider="x", version="1",
        supported_tasks=tasks, quality_score=quality, reliability_score=0.9,
        latency_ms=latency, parameters_b=params, throughput_tps=tps,
        input_cost_per_1k=cost, output_cost_per_1k=cost,
    )


def _registry(models):
    reg = ModelRegistry()
    reg.models = {m.model_id: m for m in models}
    return reg


# req.1 + req.2 + req.8
def test_recommends_top_scored_supporting_model_and_is_deterministic():
    M.metrics.reset()
    reg = _registry([
        _model("big", [TaskType.CHAT], quality=0.95, params=100, latency=900),
        _model("small", [TaskType.CHAT], quality=0.80, params=8, latency=300),
    ])
    recs = recommend_all(reg)
    # req.2: ranking uses the router's own scoring
    expected = max(reg.models.values(), key=reg._score_model).model_id
    assert recs["chat"]["model"] == expected
    # req.8: deterministic for fixed state
    assert recommend_all(reg) == recs


# req.1 + req.5 (no model must not be faked with an unsupported one)
def test_reports_no_model_for_unsupported_task():
    M.metrics.reset()
    reg = _registry([_model("chat-only", [TaskType.CHAT])])
    recs = recommend_all(reg)
    assert recs["code_generation"]["model"] is None
    assert "no registered model" in recs["code_generation"]["rationale"]


# req.3
def test_rationale_is_explainable_with_alternatives():
    M.metrics.reset()
    reg = _registry([
        _model("a", [TaskType.CHAT], quality=0.9, params=70, tps=40, cost=0.001),
        _model("b", [TaskType.CHAT], quality=0.7, params=8),
    ])
    rat = recommend_all(reg)["chat"]["rationale"]
    assert {"parameters_b", "latency_ms", "throughput_tps",
            "output_cost_per_1k", "quality_score", "score"} <= set(rat)
    alts = recommend_all(reg)["chat"]["alternatives"]
    assert alts and alts[0]["model"] != recommend_all(reg)["chat"]["model"]


# req.4
def test_basis_prefers_measured_latency_over_estimate():
    M.metrics.reset()
    reg = _registry([_model("m1", [TaskType.CHAT], latency=500)])
    assert recommend_all(reg)["chat"]["basis"] == "estimated"
    M.metrics.record_latency("m1", 123.0)
    got = recommend_all(reg)["chat"]
    assert got["basis"] == "measured"
    assert got["rationale"]["latency_ms"] == 123.0
    M.metrics.reset()


# req.5 (pure/read-only: runs with no gateway/network)
def test_advisor_runs_without_gateway_or_network():
    out = recommend_all()  # real registry, no lifespan/http_client
    assert isinstance(out, dict) and out
    assert out == recommend_all()  # side-effect free / repeatable


# req.6
def test_endpoint_all_one_and_unknown(monkeypatch):
    import nvidia_smartroute.gateway.server as srv
    monkeypatch.setattr(srv.settings, "enable_rate_limit", False)
    client = TestClient(srv.app)
    all_tasks = client.get("/v1/recommend").json()
    assert set(all_tasks.keys()) == {t.value for t in TaskType}
    one = client.get("/v1/recommend?task=chat").json()
    assert set(one.keys()) == {"chat"}
    assert client.get("/v1/recommend?task=bogus").status_code == 400


# req.7
def test_cli_recommend_renders_without_gateway():
    from typer.testing import CliRunner
    import nvidia_smartroute.cli as cli
    result = CliRunner().invoke(cli.app, ["recommend"])
    assert result.exit_code == 0
    assert "Recommended model per task" in result.stdout
    assert CliRunner().invoke(cli.app, ["recommend", "--task", "bogus"]).exit_code == 1


def test_recommend_for_helper():
    assert recommend_for("nope") is None
    assert set(recommend_for("chat").keys()) == {"chat"}
    assert is_task("vision") and not is_task("xyz")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
