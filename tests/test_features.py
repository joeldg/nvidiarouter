"""
Tests for the added capabilities: vision routing, latency-aware model scoring,
the metrics tracker, config env aliases, the autoscale engine, and the
gateway metrics endpoint.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from nvidia_smartroute.config import Settings
from nvidia_smartroute.metrics import MetricsTracker, metrics
from nvidia_smartroute.routing.router import (
    RequestRouter,
    ModelRegistry,
    TaskType,
)
from nvidia_smartroute.agents.orchestrator import AutoscaleEngine


# --- Vision -----------------------------------------------------------------

def test_multimodal_content_does_not_crash_and_detects_image():
    router = RequestRouter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this picture?"},
                {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
            ],
        }
    ]
    assert router.capability_analyzer.analyze_request(messages) == TaskType.VISION


def test_vision_model_is_selected_for_vision_task():
    registry = ModelRegistry()
    model = registry.select_best_model(TaskType.VISION)
    assert model is not None
    assert TaskType.VISION in model.supported_tasks
    assert model.supports_vision is True


# --- Scoring / latency tracker ---------------------------------------------

def test_scoring_prefers_lower_live_latency():
    metrics.reset()
    registry = ModelRegistry()
    # Two chat-capable models exist; give the normally-best one terrible live
    # latency and confirm the selection flips.
    best_static = registry.select_best_model(TaskType.CHAT)
    # Record a huge latency for the statically-best model.
    metrics.record_latency(best_static.model_id, 9999.0)
    after = registry.select_best_model(TaskType.CHAT)
    assert after.model_id != best_static.model_id
    metrics.reset()


# --- Metrics tracker --------------------------------------------------------

def test_metrics_tracker_records_and_snapshots():
    tracker = MetricsTracker()
    tracker.connection_opened()
    tracker.connection_opened()
    tracker.connection_closed()
    tracker.record_latency("m1", 100.0)
    tracker.record_latency("m1", 300.0)
    tracker.record_tokens("m1", 50)

    assert tracker.active_connections == 1
    assert tracker.get_avg_latency_ms("m1") == 200.0
    assert tracker.get_avg_latency_ms("unknown") is None

    snap = tracker.snapshot()
    assert snap["active_connections"] == 1
    assert snap["total_requests"] == 2
    m1 = next(m for m in snap["models"] if m["model_id"] == "m1")
    assert m1["request_count"] == 2
    assert m1["total_tokens"] == 50


# --- Config env aliases -----------------------------------------------------

def test_config_reads_short_env_var_names(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-123")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.com, https://b.com")
    s = Settings(_env_file=None)
    assert s.nvidia_nim_api_key == "nvapi-test-123"
    assert s.nvidia_nim_base_url == "https://example.test/v1"
    assert s.cors_origins == ["https://a.com", "https://b.com"]


# --- Autoscale engine -------------------------------------------------------

def test_should_scale_detects_multistep_code_task():
    engine = AutoscaleEngine()
    multistep = [{"role": "user", "content": "Write a parser and add unit tests"}]
    simple = [{"role": "user", "content": "add two numbers"}]
    assert engine.should_scale(TaskType.CODE_GENERATION, multistep) is True
    assert engine.should_scale(TaskType.CODE_GENERATION, simple) is False
    # Non-code tasks never scale.
    assert engine.should_scale(TaskType.CHAT, multistep) is False


def test_orchestrate_runs_writer_tester_reviewer():
    engine = AutoscaleEngine(max_concurrent=5, timeout=10)
    calls = []

    async def fake_nim(model, messages, **kwargs):
        # Echo the system role so we can see each agent ran.
        role_prompt = messages[0]["content"]
        calls.append(role_prompt)
        return {"choices": [{"message": {"content": f"output for: {role_prompt[:20]}"}}]}

    result = asyncio.run(
        engine.orchestrate(
            messages=[{"role": "user", "content": "Build a thing with tests"}],
            model_id="test/model",
            nim_call=fake_nim,
        )
    )
    # Writer + tester + reviewer = 3 sub-agents.
    assert len(result["agents"]) == 3
    assert {a["role"] for a in result["agents"]} == {"writer", "tester", "reviewer"}
    assert "## Implementation" in result["content"]
    assert "## Tests" in result["content"]
    assert "## Review" in result["content"]


def test_orchestrate_stops_when_writer_fails():
    engine = AutoscaleEngine()

    async def failing_nim(model, messages, **kwargs):
        raise RuntimeError("nim down")

    result = asyncio.run(
        engine.orchestrate(
            messages=[{"role": "user", "content": "x"}],
            model_id="test/model",
            nim_call=failing_nim,
        )
    )
    # Only the writer runs; it errored, so no tester/reviewer.
    assert len(result["agents"]) == 1
    assert result["agents"][0]["error"] is not None


# --- Gateway metrics endpoint ----------------------------------------------

def test_metrics_endpoint_returns_snapshot():
    from nvidia_smartroute.gateway.server import app

    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_connections" in data
    assert "models" in data
    assert "routing_stats" in data


def test_chat_completion_routes_and_records(monkeypatch):
    """End-to-end chat path: routes, calls NIM, records metrics (no network)."""
    from unittest.mock import AsyncMock
    import nvidia_smartroute.gateway.server as srv

    monkeypatch.setattr(
        srv.nim_client,
        "chat_completions",
        AsyncMock(
            return_value={
                "id": "chatcmpl-x",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
        ),
    )
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "What is 2+2?"}]},
    )
    assert resp.status_code == 200
    assert resp.headers["X-Task-Type"] == "mathematics"
    assert resp.headers["X-Selected-Model"]


def test_chat_completion_autoscales_multistep_code(monkeypatch):
    from unittest.mock import AsyncMock
    import nvidia_smartroute.gateway.server as srv

    monkeypatch.setattr(
        srv.nim_client,
        "chat_completions",
        AsyncMock(
            return_value={
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "code"}, "finish_reason": "stop"}],
            }
        ),
    )
    client = TestClient(srv.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Write a Python CSV parser and add unit tests to verify it"}]},
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-Autoscaled") == "true"
    assert resp.headers.get("X-Agent-Count") == "3"
    assert {a["role"] for a in resp.json()["_agents"]} == {"writer", "tester", "reviewer"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
