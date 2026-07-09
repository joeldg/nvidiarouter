"""PARKOUR OpenAI gateway contract tests."""

import json

from fastapi.testclient import TestClient

from nvidia_smartroute.parkour import (
    EngineResult,
    NodeResult,
    SchedulerResult,
    WorkerResult,
)


def _result():
    node = NodeResult("one", "worker answer", "worker-model", 5, 0.1, False)
    scheduled = SchedulerResult({"one": node}, 1, 5, 0.1)
    synthesis = WorkerResult("final answer", "synth-model", 3, 0.2)
    return EngineResult("final answer", scheduled, synthesis, 8, 0.3, False, False)


def _sse_events(text: str):
    events = []
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        events.append(json.loads(line[6:]))
    return events


def test_parkour_disabled_and_tools_errors(monkeypatch):
    import nvidia_smartroute.gateway.server as srv

    client = TestClient(srv.app)
    monkeypatch.setattr(srv.settings, "enable_parkour", False)
    response = client.post("/v1/chat/completions", json={
        "model": "parkour", "messages": [{"role": "user", "content": "hi"}]
    })
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "parkour_disabled"

    monkeypatch.setattr(srv.settings, "enable_parkour", True)
    response = client.post("/v1/chat/completions", json={
        "model": "parkour", "tools": [{"type": "function"}],
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.json()["error"]["code"] == "parkour_tools_unsupported"


def test_parkour_response_headers_and_opt_in_trace(monkeypatch):
    import nvidia_smartroute.gateway.server as srv

    async def run(messages):
        return _result()

    monkeypatch.setattr(srv.settings, "enable_parkour", True)
    monkeypatch.setattr(srv, "run_parkour", run)
    response = TestClient(srv.app).post("/v1/chat/completions", json={
        "model": "parkour", "parkour_trace": True,
        "messages": [{"role": "user", "content": "do work"}],
    })
    body = response.json()
    assert response.status_code == 200
    assert body["model"] == "parkour"
    assert body["choices"][0]["message"]["content"] == "final answer"
    assert body["usage"]["total_tokens"] == 8
    assert body["parkour"]["nodes"] == [{
        "id": "one", "status": "succeeded", "model": "worker-model",
        "context_truncated": False, "citations": 0,
    }]
    # Research lane is disabled by default, so no research summary is attached.
    assert "research" not in body["parkour"]
    assert response.headers["x-autoscale-type"] == "parkour"
    assert response.headers["x-agent-count"] == "1"
    assert len(response.headers["x-parkour-run-id"]) == 36


def test_parkour_streams_progress_events_and_final_answer(monkeypatch):
    import nvidia_smartroute.gateway.server as srv

    async def run(messages, progress=None):
        if progress is not None:
            await progress({
                "type": "worker_model_call",
                "node_id": "one",
                "model": "worker-model",
                "task_type": "reasoning",
            })
            await progress({
                "type": "node_completed",
                "node_id": "one",
                "model": "worker-model",
                "tokens": 5,
            })
        return _result()

    monkeypatch.setattr(srv.settings, "enable_parkour", True)
    monkeypatch.setattr(srv, "run_parkour", run)
    response = TestClient(srv.app).post("/v1/chat/completions", json={
        "model": "parkour", "stream": True,
        "messages": [{"role": "user", "content": "do work"}],
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-autoscale-type"] == "parkour"

    events = _sse_events(response.text)
    progress = [event.get("parkour_event", {}) for event in events]
    assert {"type": "worker_model_call", "node_id": "one",
            "model": "worker-model", "task_type": "reasoning"} in progress
    assert any(item.get("type") == "run_completed" for item in progress)
    content = "".join(
        event["choices"][0]["delta"].get("content", "") for event in events
    )
    assert content == "final answer"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert response.text.rstrip().endswith("data: [DONE]")


def test_parkour_trace_is_absent_by_default(monkeypatch):
    import nvidia_smartroute.gateway.server as srv

    async def run(messages):
        return _result()

    monkeypatch.setattr(srv.settings, "enable_parkour", True)
    monkeypatch.setattr(srv, "run_parkour", run)
    body = TestClient(srv.app).post("/v1/chat/completions", json={
        "model": "parkour",
        "messages": [{"role": "user", "content": "do work"}],
    }).json()
    assert "parkour" not in body


def test_openai_client_base_url_must_include_v1():
    import nvidia_smartroute.gateway.server as srv

    paths = {route.path for route in srv.app.routes}
    assert "/v1/chat/completions" in paths
    assert "/chat/completions" not in paths
