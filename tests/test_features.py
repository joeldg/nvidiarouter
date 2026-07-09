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
    # total_requests is bumped separately (real API traffic), not by the
    # connection gauge — so health/metrics polling doesn't inflate it.
    tracker.note_request()
    tracker.note_request()
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
    body = resp.json()
    assert {a["role"] for a in body["_agents"]} == {"writer", "tester", "reviewer"}
    assert "usage" in body and "total_tokens" in body["usage"]


def test_autoscale_aggregates_token_usage():
    engine = AutoscaleEngine(max_concurrent=5, timeout=10)

    async def fake_nim(model, messages, **kwargs):
        return {
            "choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    result = asyncio.run(engine.orchestrate(
        messages=[{"role": "user", "content": "build a thing with tests"}],
        model_id="m", nim_call=fake_nim,
    ))
    # 3 sub-agents each report 10/5 -> summed 30/15/45.
    assert result["usage"] == {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45}


def test_stop_without_pidfile_errors(tmp_path, monkeypatch):
    import nvidia_smartroute.cli as cli
    from typer.testing import CliRunner

    monkeypatch.setattr(cli.settings, "pid_file", str(tmp_path / "absent.pid"))
    result = CliRunner().invoke(cli.app, ["stop"])
    assert result.exit_code == 1


def test_stop_signals_pid(tmp_path, monkeypatch):
    import os
    import nvidia_smartroute.cli as cli
    from typer.testing import CliRunner

    pid_file = tmp_path / "gw.pid"
    pid_file.write_text("424242")
    monkeypatch.setattr(cli.settings, "pid_file", str(pid_file))
    signalled = {}
    monkeypatch.setattr(os, "kill", lambda pid, sig: signalled.update(pid=pid, sig=sig))

    result = CliRunner().invoke(cli.app, ["stop"])
    assert result.exit_code == 0
    assert signalled["pid"] == 424242


# --- Inbound rate limiting ------------------------------------------------

def test_rate_limit_returns_429(monkeypatch):
    from unittest.mock import AsyncMock
    import nvidia_smartroute.gateway.server as srv

    monkeypatch.setattr(srv.settings, "rate_limit_requests", 2)
    monkeypatch.setattr(srv.settings, "rate_limit_window", 60)
    monkeypatch.setattr(srv.settings, "enable_rate_limit", True)
    srv._rate_windows.clear()
    monkeypatch.setattr(
        srv.nim_client,
        "chat_completions",
        AsyncMock(return_value={"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}),
    )
    client = TestClient(srv.app)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    codes = [client.post("/v1/chat/completions", json=body).status_code for _ in range(3)]
    assert codes[0] == 200 and codes[1] == 200
    assert codes[2] == 429
    srv._rate_windows.clear()


def test_api_key_auth_enforced_on_v1(monkeypatch):
    from unittest.mock import AsyncMock
    import nvidia_smartroute.gateway.server as srv

    monkeypatch.setattr(srv.settings, "require_api_key", True)
    monkeypatch.setattr(srv.settings, "gateway_api_keys", "secret123")
    monkeypatch.setattr(srv.settings, "enable_rate_limit", False)
    monkeypatch.setattr(
        srv.nim_client, "chat_completions",
        AsyncMock(return_value={"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}),
    )
    client = TestClient(srv.app)
    body = {"messages": [{"role": "user", "content": "hi"}]}

    # No key -> 401
    assert client.post("/v1/chat/completions", json=body).status_code == 401
    # Wrong key -> 401
    assert client.post("/v1/chat/completions", json=body, headers={"X-API-Key": "nope"}).status_code == 401
    # Correct key (header) -> 200
    assert client.post("/v1/chat/completions", json=body, headers={"X-API-Key": "secret123"}).status_code == 200
    # Correct key (Bearer) -> 200
    assert client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer secret123"}).status_code == 200
    # Health is never gated
    assert client.get("/health").status_code == 200


def test_health_polling_does_not_inflate_total_requests(monkeypatch):
    from unittest.mock import AsyncMock
    import nvidia_smartroute.gateway.server as srv

    monkeypatch.setattr(srv.settings, "enable_rate_limit", False)
    srv.metrics.reset()
    monkeypatch.setattr(
        srv.nim_client, "chat_completions",
        AsyncMock(return_value={"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}),
    )
    client = TestClient(srv.app)
    for _ in range(5):
        client.get("/health")
        client.get("/metrics")
    assert client.get("/metrics").json()["total_requests"] == 0  # polling doesn't count
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert client.get("/metrics").json()["total_requests"] == 1  # only the API call
    srv.metrics.reset()


def test_health_endpoint_not_rate_limited(monkeypatch):
    import nvidia_smartroute.gateway.server as srv

    monkeypatch.setattr(srv.settings, "rate_limit_requests", 1)
    monkeypatch.setattr(srv.settings, "enable_rate_limit", True)
    srv._rate_windows.clear()
    client = TestClient(srv.app)
    # /health is outside /v1/* and must never be limited.
    assert all(client.get("/health").status_code == 200 for _ in range(5))


# --- Upstream retry / backoff ---------------------------------------------

class _FakeResp:
    def __init__(self, status, body=None):
        self.status_code = status
        self.headers = {}
        self._body = body if body is not None else {"ok": status == 200}
    def json(self):
        return self._body
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_post_with_retries_fails_over_on_429(monkeypatch):
    """A 429 cools the key down and the request fails over to another key."""
    import nvidia_smartroute.gateway.server as srv
    from nvidia_smartroute.keypool import KeyPool

    monkeypatch.setattr(srv.nim_client, "key_pool", KeyPool(["k1", "k2"], per_key_limit=10, window=60))
    monkeypatch.setattr(srv.settings, "upstream_max_retries", 3)
    monkeypatch.setattr(srv.settings, "upstream_backoff_base", 0.0)

    used = []

    class FakeClient:
        async def post(self, url, json=None, headers=None):
            key = headers.get("Authorization", "").replace("Bearer ", "")
            used.append(key)
            return _FakeResp(429 if key == "k1" else 200)

    monkeypatch.setattr(srv.runtime, "http_client", FakeClient())
    result = asyncio.run(srv.nim_client._post_with_retries("http://x/chat", {"model": "m"}))
    assert result == {"ok": True}
    assert "k1" in used and "k2" in used  # rotated off the rate-limited key


def test_post_with_retries_retries_5xx_same_key(monkeypatch):
    import nvidia_smartroute.gateway.server as srv
    from nvidia_smartroute.keypool import KeyPool

    monkeypatch.setattr(srv.nim_client, "key_pool", KeyPool(["k1"], per_key_limit=10, window=60))
    monkeypatch.setattr(srv.settings, "upstream_max_retries", 3)
    monkeypatch.setattr(srv.settings, "upstream_backoff_base", 0.0)
    seq = [_FakeResp(503), _FakeResp(200)]

    class FakeClient:
        async def post(self, url, json=None, headers=None):
            return seq.pop(0)

    monkeypatch.setattr(srv.runtime, "http_client", FakeClient())
    result = asyncio.run(srv.nim_client._post_with_retries("http://x/chat", {"model": "m"}))
    assert result == {"ok": True}
    assert seq == []


def test_post_with_retries_raises_on_4xx(monkeypatch):
    import nvidia_smartroute.gateway.server as srv
    from nvidia_smartroute.keypool import KeyPool

    monkeypatch.setattr(srv.nim_client, "key_pool", KeyPool(["k1"], per_key_limit=10, window=60))
    monkeypatch.setattr(srv.settings, "upstream_max_retries", 3)

    class FakeClient:
        async def post(self, url, json=None, headers=None):
            return _FakeResp(400)

    monkeypatch.setattr(srv.runtime, "http_client", FakeClient())
    with pytest.raises(RuntimeError):
        asyncio.run(srv.nim_client._post_with_retries("http://x/chat", {"model": "m"}))


# --- Remote image inlining ------------------------------------------------

def test_inline_remote_images_replaces_url(monkeypatch):
    import nvidia_smartroute.gateway.images as images
    from unittest.mock import AsyncMock

    monkeypatch.setattr(images.settings, "inline_remote_images", True)
    monkeypatch.setattr(images, "fetch_as_data_url", AsyncMock(return_value="data:image/png;base64,AAAA"))

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
    ]}]
    out = asyncio.run(images.inline_remote_images(messages))
    img_part = out[0]["content"][1]
    assert img_part["image_url"]["url"].startswith("data:image/png;base64,")
    # A data URL should be left untouched (no fetch).
    already = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,ZZ"}}]}]
    out2 = asyncio.run(images.inline_remote_images(already))
    assert out2[0]["content"][0]["image_url"]["url"] == "data:image/png;base64,ZZ"


# --- Streaming metrics ----------------------------------------------------

def test_streaming_records_latency(monkeypatch):
    import nvidia_smartroute.gateway.server as srv
    import nvidia_smartroute.gateway.streaming as streaming

    async def fake_stream(model, messages, stream, max_tokens=None, temperature=None, **kwargs):
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'

    monkeypatch.setattr(streaming, "stream_nim_request", fake_stream)
    srv.metrics.reset()

    async def consume():
        chunks = []
        async for c in srv.stream_chat_completion("m/x", [{"role": "user", "content": "hi"}], "rid"):
            chunks.append(c)
        return chunks

    chunks = asyncio.run(consume())
    assert chunks[-1] == "data: [DONE]\n\n"
    assert srv.metrics.get_avg_latency_ms("m/x") is not None
    srv.metrics.reset()


def test_streaming_records_usage_from_chunk(monkeypatch):
    """When the upstream stream includes a usage chunk, tokens are recorded."""
    import nvidia_smartroute.gateway.server as srv

    async def fake_line_stream():
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        yield 'data: {"choices":[],"usage":{"total_tokens":7}}'
        yield "data: [DONE]"

    class FakeStreamResp:
        status_code = 200
        headers = {}
        def raise_for_status(self): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def aiter_lines(self):
            async for chunk in fake_line_stream():
                yield chunk

    class FakeHTTP:
        def stream(self, *a, **k): return FakeStreamResp()

    monkeypatch.setattr(srv.runtime, "http_client", FakeHTTP())
    monkeypatch.setattr(srv.nim_client, "key_pool", __import__("nvidia_smartroute.keypool", fromlist=["KeyPool"]).KeyPool(["k"], 10, 60))
    srv.metrics.reset()

    async def consume():
        async for _ in srv._stream_nim_request("m/y", [{"role": "user", "content": "hi"}], True):
            pass

    asyncio.run(consume())
    snap = {m["model_id"]: m for m in srv.metrics.snapshot()["models"]}
    assert snap["m/y"]["total_tokens"] == 7
    srv.metrics.reset()


# --- Key pool -------------------------------------------------------------

def test_config_api_keys_merges_and_dedupes(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "k1")
    monkeypatch.setenv("NVIDIA_API_KEYS", "k2, k3, k1")  # k1 duplicated
    s = Settings(_env_file=None)
    assert s.api_keys == ["k2", "k3", "k1"]


def test_keypool_spreads_load_across_keys():
    from nvidia_smartroute.keypool import KeyPool

    pool = KeyPool(["k1", "k2", "k3"], per_key_limit=40, window=60)
    picks = [pool.acquire()[0] for _ in range(6)]
    # Picking the most-remaining key each time rotates evenly across the pool.
    assert set(picks) == {"k1", "k2", "k3"}
    assert picks.count("k1") == picks.count("k2") == picks.count("k3") == 2


def test_keypool_exhaustion_reports_wait():
    from nvidia_smartroute.keypool import KeyPool

    pool = KeyPool(["only"], per_key_limit=2, window=60)
    assert pool.acquire()[0] == "only"
    assert pool.acquire()[0] == "only"
    key, wait = pool.acquire()  # budget exhausted
    assert key is None
    assert 0 < wait <= 60


def test_keypool_cooldown_and_failover():
    from nvidia_smartroute.keypool import KeyPool

    pool = KeyPool(["a", "b"], per_key_limit=40, window=60)
    pool.record_cooldown("a", 30)
    # 'a' is cooling down, so acquire must return 'b'.
    assert pool.acquire()[0] == "b"


def test_keypool_snapshot_masks_keys():
    from nvidia_smartroute.keypool import KeyPool

    pool = KeyPool(["testkey01-abcdefghij-secret-xyz"], per_key_limit=40, window=60)
    pool.acquire()
    snap = pool.snapshot()
    assert snap[0]["used"] == 1
    assert snap[0]["remaining"] == 39
    assert "secret" not in snap[0]["key"]
    assert snap[0]["key"].startswith("testkey01")


def test_prometheus_exposition_format():
    from nvidia_smartroute.prometheus import render_prometheus

    snap = {
        "total_requests": 5, "active_connections": 1, "uptime_seconds": 12,
        "total_cost_usd": 0.01,
        "cache": {"hits": 3, "misses": 2},
        "concurrency": {"inflight": 1, "queued": 0, "rejected": 0},
        "budget": {"spend_usd": 0.01},
        "models": [{"model_id": "meta/x-70b", "request_count": 5, "error_count": 0,
                    "total_tokens": 100, "avg_latency_ms": 700, "max_tps": 50,
                    "total_cost_usd": 0.01}],
    }
    text = render_prometheus(snap)
    assert "# TYPE nsr_total_requests counter" in text
    assert "nsr_total_requests 5" in text
    assert 'nsr_model_requests{model="meta/x-70b"} 5' in text
    assert 'nsr_model_avg_latency_ms{model="meta/x-70b"} 700' in text


def test_metrics_prometheus_endpoint():
    from nvidia_smartroute.gateway.server import app

    r = TestClient(app).get("/metrics/prometheus")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "nsr_total_requests" in r.text


def test_metrics_endpoint_includes_key_pool():
    from nvidia_smartroute.gateway.server import app

    data = TestClient(app).get("/metrics").json()
    assert "api_keys" in data
    assert isinstance(data["api_keys"], list)


# --- /v1/models returns router registry -----------------------------------

def test_models_endpoint_returns_router_registry(monkeypatch):
    from nvidia_smartroute.gateway.server import app, router, settings

    monkeypatch.setattr(settings, "enable_parkour", False)
    data = TestClient(app).get("/v1/models").json()
    assert data["object"] == "list"
    ids = {m["id"] for m in data["data"]}
    assert ids == set(router.model_registry.models.keys())
    # Router metadata is exposed and OpenAI shape is preserved.
    sample = data["data"][0]
    assert sample["object"] == "model" and "supported_tasks" in sample


def test_models_endpoint_includes_enabled_parkour(monkeypatch):
    import nvidia_smartroute.gateway.server as srv

    monkeypatch.setattr(srv.settings, "enable_parkour", True)
    data = TestClient(srv.app).get("/v1/models").json()["data"]
    parkour = next(model for model in data if model["id"] == "parkour")
    assert parkour["object"] == "model"
    assert parkour["owned_by"] == "nvidia-smartroute"
    assert parkour["model_type"] == "virtual"
    assert parkour["execution_strategy"] == "multi_agent_dag"
    assert parkour["supports_streaming"] is False
    # A virtual execution strategy must never become a routing candidate.
    assert "parkour" not in srv.router.model_registry.models


def test_upstream_models_never_injects_parkour(monkeypatch):
    import nvidia_smartroute.gateway.server as srv

    async def upstream_models():
        return {"object": "list", "data": [{"id": "upstream-model"}]}

    monkeypatch.setattr(srv.settings, "enable_parkour", True)
    monkeypatch.setattr(srv.nim_client, "models", upstream_models)
    data = TestClient(srv.app).get("/v1/models?source=upstream").json()
    assert data["data"] == [{"id": "upstream-model"}]


def test_parkour_settings_are_bounded_and_default_off(monkeypatch):
    from pydantic import ValidationError
    from nvidia_smartroute.config import Settings

    monkeypatch.delenv("ENABLE_PARKOUR", raising=False)
    monkeypatch.delenv("PARKOUR_MAX_NODES", raising=False)
    settings = Settings(_env_file=None)
    assert settings.enable_parkour is False
    assert settings.parkour_max_nodes == 8
    assert settings.parkour_max_depth == 3
    assert settings.parkour_max_concurrency == 4
    assert settings.parkour_max_calls == 12
    assert settings.parkour_timeout_seconds == 300
    assert settings.parkour_max_tokens == 64_000
    assert settings.parkour_max_cost_usd == 1.0

    monkeypatch.setenv("PARKOUR_MAX_NODES", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_upstream_models_uses_key_pool_headers(monkeypatch):
    """Regression: models() must build headers from the key pool, not self.headers."""
    import nvidia_smartroute.gateway.server as srv
    from nvidia_smartroute.keypool import KeyPool

    seen = {}

    class FakeClient:
        async def get(self, url, headers=None):
            seen["auth"] = headers.get("Authorization")

            class R:
                status_code = 200
                def raise_for_status(self): pass
                def json(self): return {"object": "list", "data": [{"id": "x"}]}
            return R()

    monkeypatch.setattr(srv.runtime, "http_client", FakeClient())
    monkeypatch.setattr(srv.nim_client, "key_pool", KeyPool(["poolkey"], per_key_limit=10, window=60))
    out = asyncio.run(srv.nim_client.models())
    assert out["data"] == [{"id": "x"}]
    assert seen["auth"] == "Bearer poolkey"


# --- Autoscale sequential mode --------------------------------------------

def test_autoscale_sequential_runs_followups_one_at_a_time(monkeypatch):
    import nvidia_smartroute.agents.orchestrator as orch

    monkeypatch.setattr(orch.settings, "autoscale_sequential", True)
    engine = orch.AutoscaleEngine(max_concurrent=5, timeout=10)
    concurrent = 0
    max_seen = 0

    async def fake_nim(model, messages, **kwargs):
        nonlocal concurrent, max_seen
        concurrent += 1
        max_seen = max(max_seen, concurrent)
        await asyncio.sleep(0.01)
        concurrent -= 1
        return {"choices": [{"message": {"content": "x"}}]}

    result = asyncio.run(engine.orchestrate(
        messages=[{"role": "user", "content": "build with tests"}],
        model_id="m", nim_call=fake_nim,
    ))
    assert len(result["agents"]) == 3
    # Sequential mode: never more than one call in flight at a time.
    assert max_seen == 1


# --- Embeddings hardening --------------------------------------------------

def test_embeddings_requires_input():
    from nvidia_smartroute.gateway.server import app

    resp = TestClient(app).post("/v1/embeddings", json={"model": "m"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["type"] == "invalid_request_error"


def test_embeddings_defaults_model_and_records_metrics(monkeypatch):
    import nvidia_smartroute.gateway.server as srv

    captured = {}

    async def fake_embeddings(model, input, encoding_format="float", **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return {"object": "list", "data": [{"embedding": [0.1, 0.2]}], "usage": {"total_tokens": 4}}

    monkeypatch.setattr(srv.nim_client, "embeddings", fake_embeddings)
    srv.metrics.reset()
    resp = TestClient(srv.app).post("/v1/embeddings", json={"input": "hello"})
    assert resp.status_code == 200
    # Defaulted to the configured embedding model and forwarded input_type/truncate.
    assert captured["model"] == srv.settings.default_embedding_model
    assert captured["kwargs"].get("input_type") == "query"
    assert srv.metrics.get_avg_latency_ms(srv.settings.default_embedding_model) is not None
    srv.metrics.reset()


# --- Response cache -------------------------------------------------------

def test_response_cache_hit_skips_upstream(monkeypatch):
    from unittest.mock import AsyncMock
    import nvidia_smartroute.gateway.server as srv

    monkeypatch.setattr(srv.settings, "enable_cache", True)
    monkeypatch.setattr(srv.settings, "enable_rate_limit", False)
    srv.response_cache.clear()
    upstream = AsyncMock(return_value={"choices": [{"index": 0, "message": {"role": "assistant", "content": "42"}, "finish_reason": "stop"}]})
    monkeypatch.setattr(srv.nim_client, "chat_completions", upstream)

    client = TestClient(srv.app)
    body = {"messages": [{"role": "user", "content": "cache me please"}], "temperature": 0}
    r1 = client.post("/v1/chat/completions", json=body)
    r2 = client.post("/v1/chat/completions", json=body)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.headers.get("X-Cache") == "MISS"
    assert r2.headers.get("X-Cache") == "HIT"
    assert upstream.await_count == 1  # second request served from cache
    srv.response_cache.clear()


# --- Model fallback chain -------------------------------------------------

def test_model_fallback_on_upstream_error(monkeypatch):
    import nvidia_smartroute.gateway.server as srv
    from nvidia_smartroute.routing.router import TaskType

    monkeypatch.setattr(srv.settings, "enable_model_fallback", True)
    monkeypatch.setattr(srv.settings, "max_model_fallbacks", 2)

    ranked = srv.router.model_registry.rank_models(TaskType.CHAT)
    primary = ranked[0]

    class FakeResp:
        status_code = 500

    class UpstreamError(Exception):
        response = FakeResp()

    tried = []

    async def fake_chat(model, messages, stream=False, max_tokens=None, temperature=None, **kw):
        tried.append(model)
        if model == primary.model_id:
            raise UpstreamError()
        return {"choices": [{"message": {"content": "ok"}}], "model": model}

    monkeypatch.setattr(srv.nim_client, "chat_completions", fake_chat)
    data, used, fell_back = asyncio.run(srv._complete_with_fallback(
        TaskType.CHAT, primary, [{"role": "user", "content": "hi"}], None, None, {},
    ))
    assert fell_back is True
    assert used.model_id != primary.model_id
    assert tried[0] == primary.model_id  # primary attempted first


def test_no_fallback_on_client_error(monkeypatch):
    import nvidia_smartroute.gateway.server as srv
    from nvidia_smartroute.routing.router import TaskType

    monkeypatch.setattr(srv.settings, "enable_model_fallback", True)
    ranked = srv.router.model_registry.rank_models(TaskType.CHAT)
    primary = ranked[0]

    class FakeResp:
        status_code = 400

    class BadRequest(Exception):
        response = FakeResp()

    async def fake_chat(model, messages, **kw):
        raise BadRequest()

    monkeypatch.setattr(srv.nim_client, "chat_completions", fake_chat)
    # A 400 is the same on every model, so it must not fan out.
    with pytest.raises(BadRequest):
        asyncio.run(srv._complete_with_fallback(
            TaskType.CHAT, primary, [{"role": "user", "content": "hi"}], None, None, {},
        ))


# --- Tool / function calling passthrough ----------------------------------

def test_tools_are_forwarded_and_tool_calls_returned(monkeypatch):
    import nvidia_smartroute.gateway.server as srv

    monkeypatch.setattr(srv.settings, "enable_cache", False)
    monkeypatch.setattr(srv.settings, "enable_rate_limit", False)
    captured = {}

    async def fake_chat(model, messages, stream=False, max_tokens=None, temperature=None, **kw):
        captured.update(kw)
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
            "finish_reason": "tool_calls"}]}

    monkeypatch.setattr(srv.nim_client, "chat_completions", fake_chat)
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    resp = TestClient(srv.app).post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "weather?"}], "tools": tools, "tool_choice": "auto",
    })
    assert resp.status_code == 200
    # tools/tool_choice were forwarded upstream ...
    assert captured.get("tools") == tools
    assert captured.get("tool_choice") == "auto"
    # ... and the tool_calls response passed straight through.
    assert resp.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


# --- Circuit breaker ------------------------------------------------------

def test_circuit_breaker_trips_probes_and_recovers():
    import time
    from nvidia_smartroute.circuit import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=2, reset_seconds=0.05)
    assert cb.allow("m") and cb.state("m") == "closed"
    cb.record_failure("m")
    assert cb.allow("m")  # 1 failure, still closed
    cb.record_failure("m")  # 2 -> trips open
    assert cb.state("m") == "open"
    assert cb.allow("m") is False
    time.sleep(0.06)
    assert cb.allow("m") is True  # cooldown elapsed -> half-open probe
    cb.record_failure("m")  # failed probe re-opens
    assert cb.allow("m") is False
    time.sleep(0.06)
    assert cb.allow("m") is True  # half-open again
    cb.record_success("m")  # successful probe closes it
    assert cb.state("m") == "closed" and cb.allow("m")


def test_fallback_skips_open_circuit(monkeypatch):
    import nvidia_smartroute.gateway.server as srv
    from nvidia_smartroute.routing.router import TaskType

    srv.breaker.reset()
    monkeypatch.setattr(srv.settings, "circuit_breaker_enabled", True)
    monkeypatch.setattr(srv.settings, "enable_model_fallback", True)
    ranked = srv.router.model_registry.rank_models(TaskType.CHAT)
    primary = ranked[0]
    for _ in range(srv.settings.circuit_failure_threshold):
        srv.breaker.record_failure(primary.model_id)

    tried = []

    async def fake_chat(model, messages, **kw):
        tried.append(model)
        return {"choices": [{"message": {"content": "ok"}}], "model": model}

    monkeypatch.setattr(srv.nim_client, "chat_completions", fake_chat)
    data, used, fell_back = asyncio.run(srv._complete_with_fallback(
        TaskType.CHAT, primary, [{"role": "user", "content": "hi"}], None, None, {},
    ))
    assert primary.model_id not in tried  # open circuit was skipped
    assert used.model_id != primary.model_id and fell_back is True
    srv.breaker.reset()


# --- Persistent metrics ---------------------------------------------------

def test_metrics_dump_load_roundtrip():
    from nvidia_smartroute.metrics import MetricsTracker

    m = MetricsTracker()
    m.record_latency("x/y", 100.0)
    m.record_latency("x/y", 200.0)
    m.record_tokens("x/y", 42)
    m.record_error("x/y")

    restored = MetricsTracker()
    restored.load(m.dump())
    row = {r["model_id"]: r for r in restored.snapshot()["models"]}["x/y"]
    assert row["request_count"] == 2
    assert row["total_tokens"] == 42
    assert row["error_count"] == 1
    assert row["avg_latency_ms"] == 150.0


# --- Concurrency gate / backpressure --------------------------------------

def test_concurrency_gate_sheds_load_when_full():
    from nvidia_smartroute.concurrency import ConcurrencyGate, QueueFullError

    async def run():
        g = ConcurrencyGate(max_inflight=1, max_queued=0, timeout=1.0)
        await g.acquire()  # takes the only slot
        assert g.inflight == 1
        with pytest.raises(QueueFullError):  # queue full -> shed load
            await g.acquire()
        g.release()
        await g.acquire()  # slot free again
        g.release()

    asyncio.run(run())


def test_concurrency_gate_queues_then_proceeds():
    from nvidia_smartroute.concurrency import ConcurrencyGate

    async def run():
        g = ConcurrencyGate(max_inflight=1, max_queued=4, timeout=2.0)
        await g.acquire()

        async def waiter():
            await g.acquire()
            g.release()
            return True

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)  # let the waiter enqueue
        assert g.snapshot()["queued"] == 1
        g.release()  # free the slot; the waiter proceeds
        assert await task is True

    asyncio.run(run())


# --- Cost & budget --------------------------------------------------------

def test_compute_cost():
    from nvidia_smartroute.cost import compute_cost

    class M:
        input_cost_per_1k = 0.001
        output_cost_per_1k = 0.002

    # 1000 prompt @ 0.001 + 500 completion @ 0.002 = 0.001 + 0.001 = 0.002
    assert compute_cost(M(), 1000, 500) == pytest.approx(0.002)


def test_budget_blocks_when_exceeded():
    from nvidia_smartroute.cost import BudgetTracker

    b = BudgetTracker(daily_budget_usd=0.01)
    assert b.allow()
    b.record(0.005)
    assert b.allow()
    b.record(0.006)  # total 0.011 > cap
    assert not b.allow()
    assert b.snapshot()["remaining_usd"] == 0.0


def test_budget_unlimited_allows():
    from nvidia_smartroute.cost import BudgetTracker

    b = BudgetTracker(daily_budget_usd=0.0)
    b.record(9999.0)
    assert b.allow()
    assert b.snapshot()["daily_budget_usd"] is None


def test_metrics_records_max_throughput():
    from nvidia_smartroute.metrics import MetricsTracker

    m = MetricsTracker()
    m.record_throughput("x", 50.0)
    m.record_throughput("x", 120.0)
    m.record_throughput("x", 80.0)  # lower than peak — ignored
    snap = {r["model_id"]: r for r in m.snapshot()["models"]}
    assert snap["x"]["max_tps"] == 120.0


def test_metrics_records_cost():
    from nvidia_smartroute.metrics import MetricsTracker

    m = MetricsTracker()
    m.record_cost("a", 0.0012)
    m.record_cost("a", 0.0008)
    snap = m.snapshot()
    assert snap["total_cost_usd"] == 0.002
    assert {r["model_id"]: r for r in snap["models"]}["a"]["total_cost_usd"] == 0.002


def test_cost_aware_routing_prefers_cheaper(monkeypatch):
    import nvidia_smartroute.routing.router as rmod
    from nvidia_smartroute.routing.router import RequestRouter, TaskType

    rmod.metrics.reset()
    r = RequestRouter()
    # Cost ignored -> highest-quality chat model wins.
    monkeypatch.setattr(rmod.settings, "cost_weight", 0.0)
    assert r.model_registry.select_best_model(TaskType.CHAT).model_id == \
        "nvidia/llama-3.3-nemotron-super-49b-v1"
    # Strong cost weight -> cheapest chat model wins.
    monkeypatch.setattr(rmod.settings, "cost_weight", 1.0)
    assert r.model_registry.select_best_model(TaskType.CHAT).model_id == \
        "meta/llama-3.1-8b-instruct"


# --- Adaptive routing (bandit) --------------------------------------------

def test_reward_from():
    from nvidia_smartroute.bandit import reward_from

    assert reward_from(False, 100) == 0.0
    assert reward_from(True, 0) == 1.0
    assert reward_from(True, 1000) == 0.75
    assert reward_from(True, 2000) == 0.5  # full latency penalty


def test_bandit_exploits_best_and_explores_new():
    from nvidia_smartroute.bandit import AdaptiveRouter

    b = AdaptiveRouter(epsilon=0.0)  # pure exploitation
    b.record("chat", "good", 0.9)
    b.record("chat", "bad", 0.1)
    assert b.select("chat", ["good", "bad"]) == "good"
    # An unseen arm starts optimistic (1.0) so it gets explored first.
    assert b.select("chat", ["good", "bad", "fresh"]) == "fresh"


def test_bandit_explores_with_epsilon(monkeypatch):
    import nvidia_smartroute.bandit as bmod
    from nvidia_smartroute.bandit import AdaptiveRouter

    b = AdaptiveRouter(epsilon=1.0)  # always explore
    monkeypatch.setattr(bmod.random, "choice", lambda c: c[-1])
    assert b.select("t", ["a", "b", "c"]) == "c"


def test_adaptive_strategy_used_by_router(monkeypatch):
    import nvidia_smartroute.routing.router as rmod
    from nvidia_smartroute.routing.router import RequestRouter, TaskType

    monkeypatch.setattr(rmod.settings, "routing_strategy", "adaptive")
    monkeypatch.setattr(rmod.adaptive_router, "select", lambda task, cands: cands[-1])
    r = RequestRouter()
    chosen = r._select_model(TaskType.CHAT)
    ranked = r.model_registry.rank_models(TaskType.CHAT)
    assert chosen.model_id == ranked[-1].model_id  # bandit's pick was honoured


# --- Stress-test helpers --------------------------------------------------

def test_percentile():
    from nvidia_smartroute.cli import _percentile

    assert _percentile([], 50) == 0.0
    assert _percentile([10], 50) == 10
    vals = [10, 20, 30, 40, 50]
    assert _percentile(vals, 0) == 10
    assert _percentile(vals, 100) == 50
    assert _percentile(vals, 50) == 30


def test_summarize_stress():
    from nvidia_smartroute.cli import _summarize_stress

    results = [
        {"status": 200, "ms": 100, "model": "m1", "cache": "MISS"},
        {"status": 200, "ms": 200, "model": "m1", "cache": "HIT"},
        {"status": 503, "ms": 5, "model": None, "cache": None},
        {"status": 200, "ms": 300, "model": "m2", "cache": "MISS"},
    ]
    s = _summarize_stress(results, elapsed=2.0)
    assert s["total"] == 4 and s["ok"] == 3 and s["failed"] == 1
    assert s["rps"] == 2.0
    assert s["cache_hits"] == 1
    assert s["status_counts"] == {200: 3, 503: 1}
    assert s["model_counts"] == {"m1": 2, "m2": 1}
    assert s["p50_ms"] == 200.0  # median of ok latencies [100, 200, 300]


# --- Model discovery ------------------------------------------------------

def test_infer_capability_curated_and_inferred():
    from nvidia_smartroute.model_catalog import infer_capability, is_embedding_model
    from nvidia_smartroute.routing.router import TaskType

    kimi = infer_capability("moonshotai/kimi-k2-instruct")
    assert kimi["parameters_b"] == 1000
    assert kimi["supports_function_calling"] is True

    # Real catalog IDs (no size token) must resolve via the curated table.
    assert infer_capability("moonshotai/kimi-k2.6")["parameters_b"] == 1000
    assert infer_capability("z-ai/glm-5.2")["parameters_b"] == 355
    # Big models whose ID carries a size token infer correctly without curation.
    assert infer_capability("mistralai/mistral-large-3-675b-instruct-2512")["parameters_b"] == 675

    vision = infer_capability("meta/llama-3.2-90b-vision-instruct")
    assert vision["supports_vision"] is True
    assert vision["supported_tasks"] == [TaskType.VISION]
    assert vision["parameters_b"] == 90  # extracted from "90b"

    coder = infer_capability("qwen/qwen2.5-coder-32b-instruct")
    assert TaskType.CODE_GENERATION in coder["supported_tasks"]
    assert coder["parameters_b"] == 32

    assert is_embedding_model("nvidia/nv-embedqa-e5-v5") is True

    # Large general models also handle code (so code tasks have a home).
    big = infer_capability("mistralai/mistral-large-3-675b-instruct-2512")
    assert TaskType.CODE_GENERATION in big["supported_tasks"]
    assert TaskType.CHAT in big["supported_tasks"]
    # Tiny models stay chat-only.
    tiny = infer_capability("google/gemma-2-2b-it")
    assert TaskType.CODE_GENERATION not in tiny["supported_tasks"]


def test_is_routable_excludes_specialized_models():
    from nvidia_smartroute.model_catalog import is_routable

    assert is_routable("moonshotai/kimi-k2.6") is True
    assert is_routable("meta/llama-3.1-70b-instruct") is True
    # Guardrails, safety/PII classifiers, reward, embeddings, image-gen: excluded.
    for mid in [
        "meta/llama-guard-4-12b",
        "nvidia/llama-3.1-nemoguard-8b-content-safety",
        "nvidia/gliner-pii",
        "google/diffusiongemma-26b-a4b-it",
        "nvidia/nv-embedqa-e5-v5",
        "nvidia/nemotron-4-340b-reward",
    ]:
        assert is_routable(mid) is False, mid


def test_capability_serialize_roundtrip():
    from nvidia_smartroute.discovery import serialize, deserialize
    from nvidia_smartroute.model_catalog import infer_capability

    cap = deserialize(infer_capability("meta/llama-3.1-70b-instruct"))
    again = deserialize(serialize(cap))
    assert again.model_id == cap.model_id
    assert again.supported_tasks == cap.supported_tasks
    assert again.parameters_b == cap.parameters_b


def test_discover_probes_and_filters(monkeypatch):
    import nvidia_smartroute.discovery as disc

    monkeypatch.setattr(disc, "fetch_catalog", lambda base, key: [
        "moonshotai/kimi-k2-instruct",
        "meta/llama-3.1-8b-instruct",
        "nvidia/nv-embedqa-e5-v5",  # embedding -> filtered out
        "some/broken-model",
    ])
    # kimi + llama servable; broken one not.
    servable = {"moonshotai/kimi-k2-instruct", "meta/llama-3.1-8b-instruct"}
    monkeypatch.setattr(disc, "probe_servable", lambda base, key, mid: mid in servable)

    caps, report = disc.discover("http://x", "k")
    ids = {c.model_id for c in caps}
    assert ids == servable  # embedding filtered, broken excluded
    assert report["servable"] == 2
    assert report["catalog_size"] == 3  # embedding removed before probing


def test_apply_benchmark_updates_latency_and_throughput():
    from nvidia_smartroute.discovery import apply_benchmark, deserialize
    from nvidia_smartroute.model_catalog import infer_capability

    caps = [
        deserialize(infer_capability("moonshotai/kimi-k2.6")),
        deserialize(infer_capability("meta/llama-3.1-8b-instruct")),
    ]
    results = {
        "moonshotai/kimi-k2.6": {"ok": True, "p50_ms": 1255.0, "tps": 31.5},
        "meta/llama-3.1-8b-instruct": {"ok": False, "p50_ms": 0.0, "tps": 0.0},  # skipped
        "unknown/model": {"ok": True, "p50_ms": 100.0, "tps": 90.0},  # not present
    }
    updated = apply_benchmark(caps, results)
    assert updated == 1
    kimi = next(c for c in caps if c.model_id == "moonshotai/kimi-k2.6")
    assert kimi.latency_ms == 1255
    assert kimi.throughput_tps == 31.5


def test_registry_loads_discovered_file(tmp_path, monkeypatch):
    from nvidia_smartroute.discovery import save_models, deserialize
    from nvidia_smartroute.model_catalog import infer_capability
    import nvidia_smartroute.routing.router as rmod
    from nvidia_smartroute.routing.router import ModelRegistry

    path = tmp_path / "discovered.json"
    save_models(str(path), [deserialize(infer_capability("moonshotai/kimi-k2-instruct"))])
    monkeypatch.setattr(rmod.settings, "models_file", str(path))

    reg = ModelRegistry()
    assert "moonshotai/kimi-k2-instruct" in reg.models  # discovered model registered
    assert "meta/llama-3.1-8b-instruct" in reg.models   # defaults still present


# --- Web dashboard + playground -------------------------------------------

def test_dashboard_page_served():
    from nvidia_smartroute.gateway.server import app

    r = TestClient(app).get("/dashboard")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "NVIDIA SmartRoute" in r.text
    assert "Route &amp; Explain" in r.text  # playground present


def test_explain_returns_routing_detail(monkeypatch):
    from unittest.mock import AsyncMock
    import nvidia_smartroute.gateway.server as srv

    monkeypatch.setattr(srv.settings, "enable_rate_limit", False)
    monkeypatch.setattr(
        srv.nim_client, "chat_completions",
        AsyncMock(return_value={
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "391"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }),
    )
    r = TestClient(srv.app).post("/explain", json={"messages": [{"role": "user", "content": "What is 17 * 23?"}]})
    assert r.status_code == 200
    d = r.json()
    assert d["answer"] == "391"
    assert d["routing"]["task_type"] == "mathematics"
    assert "mathematics" in d["routing"]["scores"]
    assert d["routing"]["selected_model"]
    assert d["cost_usd"] >= 0
    assert d["latency_ms"] >= 0


def test_explain_requires_prompt():
    from nvidia_smartroute.gateway.server import app

    assert TestClient(app).post("/explain", json={}).status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
