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

    monkeypatch.setattr(srv, "http_client", FakeClient())
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

    monkeypatch.setattr(srv, "http_client", FakeClient())
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

    monkeypatch.setattr(srv, "http_client", FakeClient())
    with pytest.raises(RuntimeError):
        asyncio.run(srv.nim_client._post_with_retries("http://x/chat", {"model": "m"}))


# --- Remote image inlining ------------------------------------------------

def test_inline_remote_images_replaces_url(monkeypatch):
    import nvidia_smartroute.gateway.server as srv
    from unittest.mock import AsyncMock

    monkeypatch.setattr(srv.settings, "inline_remote_images", True)
    monkeypatch.setattr(srv, "_fetch_as_data_url", AsyncMock(return_value="data:image/png;base64,AAAA"))

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
    ]}]
    out = asyncio.run(srv._inline_remote_images(messages))
    img_part = out[0]["content"][1]
    assert img_part["image_url"]["url"].startswith("data:image/png;base64,")
    # A data URL should be left untouched (no fetch).
    already = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,ZZ"}}]}]
    out2 = asyncio.run(srv._inline_remote_images(already))
    assert out2[0]["content"][0]["image_url"]["url"] == "data:image/png;base64,ZZ"


# --- Streaming metrics ----------------------------------------------------

def test_streaming_records_latency(monkeypatch):
    import nvidia_smartroute.gateway.server as srv

    async def fake_stream(model, messages, stream, max_tokens=None, temperature=None, **kwargs):
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'

    monkeypatch.setattr(srv, "_stream_nim_request", fake_stream)
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
            async for l in fake_line_stream():
                yield l

    class FakeHTTP:
        def stream(self, *a, **k): return FakeStreamResp()

    monkeypatch.setattr(srv, "http_client", FakeHTTP())
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


def test_metrics_endpoint_includes_key_pool():
    from nvidia_smartroute.gateway.server import app

    data = TestClient(app).get("/metrics").json()
    assert "api_keys" in data
    assert isinstance(data["api_keys"], list)


# --- /v1/models returns router registry -----------------------------------

def test_models_endpoint_returns_router_registry():
    from nvidia_smartroute.gateway.server import app, router

    data = TestClient(app).get("/v1/models").json()
    assert data["object"] == "list"
    ids = {m["id"] for m in data["data"]}
    assert ids == set(router.model_registry.models.keys())
    # Router metadata is exposed and OpenAI shape is preserved.
    sample = data["data"][0]
    assert sample["object"] == "model" and "supported_tasks" in sample


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

    monkeypatch.setattr(srv, "http_client", FakeClient())
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
    from unittest.mock import AsyncMock
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
