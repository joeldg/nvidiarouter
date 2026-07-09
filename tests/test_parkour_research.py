# @spec[PARKOUR_RESEARCH.md#Requirements]
"""Tests for the PARKOUR governed research lane."""

import types

import pytest

from nvidia_smartroute.parkour.research import (
    ResearchBlockedError,
    ResearchError,
    ResearchLimitError,
    ResearchLimits,
    ResearchSession,
    ResearchTelemetry,
    RawResult,
    build_research_session,
    domain_allowed,
    guard_url,
    mask_secret,
)
from nvidia_smartroute.parkour.scheduler import _inject_research
from nvidia_smartroute.parkour.planning import SubtaskSpec
from nvidia_smartroute.routing.router import TaskType


def _resolver(mapping):
    def resolve(host, port, **kw):
        ip = mapping.get(host, "93.184.216.34")  # default: a public address
        return [(2, 1, 6, "", (ip, port))]
    return resolve


PUBLIC = _resolver({"ok.com": "93.184.216.34"})


def _limits(**over):
    base = dict(
        max_searches_per_run=6, max_searches_per_node=2, max_query_chars=256,
        max_results=5, snippet_chars=500, max_bytes=200_000,
        timeout_seconds=15.0, cost_per_search_usd=0.005, max_cost_usd=0.1,
    )
    base.update(over)
    return ResearchLimits(**base)


class FakeProvider:
    def __init__(self, results=None, error=None):
        self._results = results or []
        self._error = error
        self.calls = 0

    async def search(self, query, max_results, timeout, domains):
        self.calls += 1
        if self._error:
            raise self._error
        return self._results


def _session(provider, limits=None, **kw):
    return ResearchSession(
        limits or _limits(), provider, resolver=kw.pop("resolver", PUBLIC), **kw
    )


# --- SSRF / egress guard -----------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://ok.com/x",              # non-https scheme
    "https://ok.com:8080/x",        # disallowed port
    "https://user:pw@ok.com/x",     # embedded credentials
    "ftp://ok.com/x",               # non-https scheme
    "https:///nohost",              # missing host
])
def test_guard_url_rejects_bad_urls(url):
    with pytest.raises(ResearchBlockedError):
        guard_url(url, resolver=PUBLIC)


@pytest.mark.parametrize("ip", [
    "10.0.0.5", "127.0.0.1", "169.254.169.254", "192.168.1.1",
    "172.16.0.1", "::1", "fd00::1", "0.0.0.0",
])
def test_guard_url_blocks_non_public_addresses(ip):
    resolver = _resolver({"target.com": ip})
    with pytest.raises(ResearchBlockedError):
        guard_url("https://target.com/x", resolver=resolver)


def test_guard_url_allows_public_host():
    assert guard_url("https://ok.com/path", resolver=PUBLIC) == "ok.com"


def test_guard_url_blocks_dns_rebinding_to_loopback():
    resolver = _resolver({"rebind.com": "127.0.0.1"})
    with pytest.raises(ResearchBlockedError):
        guard_url("https://rebind.com/x", resolver=resolver)


# --- domain allow/block ------------------------------------------------------

def test_domain_allow_block_policy():
    assert domain_allowed("x.b.com", ["b.com"], []) is True
    assert domain_allowed("a.com", ["b.com"], []) is False       # allow miss
    assert domain_allowed("x.b.com", [], ["b.com"]) is False     # blocked
    assert domain_allowed("x.b.com", ["b.com"], ["b.com"]) is False  # block wins
    assert domain_allowed("", [], []) is False


def test_guard_url_enforces_allow_and_block():
    with pytest.raises(ResearchBlockedError):
        guard_url("https://evil.com/x", allow=["ok.com"], resolver=PUBLIC)
    with pytest.raises(ResearchBlockedError):
        guard_url("https://ok.com/x", block=["ok.com"], resolver=PUBLIC)


# --- query validation & limits ----------------------------------------------

@pytest.mark.asyncio
async def test_empty_query_rejected():
    s = _session(FakeProvider())
    with pytest.raises(ResearchLimitError):
        await s.search("   ", "n1")


@pytest.mark.asyncio
async def test_query_length_limit():
    s = _session(FakeProvider(), _limits(max_query_chars=10))
    with pytest.raises(ResearchLimitError):
        await s.search("x" * 11, "n1")


@pytest.mark.asyncio
async def test_per_node_and_per_run_limits():
    provider = FakeProvider([RawResult("https://ok.com/1", "T", "s")])
    s = _session(provider, _limits(max_searches_per_node=2, max_searches_per_run=3))
    await s.search("q1", "n1")
    await s.search("q2", "n1")
    with pytest.raises(ResearchLimitError):
        await s.search("q3", "n1")  # per-node limit
    await s.search("q4", "n2")
    with pytest.raises(ResearchLimitError):
        await s.search("q5", "n2")  # per-run limit (3 used)


@pytest.mark.asyncio
async def test_cost_limit_enforced():
    provider = FakeProvider([RawResult("https://ok.com/1", "T", "s")])
    s = _session(provider, _limits(cost_per_search_usd=0.05, max_cost_usd=0.06))
    await s.search("q1", "n1")
    with pytest.raises(ResearchLimitError):
        await s.search("q2", "n2")


@pytest.mark.asyncio
async def test_wall_clock_limit_enforced():
    clock = types.SimpleNamespace(t=0.0)
    provider = FakeProvider([RawResult("https://ok.com/1", "T", "s")])
    s = ResearchSession(
        _limits(timeout_seconds=5.0), provider, resolver=PUBLIC,
        clock=lambda: clock.t,
    )
    clock.t = 10.0  # elapsed beyond budget
    with pytest.raises(ResearchLimitError):
        await s.search("q1", "n1")


# --- shaping, dedup, citations ----------------------------------------------

@pytest.mark.asyncio
async def test_results_are_bounded_and_guarded():
    provider = FakeProvider([
        RawResult("https://ok.com/1", "A", "x" * 999),
        RawResult("https://evil.com/2", "B", "leak"),  # dropped: non-public
        RawResult("https://ok.com/3", "C", "short"),
    ])
    resolver = _resolver({"ok.com": "93.184.216.34", "evil.com": "10.0.0.9"})
    s = _session(provider, _limits(snippet_chars=50), resolver=resolver)
    result = await s.search("q", "n1")
    urls = [c.url for c in result.citations]
    assert urls == ["https://ok.com/1", "https://ok.com/3"]
    assert len(result.citations[0].snippet) == 50   # truncated
    assert result.truncated is True


@pytest.mark.asyncio
async def test_byte_budget_truncates():
    provider = FakeProvider([
        RawResult("https://ok.com/1", "A", "a" * 40),
        RawResult("https://ok.com/2", "B", "b" * 40),
    ])
    s = _session(provider, _limits(max_bytes=50, snippet_chars=100))
    result = await s.search("q", "n1")
    total = sum(len(c.snippet) for c in result.citations)
    assert total <= 50
    assert result.truncated is True


@pytest.mark.asyncio
async def test_identical_queries_deduplicated():
    provider = FakeProvider([RawResult("https://ok.com/1", "T", "s")])
    tel = ResearchTelemetry()
    s = _session(provider, telemetry=tel)
    await s.search("same", "n1")
    await s.search("same", "n2")   # served from cache; provider not called again
    assert provider.calls == 1
    assert tel.cache_hits == 1
    assert s.searches == 1


# --- secret redaction --------------------------------------------------------

def test_mask_secret():
    assert mask_secret("token=ABC123 tail", "ABC123") == "token=*** tail"
    assert mask_secret("no secret", None) == "no secret"


@pytest.mark.asyncio
async def test_provider_key_never_appears_in_results_or_errors():
    key = "sk-supersecret-123"
    provider = FakeProvider(
        results=[RawResult("https://ok.com/1", f"has {key}", f"snip {key}")],
    )
    s = _session(provider, api_key=key)
    result = await s.search("q", "n1")
    assert key not in result.citations[0].title
    assert key not in result.citations[0].snippet

    boom = FakeProvider(error=RuntimeError(f"auth failed for {key}"))
    tel = ResearchTelemetry()
    s2 = _session(boom, api_key=key, telemetry=tel)
    with pytest.raises(ResearchError):
        await s2.search("q", "n1")
    assert tel.failures == 1


# --- telemetry ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_telemetry_snapshot_shape():
    provider = FakeProvider([RawResult("https://ok.com/1", "T", "hello")])
    tel = ResearchTelemetry()
    s = _session(provider, telemetry=tel)
    await s.search("q", "n1")
    snap = tel.snapshot()
    assert snap["searches"] == 1
    assert snap["results_retained"] == 1
    assert snap["distinct_domains"] == 1
    assert snap["total_cost_usd"] == pytest.approx(0.005)
    assert set(snap) >= {
        "searches", "failures", "cache_hits", "limit_stops",
        "results_retained", "bytes_retained", "truncations",
        "distinct_domains", "avg_latency_ms", "total_cost_usd",
    }


# --- progress events ---------------------------------------------------------

@pytest.mark.asyncio
async def test_progress_events_are_bounded():
    events = []

    async def progress(e):
        events.append(e)

    provider = FakeProvider([RawResult("https://ok.com/1", "T", "s")])
    s = _session(provider)
    await s.search("some query", "n1", progress=progress)
    types_seen = [e["type"] for e in events]
    assert "research_query_started" in types_seen
    assert "research_query_completed" in types_seen
    # No full query text, keys, or page bodies leak into events.
    for e in events:
        assert "query" not in e
        assert all("secret" not in str(v).lower() for v in e.values())


# --- factory -----------------------------------------------------------------

def _settings(**over):
    base = dict(
        enable_parkour_research=True,
        parkour_research_endpoint="https://search.example.com/api",
        parkour_research_api_key="k",
        parkour_research_allowlist=[], parkour_research_blocklist=[],
        parkour_research_max_searches_per_run=6,
        parkour_research_max_searches_per_node=2,
        parkour_research_max_query_chars=256, parkour_research_max_results=5,
        parkour_research_snippet_chars=500, parkour_research_max_bytes=200_000,
        parkour_research_timeout_seconds=15.0,
        parkour_research_cost_per_search_usd=0.005,
        parkour_research_max_cost_usd=0.1,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_factory_disabled_returns_none():
    assert build_research_session(_settings(enable_parkour_research=False)) is None


def test_factory_without_endpoint_returns_none():
    assert build_research_session(_settings(parkour_research_endpoint=None)) is None


def test_factory_builds_session_when_configured():
    s = build_research_session(_settings(), telemetry=ResearchTelemetry())
    assert isinstance(s, ResearchSession)


# --- worker injection --------------------------------------------------------

def _node(research=False):
    return SubtaskSpec(
        id="n1", task_type=TaskType.CHAT, role="r",
        system_prompt="sys", user_prompt="find current facts",
        dependencies=[], research=research,
    )


@pytest.mark.asyncio
async def test_inject_research_noop_without_session_or_flag():
    messages = [{"role": "system", "content": "sys"}]
    # No session
    assert await _inject_research(_node(True), messages, None, None) == ()
    # Session present but node not opted in
    provider = FakeProvider([RawResult("https://ok.com/1", "T", "s")])
    assert await _inject_research(_node(False), messages, _session(provider), None) == ()
    assert len(messages) == 1  # untouched


@pytest.mark.asyncio
async def test_inject_research_adds_sources_and_citations():
    messages = [{"role": "system", "content": "sys"}]
    provider = FakeProvider([RawResult("https://ok.com/1", "Title", "snippet")])
    citations = await _inject_research(_node(True), messages, _session(provider), None)
    assert len(citations) == 1
    assert citations[0]["url"] == "https://ok.com/1"
    assert any("web sources" in m["content"].lower() for m in messages)


@pytest.mark.asyncio
async def test_inject_research_failure_is_non_fatal():
    messages = [{"role": "system", "content": "sys"}]
    boom = FakeProvider(error=RuntimeError("provider down"))
    citations = await _inject_research(_node(True), messages, _session(boom), None)
    assert citations == ()
    assert len(messages) == 1  # no sources injected on failure


# --- gateway: client tools still rejected with research enabled --------------

def test_client_tools_rejected_even_with_research_enabled(monkeypatch):
    from fastapi.testclient import TestClient
    import nvidia_smartroute.gateway.server as srv

    monkeypatch.setattr(srv.settings, "enable_parkour", True)
    monkeypatch.setattr(srv.settings, "enable_parkour_research", True)
    client = TestClient(srv.app)
    response = client.post("/v1/chat/completions", json={
        "model": "parkour", "tools": [{"type": "function"}],
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "parkour_tools_unsupported"
