# @spec[ROUTING.md#Acceptance Evidence]
"""
Tests for optional session affinity, mapped to ROUTING.md req.9-11.
"""

import asyncio

import pytest

import nvidia_smartroute.config as cfg
import nvidia_smartroute.metrics as M
from nvidia_smartroute.affinity import SessionAffinity, session_affinity
from nvidia_smartroute.circuit import breaker
from nvidia_smartroute.routing.router import (
    ModelCapability,
    RequestRouter,
    TaskType,
)


def _model(mid, quality=0.85, params=20.0):
    return ModelCapability(
        model_id=mid, name=mid, provider="x", version="1",
        supported_tasks=[TaskType.CHAT], quality_score=quality,
        reliability_score=0.9, latency_ms=500, parameters_b=params,
        throughput_tps=0.0, input_cost_per_1k=0.0, output_cost_per_1k=0.0,
    )


def _router(models):
    r = RequestRouter()
    r.model_registry.models = {m.model_id: m for m in models}
    return r


def _route(r, key=None, model=None):
    msg = [{"role": "user", "content": "Hello, how are you today?"}]
    return asyncio.run(r.route_request(msg, model=model, session_key=key))


@pytest.fixture(autouse=True)
def _clean():
    M.metrics.reset()
    session_affinity.reset()
    breaker.reset()
    yield
    session_affinity.reset()
    breaker.reset()


# req.9: default OFF -> stateless, no pin read or written.
def test_affinity_off_is_stateless(monkeypatch):
    monkeypatch.setattr(cfg.settings, "session_affinity", False)
    r = _router([_model("m_hi", quality=0.95), _model("m_lo", quality=0.60)])
    # Even a pre-seeded pin must be ignored when the feature is off.
    session_affinity.set("s1", "m_lo")
    dec = _route(r, key="s1")
    assert dec.selected_model.model_id == "m_hi"   # normal best, not the pin
    assert dec.from_session is False
    assert session_affinity.get("s1") == "m_lo"    # untouched (not re-pinned)


# req.9: ON + session key -> reuse the pinned model over normal scoring.
def test_affinity_reuses_pinned_model(monkeypatch):
    monkeypatch.setattr(cfg.settings, "session_affinity", True)
    r = _router([_model("m_hi", quality=0.95), _model("m_lo", quality=0.60)])
    # Seed a pin to the NON-default model; the pin must win over scoring.
    session_affinity.set("s2", "m_lo")
    dec = _route(r, key="s2")
    assert dec.selected_model.model_id == "m_lo"
    assert dec.from_session is True
    # First-touch of a fresh session records the normal pick.
    fresh = _route(r, key="s-new")
    assert fresh.selected_model.model_id == "m_hi"
    assert fresh.from_session is False
    assert session_affinity.get("s-new") == "m_hi"


# req.9: no session key -> stateless even when the feature is on.
def test_affinity_on_but_no_key_is_stateless(monkeypatch):
    monkeypatch.setattr(cfg.settings, "session_affinity", True)
    r = _router([_model("m_hi", quality=0.95), _model("m_lo", quality=0.60)])
    dec = _route(r, key=None)
    assert dec.selected_model.model_id == "m_hi"
    assert dec.from_session is False
    assert session_affinity.size() == 0


# req.10: a deregistered pin fails safe -> re-route by scoring and re-pin.
def test_affinity_failsafe_on_deregistered_pin(monkeypatch):
    monkeypatch.setattr(cfg.settings, "session_affinity", True)
    r = _router([_model("m_hi", quality=0.95), _model("m_lo", quality=0.60)])
    session_affinity.set("s3", "ghost-model")  # not in the registry
    dec = _route(r, key="s3")
    assert dec.selected_model.model_id == "m_hi"       # normal selection
    assert dec.from_session is False
    assert session_affinity.get("s3") == "m_hi"        # re-pinned to served model


# req.10: a circuit-broken pin fails safe -> re-route by scoring and re-pin.
def test_affinity_failsafe_on_circuit_broken_pin(monkeypatch):
    monkeypatch.setattr(cfg.settings, "session_affinity", True)
    monkeypatch.setattr(cfg.settings, "circuit_breaker_enabled", True)
    r = _router([_model("m_hi", quality=0.95), _model("m_lo", quality=0.60)])
    session_affinity.set("s4", "m_lo")
    for _ in range(3):                    # trip m_lo's circuit open
        breaker.record_failure("m_lo")
    assert breaker.allow("m_lo") is False
    dec = _route(r, key="s4")
    assert dec.selected_model.model_id == "m_hi"       # pinned model was skipped
    assert dec.from_session is False
    assert session_affinity.get("s4") == "m_hi"        # re-pinned to healthy model


# req.10: an explicit model override beats the pin and is NOT stored.
def test_explicit_model_beats_pin_and_is_not_stored(monkeypatch):
    monkeypatch.setattr(cfg.settings, "session_affinity", True)
    r = _router([_model("m_hi", quality=0.95), _model("m_lo", quality=0.60)])
    session_affinity.set("s5", "m_lo")
    dec = _route(r, key="s5", model="m_hi")
    assert dec.selected_model.model_id == "m_hi"
    assert dec.from_session is False
    assert session_affinity.get("s5") == "m_lo"        # pin unchanged by override


# req.11: the store is bounded (LRU) and honors TTL expiry.
def test_affinity_store_is_bounded_and_ttl_expires():
    store = SessionAffinity(max_entries=2, ttl=100)
    store.set("a", "m1")
    store.set("b", "m2")
    store.set("c", "m3")                  # evicts LRU ("a")
    assert store.get("a") is None
    assert store.get("b") == "m2"
    assert store.get("c") == "m3"
    assert store.size() == 2

    # An entry past its TTL is dropped on read.
    expired = SessionAffinity(max_entries=10, ttl=100)
    expired.set("k", "m")
    expired._store["k"] = (0.0, "m")      # force an already-expired timestamp
    assert expired.get("k") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
