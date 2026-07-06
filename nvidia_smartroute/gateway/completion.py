# @spec[PROJECT_PROFILE.md#Requirements]
"""
Non-streaming completion with model fallback + circuit breaker + adaptive
routing feedback. Tries the routed model, then the next-best models for the task
on hard upstream failures, recording latency/tokens/cost/outcomes along the way.
"""

import time
from typing import Optional

import structlog

from ..config import settings
from ..metrics import metrics
from ..circuit import breaker
from ..routing.router import router
from ..bandit import adaptive_router, reward_from
from ..keypool import KeyPoolExhaustedError
from .nim_client import nim_client
from .recording import record_cost, record_throughput

logger = structlog.get_logger()


# @spec[PROJECT_PROFILE.md#Requirements]
def should_fallback(exc: Exception) -> bool:
    """Whether an upstream failure warrants trying a different model.

    Fall back on 404 (model unavailable), 5xx, and transport errors; not on
    other 4xx (a bad request will fail identically on every model).
    """
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = resp.status_code
        return code == 404 or code >= 500
    return True


# @spec[PROJECT_PROFILE.md#Requirements]
async def complete_with_fallback(  # noqa: C901
    task_type,
    primary,
    messages: list,
    max_tokens,
    temperature,
    extra: dict,
):
    """Call the primary model, failing over to next-best models on hard errors.

    Returns (response_data, used_model, fallback_used).
    """
    candidates = [primary]
    if settings.enable_model_fallback:
        for m in router.model_registry.rank_models(task_type):
            if all(m.model_id != c.model_id for c in candidates):
                candidates.append(m)
        candidates = candidates[: 1 + settings.max_model_fallbacks]

    # Circuit breaker: skip models whose circuit is open. If that would leave
    # nothing, keep the original list (better to try than to hard-fail).
    if settings.circuit_breaker_enabled:
        healthy = [m for m in candidates if breaker.allow(m.model_id)]
        if healthy:
            candidates = healthy

    task = getattr(task_type, "value", str(task_type))
    last_exc: Optional[Exception] = None
    for index, model in enumerate(candidates):
        nim_start = time.time()
        try:
            data = await nim_client.chat_completions(
                model=model.model_id,
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
            latency_ms = (time.time() - nim_start) * 1000.0
            metrics.record_latency(model.model_id, latency_ms)
            usage = data.get("usage") if isinstance(data, dict) else None
            if isinstance(usage, dict) and usage.get("total_tokens"):
                metrics.record_tokens(model.model_id, int(usage["total_tokens"]))
                record_cost(model, usage)
                record_throughput(model.model_id, usage, latency_ms)
            if settings.circuit_breaker_enabled:
                breaker.record_success(model.model_id)
            adaptive_router.record(task, model.model_id, reward_from(True, latency_ms))
            return data, model, model.model_id != primary.model_id
        except KeyPoolExhaustedError:
            raise  # a key problem, not a model problem — don't fail over
        except Exception as exc:
            metrics.record_error(model.model_id)
            # Only hard upstream failures count against the model's circuit.
            if settings.circuit_breaker_enabled and should_fallback(exc):
                breaker.record_failure(model.model_id)
            adaptive_router.record(task, model.model_id, 0.0)
            last_exc = exc
            if index + 1 < len(candidates) and should_fallback(exc):
                logger.warning(
                    "model failed; trying fallback",
                    model=model.model_id,
                    error=str(exc) or repr(exc),
                )
                continue
            raise
    raise last_exc if last_exc else RuntimeError("no model candidates")
