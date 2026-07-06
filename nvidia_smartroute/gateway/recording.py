# @spec[GATEWAY_API.md#Requirements]
"""
Per-request accounting helpers: cost, throughput, and streamed-response usage.
Shared by the completion and streaming paths.
"""

from ..metrics import metrics
from ..cost import budget, compute_cost
from ..routing.router import router


# @spec[COST.md#Requirements]
def record_cost(model, usage: dict) -> None:
    """Compute a request's USD cost from usage + model pricing and record it."""
    cost = compute_cost(
        model,
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )
    if cost:
        metrics.record_cost(model.model_id, cost)
        budget.record(cost)


# @spec[OBSERVABILITY.md#Requirements]
def record_throughput(model_id: str, usage: dict, latency_ms: float) -> None:
    """Record per-request generation throughput (tokens/sec) as a peak."""
    completion = int(usage.get("completion_tokens") or usage.get("total_tokens") or 0)
    if completion and latency_ms > 0:
        metrics.record_throughput(model_id, completion / (latency_ms / 1000.0))


# @spec[OBSERVABILITY.md#Requirements]
def record_stream_usage(model_id: str, usage: dict) -> None:
    """Record tokens + cost for a streamed response (usage in the final chunk)."""
    if not usage or not usage.get("total_tokens"):
        return
    metrics.record_tokens(model_id, int(usage["total_tokens"]))
    model = router.model_registry.get_model(model_id)
    if model:
        record_cost(model, usage)
