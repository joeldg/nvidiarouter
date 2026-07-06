# @spec[RECOMMENDATION.md#Requirements]
"""
Model recommendation advisor.

For each task type, selects the best registered model using the router's own
scoring (so recommendations never contradict real routing) and returns an
explainable rationale. Pure and read-only: no upstream NIM call, no state
mutation, no network — a function of the registry + live metrics only.
"""

from typing import Any, Dict, Optional


# @spec[RECOMMENDATION.md#Requirements]
def _rationale(registry, model) -> tuple:
    """Return (basis, rationale) for a chosen model.

    Prefers observed (measured) latency over the size-based estimate and records
    which basis was used (RECOMMENDATION.md req.4).
    """
    from .metrics import metrics

    live = metrics.get_avg_latency_ms(model.model_id)
    if live is not None:
        basis, latency = "measured", live
    else:
        basis = "estimated"
        latency = float(model.latency_ms or (250 + model.parameters_b))
    return basis, {
        "parameters_b": model.parameters_b,
        "latency_ms": round(latency, 1),
        "throughput_tps": round(model.throughput_tps, 1),
        "output_cost_per_1k": model.output_cost_per_1k,
        "quality_score": model.quality_score,
        "score": round(registry._score_model(model), 4),
    }


# @spec[RECOMMENDATION.md#Requirements]
def recommend_all(registry=None) -> Dict[str, Any]:
    """Recommend the best model for every task type.

    For each task, ranks only the models that actually support it by the router's
    scoring and returns the top pick plus alternatives; reports no model when the
    task is unsupported (req.1, req.2, req.3, req.5, req.8).
    """
    from .routing.router import TaskType, router

    registry = registry if registry is not None else router.model_registry
    out: Dict[str, Any] = {}
    for task in TaskType:
        supporting = [m for m in registry.models.values() if task in m.supported_tasks]
        if not supporting:
            out[task.value] = {
                "model": None,
                "basis": None,
                "rationale": "no registered model supports this task",
                "alternatives": [],
            }
            continue
        ranked = sorted(supporting, key=registry._score_model, reverse=True)
        best = ranked[0]
        basis, rationale = _rationale(registry, best)
        out[task.value] = {
            "model": best.model_id,
            "basis": basis,
            "rationale": rationale,
            "alternatives": [
                {"model": m.model_id, "score": round(registry._score_model(m), 4)}
                for m in ranked[1:3]
            ],
        }
    return out


# @spec[RECOMMENDATION.md#Requirements]
def recommend_for(task_name: str, registry=None) -> Optional[Dict[str, Any]]:
    """Recommendation for a single task, or None if the task name is unknown."""
    if not is_task(task_name):
        return None
    return {task_name: recommend_all(registry)[task_name]}


# @spec[RECOMMENDATION.md#Requirements]
def is_task(name: str) -> bool:
    """Whether `name` is a valid TaskType value."""
    from .routing.router import TaskType

    return name in {t.value for t in TaskType}
