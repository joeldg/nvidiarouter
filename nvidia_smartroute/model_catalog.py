# @spec[PROJECT_PROFILE.md#Requirements]
"""
Capability inference for build.nvidia.com (NIM) models.

The NIM `/v1/models` endpoint returns model IDs but no structured metadata
(size, modality, capabilities). This module enriches an ID into a capability
profile: first from a curated table of notable models, then by pattern-matching
the ID (size tokens, vision/code keywords) so unknown models still get a
sensible profile. Used by the `discover` command to populate the router.
"""

import re
from typing import Any, Dict, List

from .routing.router import TaskType

# General instruct/chat task set shared by most large text models.
_GENERAL_TASKS = [
    TaskType.CHAT,
    TaskType.REASONING,
    TaskType.MATHEMATICS,
    TaskType.SUMMARIZATION,
    TaskType.QUESTION_ANSWERING,
    TaskType.TRANSLATION,
]
_CODE_TASKS = [
    TaskType.CODE_GENERATION,
    TaskType.CODE_COMPLETION,
    TaskType.CODE_EXPLANATION,
    TaskType.CODE_REVIEW,
]

# Curated metadata for notable models (approx params in billions). Values are
# best-effort and used for display/scoring, not billing.
# @spec[PROJECT_PROFILE.md#Requirements]
_CURATED: Dict[str, Dict[str, Any]] = {
    # Kimi / GLM ship without a size token in the ID, so curate them explicitly.
    "moonshotai/kimi-k2.6": {"parameters_b": 1000, "context_window": 256000},
    "moonshotai/kimi-k2-instruct": {"parameters_b": 1000, "context_window": 128000},
    "z-ai/glm-5.2": {"parameters_b": 355, "context_window": 200000},
    "zai-org/glm-4.6": {"parameters_b": 357, "context_window": 200000},
    "zai-org/glm-4.5-air": {"parameters_b": 106, "context_window": 128000},
    "deepseek-ai/deepseek-r1": {"parameters_b": 671, "context_window": 128000,
                                "tags": ["reasoning"]},
    "deepseek-ai/deepseek-v3": {"parameters_b": 671, "context_window": 128000},
    "meta/llama-3.1-405b-instruct": {"parameters_b": 405, "context_window": 128000},
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": {"parameters_b": 253},
    "qwen/qwen3.5-397b-a17b": {"parameters_b": 397, "context_window": 128000},
    "qwen/qwen3-next-80b-a3b-instruct": {"parameters_b": 80},
    "nvidia/llama-3.3-nemotron-super-49b-v1": {"parameters_b": 49},
    "meta/llama-3.3-70b-instruct": {"parameters_b": 70, "context_window": 128000},
    "meta/llama-3.1-70b-instruct": {"parameters_b": 70},
    "meta/llama-3.1-8b-instruct": {"parameters_b": 8},
    "mistralai/mixtral-8x22b-instruct-v0.1": {"parameters_b": 141},
    "mistralai/codestral-22b-instruct-v0.1": {"parameters_b": 22},
    "meta/llama-3.2-90b-vision-instruct": {"parameters_b": 90},
    "meta/llama-3.2-11b-vision-instruct": {"parameters_b": 11},
}

_VISION_HINTS = ("vision", "vlm", "-vl-", "neva", "pixtral", "internvl", "kosmos")
_CODE_HINTS = ("coder", "codestral", "codegemma", "starcoder", "code-", "-code")
_EMBED_HINTS = ("embed", "embedqa", "rerank")
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?:\b|-|_|$)", re.IGNORECASE)


def _extract_params_b(model_id: str) -> float:
    """Infer parameter count (billions) from size tokens like '70b' in the ID."""
    best = 0.0
    for match in _SIZE_RE.finditer(model_id.lower()):
        try:
            best = max(best, float(match.group(1)))
        except ValueError:
            continue
    return best


# @spec[PROJECT_PROFILE.md#Requirements]
def is_embedding_model(model_id: str) -> bool:
    mid = model_id.lower()
    return any(h in mid for h in _EMBED_HINTS)


# @spec[PROJECT_PROFILE.md#Requirements]
def infer_capability(model_id: str) -> Dict[str, Any]:
    """Return ModelCapability kwargs for a model ID (curated + inferred)."""
    mid = model_id.lower()
    provider = model_id.split("/")[0] if "/" in model_id else "nvidia"
    curated = _CURATED.get(model_id, {})

    is_vision = any(h in mid for h in _VISION_HINTS)
    is_code = any(h in mid for h in _CODE_HINTS)

    if "tasks" in curated:
        tasks = list(curated["tasks"])
    elif is_vision:
        tasks = [TaskType.VISION]
    elif is_code:
        tasks = list(_CODE_TASKS)
    else:
        tasks = list(_GENERAL_TASKS)

    params_b = curated.get("parameters_b") or _extract_params_b(model_id)
    # Larger models are treated as higher-quality by default; small = faster.
    quality = min(0.75 + params_b / 800.0, 0.98) if params_b else 0.82

    return {
        "model_id": model_id,
        "name": model_id.split("/")[-1],
        "provider": provider,
        "version": "1.0",
        "supported_tasks": tasks,
        "parameters_b": float(params_b),
        "context_window": int(curated.get("context_window", 32768)),
        "quality_score": round(quality, 3),
        "reliability_score": 0.85,
        "supports_streaming": True,
        "supports_vision": is_vision,
        "supports_function_calling": not is_vision and params_b >= 7,
        "description": model_id,
        "tags": list(curated.get("tags", [])),
    }


# @spec[PROJECT_PROFILE.md#Requirements]
def rank_by_capability(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort capability profiles by size then quality (largest/best first)."""
    return sorted(
        profiles,
        key=lambda p: (p.get("parameters_b", 0.0), p.get("quality_score", 0.0)),
        reverse=True,
    )
