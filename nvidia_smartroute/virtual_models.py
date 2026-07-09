# @spec[PARKOUR.md#Requirements]
"""Descriptors for local virtual models exposed by the gateway."""

from typing import Any, Dict, List

from .config import Settings

PARKOUR_MODEL_ID = "parkour"


# @spec[PARKOUR.md#Requirements]
def list_virtual_models(settings: Settings, created: int) -> List[Dict[str, Any]]:
    """Return enabled virtual models in OpenAI's model-list shape.

    Virtual models deliberately live outside ``ModelRegistry``. That registry
    contains upstream routing candidates, while entries returned here describe
    local execution strategies that a client must select explicitly.
    """
    if not settings.enable_parkour:
        return []
    return [
        {
            "id": PARKOUR_MODEL_ID,
            "object": "model",
            "created": created,
            "owned_by": "nvidia-smartroute",
            "model_type": "virtual",
            "execution_strategy": "multi_agent_dag",
            "display_name": "PARKOUR",
            "supports_streaming": True,
        }
    ]
