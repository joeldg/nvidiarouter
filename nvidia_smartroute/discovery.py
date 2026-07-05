# @spec[PROJECT_PROFILE.md#Requirements]
"""
Live model discovery for build.nvidia.com (NIM).

Fetches the NIM catalog, probes which models are actually servable for the
account (a 200 to a 1-token chat request), enriches each with a capability
profile, and persists them so the router can route across the full set instead
of a hardcoded handful.
"""

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from .routing.router import ModelCapability, TaskType
from .model_catalog import infer_capability, is_routable

_FIELD_NAMES = {f.name for f in fields(ModelCapability)}


# @spec[PROJECT_PROFILE.md#Requirements]
def serialize(cap: ModelCapability) -> Dict[str, Any]:
    """ModelCapability -> JSON-safe dict (enums to their values)."""
    data = asdict(cap)
    data["supported_tasks"] = [t.value for t in cap.supported_tasks]
    return data


# @spec[PROJECT_PROFILE.md#Requirements]
def deserialize(data: Dict[str, Any]) -> ModelCapability:
    """JSON dict -> ModelCapability (ignores unknown keys for forward-compat)."""
    kwargs = {k: v for k, v in data.items() if k in _FIELD_NAMES}
    kwargs["supported_tasks"] = [TaskType(v) for v in data.get("supported_tasks", [])]
    return ModelCapability(**kwargs)


# @spec[PROJECT_PROFILE.md#Requirements]
def save_models(path: str, caps: List[ModelCapability]) -> None:
    Path(path).write_text(json.dumps([serialize(c) for c in caps], indent=2))


# @spec[PROJECT_PROFILE.md#Requirements]
def load_models(path: str) -> List[ModelCapability]:
    data = json.loads(Path(path).read_text())
    return [deserialize(d) for d in data]


# @spec[PROJECT_PROFILE.md#Requirements]
def fetch_catalog(base_url: str, api_key: str, timeout: float = 20.0) -> List[str]:
    """Return the list of model IDs from the NIM /models endpoint."""
    resp = httpx.get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return sorted(m["id"] for m in resp.json().get("data", []))


# @spec[PROJECT_PROFILE.md#Requirements]
def probe_servable(base_url: str, api_key: str, model_id: str, timeout: float = 30.0) -> bool:
    """True if a 1-token chat request to the model returns 200 for this account."""
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            json={"model": model_id, "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1},
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


# @spec[PROJECT_PROFILE.md#Requirements]
def discover(
    base_url: str,
    api_key: str,
    probe: bool = True,
    include_embeddings: bool = False,
    limit: Optional[int] = None,
    on_result: Optional[Callable[[str, bool], None]] = None,
) -> Tuple[List[ModelCapability], Dict[str, Any]]:
    """
    Discover servable models and return (capabilities, report).

    When ``probe`` is False, every catalog model is enriched without a live
    servability check (useful for offline runs).
    """
    catalog = fetch_catalog(base_url, api_key)
    if not include_embeddings:
        # Skip embeddings and specialized non-chat models (guard/safety/etc.).
        catalog = [m for m in catalog if is_routable(m)]
    if limit is not None:
        catalog = catalog[:limit]

    caps: List[ModelCapability] = []
    servable, unservable = 0, 0
    for model_id in catalog:
        ok = probe_servable(base_url, api_key, model_id) if probe else True
        if on_result:
            on_result(model_id, ok)
        if ok:
            caps.append(deserialize(infer_capability(model_id)))
            servable += 1
        else:
            unservable += 1

    report = {
        "catalog_size": len(catalog),
        "servable": servable,
        "unservable": unservable,
        "probed": probe,
    }
    return caps, report
