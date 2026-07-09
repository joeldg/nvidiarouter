# @spec[PARKOUR.md#Requirements]
"""Pure PARKOUR execution-plan parsing and DAG validation."""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from ..routing.router import TaskType

_NODE_ID = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_COMPLEXITY_MARKERS = (
    "step by step",
    "step-by-step",
    "and then",
    "compare and",
    "research and",
    "implement and test",
    "write and test",
    "review and revise",
    "multiple approaches",
    "pros and cons",
    "end to end",
    "end-to-end",
)


class PlanValidationError(ValueError):
    """Raised when conductor output is not a safe, schedulable execution plan."""


# @spec[PARKOUR.md#Requirements]
class SubtaskSpec(BaseModel):
    """One validated node in a PARKOUR execution graph."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=32)
    task_type: TaskType
    role: str = Field(min_length=1, max_length=80)
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    dependencies: List[str] = Field(default_factory=list)
    optional: bool = False
    # Server/conductor-controlled opt-in to the governed research lane. Ignored
    # unless ENABLE_PARKOUR_RESEARCH is set; never client-parameterizable.
    # @spec[PARKOUR_RESEARCH.md#Requirements]
    research: bool = False


# @spec[PARKOUR.md#Requirements]
class ExecutionPlan(BaseModel):
    """A conductor-produced DAG plus optional final synthesis instructions."""

    model_config = ConfigDict(extra="forbid")

    tasks: List[SubtaskSpec] = Field(min_length=1)
    synthesis_instructions: Optional[str] = Field(default=None, min_length=1)

    @property
    def requires_synthesis(self) -> bool:
        """Whether this plan requests a separate synthesis pass."""
        return bool(self.synthesis_instructions)


# @spec[PARKOUR.md#Requirements]
@dataclass(frozen=True)
class PlanLimits:
    """Structural limits applied before any graph node is scheduled."""

    max_nodes: int
    max_depth: int
    max_width: int
    max_prompt_chars: int

    @classmethod
    def from_settings(cls, settings: Any) -> "PlanLimits":
        """Build plan limits from the PARKOUR settings group."""
        return cls(
            max_nodes=settings.parkour_max_nodes,
            max_depth=settings.parkour_max_depth,
            max_width=settings.parkour_max_concurrency,
            max_prompt_chars=settings.parkour_max_context_chars,
        )


@dataclass(frozen=True)
class GraphAnalysis:
    """Deterministic scheduling facts for a validated execution plan."""

    topological_order: Tuple[str, ...]
    depth: int
    width: int


def _decode_plan(value: Union[str, Mapping[str, Any], ExecutionPlan]) -> Any:
    if isinstance(value, ExecutionPlan):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise PlanValidationError("execution plan must be an object or JSON string")

    try:
        return json.loads(value)
    except json.JSONDecodeError as direct_error:
        match = _FENCED_JSON.search(value)
        if not match:
            raise PlanValidationError("execution plan is not valid JSON") from direct_error
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as fenced_error:
            raise PlanValidationError(
                "fenced execution plan is not valid JSON"
            ) from fenced_error


# @spec[PARKOUR.md#Requirements]
def parse_execution_plan(
    value: Union[str, Mapping[str, Any], ExecutionPlan],
    limits: PlanLimits,
) -> ExecutionPlan:
    """Parse conductor output and return a fully validated, schedulable plan."""
    try:
        decoded = _decode_plan(value)
        plan = decoded if isinstance(decoded, ExecutionPlan) else ExecutionPlan.model_validate(decoded)
    except PlanValidationError:
        raise
    except Exception as exc:
        raise PlanValidationError(f"execution plan schema is invalid: {exc}") from exc
    analyze_plan(plan, limits)
    return plan


# @spec[PARKOUR.md#Requirements]
def analyze_plan(plan: ExecutionPlan, limits: PlanLimits) -> GraphAnalysis:
    """Validate graph references and limits, returning deterministic DAG facts."""
    if len(plan.tasks) > limits.max_nodes:
        raise PlanValidationError(
            f"execution plan has {len(plan.tasks)} nodes; limit is {limits.max_nodes}"
        )

    ids = _validate_nodes(plan, limits)
    return _topological_analysis(plan, ids, limits)


def _validate_nodes(plan: ExecutionPlan, limits: PlanLimits) -> List[str]:
    """Validate node identity, references, and prompt bounds."""
    ids = [task.id for task in plan.tasks]
    if len(ids) != len(set(ids)):
        raise PlanValidationError("execution plan contains duplicate node IDs")
    invalid_ids = [node_id for node_id in ids if not _NODE_ID.fullmatch(node_id)]
    if invalid_ids:
        raise PlanValidationError(f"invalid node ID: {invalid_ids[0]}")

    known = set(ids)
    for task in plan.tasks:
        if len(task.dependencies) != len(set(task.dependencies)):
            raise PlanValidationError(f"node '{task.id}' repeats a dependency")
        if task.id in task.dependencies:
            raise PlanValidationError(f"node '{task.id}' depends on itself")
        missing = [dep for dep in task.dependencies if dep not in known]
        if missing:
            raise PlanValidationError(
                f"node '{task.id}' depends on missing node '{missing[0]}'"
            )
        prompt_chars = len(task.system_prompt) + len(task.user_prompt)
        if prompt_chars > limits.max_prompt_chars:
            raise PlanValidationError(
                f"node '{task.id}' prompt has {prompt_chars} characters; "
                f"limit is {limits.max_prompt_chars}"
            )
    return ids


def _topological_analysis(
    plan: ExecutionPlan,
    ids: List[str],
    limits: PlanLimits,
) -> GraphAnalysis:
    """Run deterministic Kahn traversal and enforce graph depth/width."""
    children: Dict[str, List[str]] = {node_id: [] for node_id in ids}
    indegree = {task.id: len(task.dependencies) for task in plan.tasks}
    for task in plan.tasks:
        for dependency in task.dependencies:
            children[dependency].append(task.id)

    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    order: List[str] = []
    depth = 0
    width = 0
    while ready:
        depth += 1
        width = max(width, len(ready))
        if depth > limits.max_depth:
            raise PlanValidationError(
                f"execution plan depth exceeds limit {limits.max_depth}"
            )
        if len(ready) > limits.max_width:
            raise PlanValidationError(
                f"execution plan width {len(ready)} exceeds limit {limits.max_width}"
            )
        current = ready
        ready = []
        for node_id in current:
            order.append(node_id)
            for child in sorted(children[node_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        ready.sort()

    if len(order) != len(ids):
        raise PlanValidationError("execution plan contains a dependency cycle")
    return GraphAnalysis(tuple(order), depth, width)


def _message_text(messages: Iterable[Mapping[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            )
    return "\n".join(part for part in parts if part).strip()


# @spec[PARKOUR.md#Requirements]
def should_decompose(messages: Iterable[Mapping[str, Any]]) -> bool:
    """Return whether local evidence justifies paying for conductor decomposition."""
    text = _message_text(messages).lower()
    if len(text) > 600:
        return True
    return any(marker in text for marker in _COMPLEXITY_MARKERS)


# @spec[PARKOUR.md#Requirements]
def build_direct_plan(
    messages: Iterable[Mapping[str, Any]],
    task_type: TaskType = TaskType.CHAT,
) -> ExecutionPlan:
    """Build the single-worker, no-synthesis plan used for simple requests."""
    prompt = _message_text(messages)
    if not prompt:
        raise PlanValidationError("cannot build a direct plan from empty messages")
    return ExecutionPlan(
        tasks=[
            SubtaskSpec(
                id="direct",
                task_type=task_type,
                role="direct responder",
                system_prompt="Answer the user's request directly and accurately.",
                user_prompt=prompt,
                dependencies=[],
            )
        ],
        synthesis_instructions=None,
    )
