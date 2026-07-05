# @spec[PROJECT_PROFILE.md#Requirements]
"""
Dynamic Agent Autoscale Engine.

When a routed request is a complex, multi-step task the gateway can spin up
secondary sub-agents to parallelize the work (e.g. a code-writing agent and a
code-testing/review agent). Each sub-agent issues a real NVIDIA NIM chat
completion via an injected async callable, so the engine is transport-agnostic
and testable.

The engine bounds fan-out with a concurrency semaphore
(``max_concurrent_agents``) and guards each sub-agent with a timeout
(``agent_timeout``).
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import structlog

from ..config import settings
from ..routing.router import TaskType

logger = structlog.get_logger()

# An injected NIM caller: (model_id, messages, **kwargs) -> OpenAI-format dict.
NIMCaller = Callable[..., Awaitable[Dict[str, Any]]]


# @spec[PROJECT_PROFILE.md#Requirements]
@dataclass
class SubAgent:
    """A specialized sub-agent with its own role and system prompt."""

    name: str
    role: str
    system_prompt: str


# @spec[PROJECT_PROFILE.md#Requirements]
@dataclass
class SubAgentResult:
    """The outcome of a single sub-agent execution."""

    name: str
    role: str
    content: str
    model_id: str
    latency_ms: float
    error: Optional[str] = None


# @spec[PROJECT_PROFILE.md#Requirements]
# Multi-step signals in a code request that justify fanning out to sub-agents.
_MULTISTEP_MARKERS = (
    "and test",
    "with test",
    "unit test",
    "then ",
    "step by step",
    "step-by-step",
    "and verify",
    "production-ready",
    "production ready",
    "end to end",
    "end-to-end",
    "and document",
)

_CODE_TASKS = {
    TaskType.CODE_GENERATION,
    TaskType.CODE_COMPLETION,
    TaskType.CODE_REVIEW,
}


# @spec[PROJECT_PROFILE.md#Requirements]
class AutoscaleEngine:
    """Coordinates specialized sub-agents for complex tasks."""

    def __init__(self, max_concurrent: Optional[int] = None, timeout: Optional[int] = None):
        self.max_concurrent = max_concurrent or settings.max_concurrent_agents
        self.timeout = timeout or settings.agent_timeout

    # @spec[PROJECT_PROFILE.md#Requirements]
    def should_scale(self, task_type: TaskType, messages: List[Dict[str, Any]]) -> bool:
        """
        Decide whether a request warrants multi-agent orchestration.

        Heuristic: enabled in config, the task is code-oriented, and the prompt
        shows multi-step intent (explicit markers or a sizable request).
        """
        if not settings.enable_autoscale:
            return False
        if task_type not in _CODE_TASKS:
            return False

        text = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for msg in messages
            for part in (
                msg.get("content", []) if isinstance(msg.get("content"), list) else [msg.get("content", "")]
            )
        ).lower()

        if any(marker in text for marker in _MULTISTEP_MARKERS):
            return True
        # Long, code-generation requests tend to be multi-step.
        return task_type == TaskType.CODE_GENERATION and len(text) > 400

    # @spec[PROJECT_PROFILE.md#Requirements]
    async def _run_agent(
        self,
        agent: SubAgent,
        model_id: str,
        base_messages: List[Dict[str, Any]],
        nim_call: NIMCaller,
        semaphore: asyncio.Semaphore,
        extra_context: Optional[str] = None,
    ) -> SubAgentResult:
        """Execute one sub-agent under the concurrency + timeout guards."""
        messages: List[Dict[str, Any]] = [{"role": "system", "content": agent.system_prompt}]
        messages.extend(base_messages)
        if extra_context:
            messages.append({"role": "user", "content": extra_context})

        start = time.time()
        async with semaphore:
            try:
                response = await asyncio.wait_for(
                    nim_call(model=model_id, messages=messages),
                    timeout=self.timeout,
                )
                content = _extract_content(response)
                return SubAgentResult(
                    name=agent.name,
                    role=agent.role,
                    content=content,
                    model_id=model_id,
                    latency_ms=(time.time() - start) * 1000.0,
                )
            except asyncio.TimeoutError:
                logger.warning("sub-agent timed out", agent=agent.name, timeout=self.timeout)
                return SubAgentResult(
                    name=agent.name,
                    role=agent.role,
                    content="",
                    model_id=model_id,
                    latency_ms=(time.time() - start) * 1000.0,
                    error=f"timeout after {self.timeout}s",
                )
            except Exception as exc:  # pragma: no cover - defensive
                message = str(exc) or repr(exc)
                logger.error("sub-agent failed", agent=agent.name, error=message)
                return SubAgentResult(
                    name=agent.name,
                    role=agent.role,
                    content="",
                    model_id=model_id,
                    latency_ms=(time.time() - start) * 1000.0,
                    error=message,
                )

    # @spec[PROJECT_PROFILE.md#Requirements]
    async def orchestrate(
        self,
        messages: List[Dict[str, Any]],
        model_id: str,
        nim_call: NIMCaller,
    ) -> Dict[str, Any]:
        """
        Run the writer -> (tester || reviewer) pipeline and compose a result.

        The writer produces an implementation; the tester and reviewer then run
        concurrently against the writer's output. Returns a structured dict with
        the composed content and per-agent metadata.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: List[SubAgentResult] = []

        writer = SubAgent(
            name="code-writer",
            role="writer",
            system_prompt=(
                "You are a senior code-writing agent. Produce a complete, "
                "correct, production-grade implementation for the user's request. "
                "Return only the code with brief inline comments."
            ),
        )
        writer_result = await self._run_agent(writer, model_id, messages, nim_call, semaphore)
        results.append(writer_result)

        # If the writer failed, don't fan out further.
        if writer_result.error or not writer_result.content:
            return self._compose(results)

        review_context = (
            "Here is an implementation produced for the request:\n\n"
            f"{writer_result.content}\n\n"
            "Analyze it as instructed by your role."
        )
        tester = SubAgent(
            name="code-tester",
            role="tester",
            system_prompt=(
                "You are a code-testing agent. Given an implementation, write a "
                "thorough suite of unit tests and note any failing edge cases."
            ),
        )
        reviewer = SubAgent(
            name="code-reviewer",
            role="reviewer",
            system_prompt=(
                "You are a code-review agent. Given an implementation, identify "
                "correctness bugs, security issues, and concrete improvements."
            ),
        )

        # Tester and reviewer are independent -> run them concurrently.
        followups = await asyncio.gather(
            self._run_agent(tester, model_id, messages, nim_call, semaphore, review_context),
            self._run_agent(reviewer, model_id, messages, nim_call, semaphore, review_context),
        )
        results.extend(followups)
        return self._compose(results)

    # @spec[PROJECT_PROFILE.md#Requirements]
    @staticmethod
    def _compose(results: List[SubAgentResult]) -> Dict[str, Any]:
        """Merge sub-agent outputs into a single structured result."""
        sections: List[str] = []
        heading = {
            "writer": "## Implementation",
            "tester": "## Tests",
            "reviewer": "## Review",
        }
        for r in results:
            if r.error:
                sections.append(f"{heading.get(r.role, '## ' + r.role)}\n_(agent error: {r.error})_")
            elif r.content:
                sections.append(f"{heading.get(r.role, '## ' + r.role)}\n{r.content}")

        return {
            "content": "\n\n".join(sections),
            "agents": [
                {
                    "name": r.name,
                    "role": r.role,
                    "model_id": r.model_id,
                    "latency_ms": round(r.latency_ms, 2),
                    "error": r.error,
                }
                for r in results
            ],
        }


# @spec[PROJECT_PROFILE.md#Requirements]
def _extract_content(response: Dict[str, Any]) -> str:
    """Pull assistant content out of an OpenAI-format chat completion."""
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


# Process-wide engine instance.
# @spec[PROJECT_PROFILE.md#Requirements]
autoscale_engine = AutoscaleEngine()
