# @spec[PARKOUR_REFINEMENT.md#Requirements]
"""Bounded, terminating verify-and-refine loop for PARKOUR final answers.

The loop scores a candidate answer with a server-owned verifier and, when the
verifier asks to revise and budget remains, revises the candidate with bounded
verifier feedback and re-scores it. It is guaranteed to terminate, always
returns the best-scored candidate observed (ties resolve to the earlier one),
treats a malformed verdict as a verifier failure rather than an implicit accept,
and is non-fatal: any verifier/revision failure returns the best answer produced
so far marked unverified.

The verify/revise LLM calls are injected as callables so the loop is pure and
testable without the gateway.
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


class RefinementError(RuntimeError):
    """Raised when a verifier verdict cannot be parsed or is out of range."""


# @spec[PARKOUR_REFINEMENT.md#Requirements]
class Verdict(BaseModel):
    """A bounded, schema-validated verifier judgement of one candidate."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    accept: bool
    feedback: str = ""


# @spec[PARKOUR_REFINEMENT.md#Requirements]
def parse_verdict(value: Any) -> Verdict:
    """Parse a verifier response; raise RefinementError on malformed output.

    A malformed, unparseable, or out-of-range verdict is never treated as an
    implicit accept — it surfaces as a failure the loop handles non-fatally.
    """
    if isinstance(value, Verdict):
        return value
    decoded: Any = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as direct_error:
            match = _FENCED_JSON.search(value)
            if not match:
                raise RefinementError("verifier verdict is not valid JSON") from direct_error
            try:
                decoded = json.loads(match.group(1))
            except json.JSONDecodeError as fenced_error:
                raise RefinementError(
                    "fenced verifier verdict is not valid JSON"
                ) from fenced_error
    try:
        return cast(Verdict, Verdict.model_validate(decoded))
    except ValidationError as exc:
        raise RefinementError(f"verifier verdict schema is invalid: {exc}") from exc


@dataclass(frozen=True)
class VerifyOutcome:
    verdict: Verdict
    tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class ReviseOutcome:
    text: str
    model_id: str
    tokens: int = 0
    cost_usd: float = 0.0


# @spec[PARKOUR_REFINEMENT.md#Requirements]
@dataclass(frozen=True)
class RefinementLimits:
    max_iterations: int
    max_verifier_calls: int
    max_revision_calls: int
    timeout_seconds: float
    max_added_tokens: int
    max_added_cost_usd: float
    accept_threshold: float
    min_improvement: float
    feedback_chars: int

    @classmethod
    def from_settings(cls, settings: Any) -> "RefinementLimits":
        return cls(
            max_iterations=settings.parkour_refine_max_iterations,
            max_verifier_calls=settings.parkour_refine_max_verifier_calls,
            max_revision_calls=settings.parkour_refine_max_revision_calls,
            timeout_seconds=settings.parkour_refine_timeout_seconds,
            max_added_tokens=settings.parkour_refine_max_added_tokens,
            max_added_cost_usd=settings.parkour_refine_max_added_cost_usd,
            accept_threshold=settings.parkour_refine_accept_threshold,
            min_improvement=settings.parkour_refine_min_improvement,
            feedback_chars=settings.parkour_refine_feedback_chars,
        )


# @spec[PARKOUR_REFINEMENT.md#Requirements]
@dataclass(frozen=True)
class RefinementResult:
    output: str
    best_score: Optional[float]
    iterations: int
    verifier_calls: int
    revision_calls: int
    accepted: bool
    verified: bool
    stop_reason: str
    added_tokens: int
    added_cost_usd: float

    def summary(self) -> Dict[str, Any]:
        """Compact, bounded trace summary safe for response metadata."""
        return {
            "iterations": self.iterations,
            "verifier_calls": self.verifier_calls,
            "revision_calls": self.revision_calls,
            "accepted": self.accepted,
            "verified": self.verified,
            "stop_reason": self.stop_reason,
            "best_score": self.best_score,
            "added_tokens": self.added_tokens,
            "added_cost_usd": round(self.added_cost_usd, 8),
        }


VerifyCallable = Callable[[str], Awaitable[VerifyOutcome]]
ReviseCallable = Callable[[str, str], Awaitable[ReviseOutcome]]
ProgressCallback = Callable[[Dict[str, Any]], Any]


# @spec[PARKOUR_REFINEMENT.md#Requirements]
def _stop_after_score(
    state: "_LoopState", verdict: "Verdict", score: float, limits: "RefinementLimits"
) -> Optional[str]:
    """Return the terminating reason after a candidate is scored, or None.

    Checked in priority order: no-improvement, resource limits, acceptance, then
    the iteration/revision-call ceilings that gate another revision.
    """
    if state.prev_score is not None and score - state.prev_score < limits.min_improvement:
        return "no_improvement"
    if state.added_tokens > limits.max_added_tokens or state.added_cost > limits.max_added_cost_usd:
        return "resource_limit"
    if verdict.accept and score >= limits.accept_threshold:
        return "accepted"
    if state.iterations >= limits.max_iterations:
        return "max_iterations"
    if state.revision_calls >= limits.max_revision_calls:
        return "resource_limit"
    return None


async def _emit(progress: Optional[ProgressCallback], event: Dict[str, Any]) -> None:
    if progress is None:
        return
    outcome = progress(event)
    if hasattr(outcome, "__await__"):
        await outcome


# @spec[PARKOUR_REFINEMENT.md#Requirements]
class _LoopState:
    """Mutable accounting for one refinement run."""

    def __init__(self, initial: str) -> None:
        self.best_text = initial
        self.best_score: Optional[float] = None
        self.current = initial
        self.prev_score: Optional[float] = None
        self.iterations = 0
        self.verifier_calls = 0
        self.revision_calls = 0
        self.added_tokens = 0
        self.added_cost = 0.0
        self.verified = False
        self.accepted = False

    def consider(self, text: str, score: float) -> None:
        """Track the best candidate; ties keep the earlier (strictly-greater)."""
        if self.best_score is None or score > self.best_score:
            self.best_text = text
            self.best_score = score


def _pre_verify_stop(
    state: "_LoopState", limits: RefinementLimits, elapsed: float
) -> Optional[str]:
    """Resource guards checked before each verification; None to continue."""
    if elapsed > limits.timeout_seconds:
        return "resource_limit"
    if state.verifier_calls >= limits.max_verifier_calls:
        return "resource_limit"
    return None


async def _verify_step(
    state: "_LoopState",
    verify: "VerifyCallable",
    limits: RefinementLimits,
    telemetry: Optional["RefinementTelemetry"],
    progress: Optional[ProgressCallback],
) -> Optional[Verdict]:
    """Verify the current candidate; account, track best, and emit. None on failure."""
    await _emit(progress, {
        "type": "verification_started",
        "iteration": state.iterations,
    })
    try:
        outcome = await verify(state.current)
        verdict = outcome.verdict
    except Exception as exc:
        if getattr(exc, "is_parkour_limit", False):
            raise
        if telemetry:
            telemetry.record_verifier_failure()
        await _emit(progress, {"type": "verification_failed"})
        return None
    state.verifier_calls += 1
    state.added_tokens += outcome.tokens
    state.added_cost += outcome.cost_usd
    state.verified = True
    state.consider(state.current, verdict.score)
    await _emit(progress, {
        "type": "verification_completed",
        "iteration": state.iterations,
        "score": round(verdict.score, 4),
        "accepted": bool(verdict.accept and verdict.score >= limits.accept_threshold),
    })
    return verdict


async def _revise_step(
    state: "_LoopState",
    verdict: Verdict,
    revise: "ReviseCallable",
    limits: RefinementLimits,
    telemetry: Optional["RefinementTelemetry"],
    progress: Optional[ProgressCallback],
) -> bool:
    """Revise the current candidate with bounded feedback. False on failure."""
    state.iterations += 1
    feedback = verdict.feedback[: limits.feedback_chars]
    await _emit(progress, {"type": "revision_started", "iteration": state.iterations})
    try:
        revised = await revise(state.current, feedback)
    except Exception as exc:
        if getattr(exc, "is_parkour_limit", False):
            raise
        if telemetry:
            telemetry.record_verifier_failure()
        await _emit(progress, {"type": "revision_failed", "iteration": state.iterations})
        return False
    state.revision_calls += 1
    state.added_tokens += revised.tokens
    state.added_cost += revised.cost_usd
    await _emit(progress, {
        "type": "revision_completed",
        "iteration": state.iterations,
        "model": revised.model_id,
    })
    state.current = revised.text
    return True


async def _verify_or_limit(
    state: "_LoopState",
    verify: "VerifyCallable",
    limits: RefinementLimits,
    telemetry: Optional["RefinementTelemetry"],
    progress: Optional[ProgressCallback],
) -> Tuple[Optional[Verdict], bool]:
    """Verify once, returning whether the shared run budget stopped the loop."""
    try:
        return await _verify_step(
            state, verify, limits, telemetry, progress
        ), False
    except Exception as exc:
        if not getattr(exc, "is_parkour_limit", False):
            raise
        return None, True


async def _revise_or_limit(
    state: "_LoopState",
    verdict: Verdict,
    revise: "ReviseCallable",
    limits: RefinementLimits,
    telemetry: Optional["RefinementTelemetry"],
    progress: Optional[ProgressCallback],
) -> Tuple[bool, bool]:
    """Revise once, returning whether the shared run budget stopped the loop."""
    try:
        return await _revise_step(
            state, verdict, revise, limits, telemetry, progress
        ), False
    except Exception as exc:
        if not getattr(exc, "is_parkour_limit", False):
            raise
        return False, True


# @spec[PARKOUR_REFINEMENT.md#Requirements]
async def run_refinement(
    initial: str,
    verify: VerifyCallable,
    revise: ReviseCallable,
    limits: RefinementLimits,
    telemetry: Optional["RefinementTelemetry"] = None,
    progress: Optional[ProgressCallback] = None,
    clock: Callable[[], float] = time.monotonic,
) -> RefinementResult:
    """Run the bounded verify-and-refine loop over an initial candidate."""
    state = _LoopState(initial)
    started = clock()
    # Default reason if the loop body never sets one (e.g. a zero-budget config);
    # every break path below overwrites it.
    stop_reason = "max_iterations"

    while True:
        guard = _pre_verify_stop(state, limits, clock() - started)
        if guard is not None:
            stop_reason = guard
            break

        verdict, limit_reached = await _verify_or_limit(
            state, verify, limits, telemetry, progress
        )
        if limit_reached:
            stop_reason = "resource_limit"
            break
        if verdict is None:
            stop_reason = "verifier_failed"
            break

        stop = _stop_after_score(state, verdict, verdict.score, limits)
        if stop is not None:
            if stop == "accepted":
                state.accepted = True
            stop_reason = stop
            break

        revised, limit_reached = await _revise_or_limit(
            state, verdict, revise, limits, telemetry, progress
        )
        if limit_reached:
            stop_reason = "resource_limit"
            break
        if not revised:
            stop_reason = "revision_failed"
            break
        state.prev_score = verdict.score

    await _emit(progress, {"type": "refinement_stopped", "reason": stop_reason})
    result = RefinementResult(
        output=state.best_text,
        best_score=state.best_score,
        iterations=state.iterations,
        verifier_calls=state.verifier_calls,
        revision_calls=state.revision_calls,
        accepted=state.accepted,
        verified=state.verified,
        stop_reason=stop_reason,
        added_tokens=state.added_tokens,
        added_cost_usd=state.added_cost,
    )
    if telemetry:
        telemetry.record_loop(result)
    return result


# @spec[PARKOUR_REFINEMENT.md#Requirements]
class RefinementTelemetry:
    """Bounded in-process telemetry for the refinement loop."""

    def __init__(self) -> None:
        self.loops = 0
        self.iterations = 0
        self.accepts = 0
        self.rejects = 0
        self.no_improvement_stops = 0
        self.limit_stops = 0
        self.verifier_failures = 0
        self.returned_score_sum = 0.0
        self.returned_score_count = 0
        self.added_tokens = 0
        self.added_cost_usd = 0.0

    def record_verifier_failure(self) -> None:
        self.verifier_failures += 1

    def record_loop(self, result: RefinementResult) -> None:
        self.loops += 1
        self.iterations += result.iterations
        self.added_tokens += result.added_tokens
        self.added_cost_usd += result.added_cost_usd
        if result.accepted:
            self.accepts += 1
        elif result.verified:
            self.rejects += 1
        if result.stop_reason == "no_improvement":
            self.no_improvement_stops += 1
        if result.stop_reason == "resource_limit":
            self.limit_stops += 1
        if result.best_score is not None:
            self.returned_score_sum += result.best_score
            self.returned_score_count += 1

    def snapshot(self) -> Dict[str, Any]:
        avg = (
            self.returned_score_sum / self.returned_score_count
            if self.returned_score_count else 0.0
        )
        return {
            "loops": self.loops,
            "iterations": self.iterations,
            "accepts": self.accepts,
            "rejects": self.rejects,
            "no_improvement_stops": self.no_improvement_stops,
            "limit_stops": self.limit_stops,
            "verifier_failures": self.verifier_failures,
            "avg_returned_score": round(avg, 4),
            "added_tokens": self.added_tokens,
            "added_cost_usd": round(self.added_cost_usd, 8),
        }


refinement_telemetry = RefinementTelemetry()
