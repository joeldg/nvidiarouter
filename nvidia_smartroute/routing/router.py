# @spec[PROJECT_PROFILE.md]
"""
Model capability routing and NVIDIA NIM integration for NVIDIA-SmartRoute-CLI.
"""

import re
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import structlog

from ..metrics import metrics
from ..config import settings

# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
logger = structlog.get_logger()


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
class TaskType(Enum):
    """Types of tasks that can be routed to different models."""
    CODE_GENERATION = "code_generation"
    CODE_COMPLETION = "code_completion"
    CODE_EXPLANATION = "code_explanation"
    CODE_REVIEW = "code_review"
    CREATIVE_WRITING = "creative_writing"
    REASONING = "reasoning"
    MATHEMATICS = "mathematics"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    VISION = "vision"
    CHAT = "chat"


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@dataclass
class ModelCapability:
    """Represents the capabilities and characteristics of a model."""
    model_id: str
    name: str
    provider: str
    version: str

    # Task types this model excels at
    supported_tasks: List[TaskType] = field(default_factory=list)

    # Performance characteristics
    latency_ms: int = 0  # Average latency in milliseconds
    throughput_tps: float = 0.0  # Tokens per second
    cost_per_token: float = 0.0  # Deprecated; see input/output_cost_per_1k
    # USD per 1,000 tokens. Free on the NIM free tier ($0), but representative
    # hosted rates so cost tracking/routing produce meaningful numbers when set.
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0

    # Quality scores (0.0 to 1.0)
    quality_score: float = 0.0
    reliability_score: float = 0.0

    # Context window size
    context_window: int = 4096

    # Specialized capabilities
    supports_streaming: bool = False
    supports_function_calling: bool = False
    supports_vision: bool = False

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@dataclass
class RoutingDecision:
    """Represents a decision made by the router."""
    request_id: str
    task_type: TaskType
    selected_model: Optional[ModelCapability]
    confidence: float  # 0.0 to 1.0
    reasoning: str = ""


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@dataclass
class Classification:
    """Result of analysing a request: the task type plus scoring detail."""
    task_type: TaskType
    confidence: float
    scores: Dict[str, float]


# Weighted keyword rules per task type.
#
# Each rule is (weight, kind, patterns):
#   - kind "word":   matched on word boundaries (so "sum" != "summarize")
#   - kind "phrase": matched as a substring (multi-word signals)
# A pattern contributes its weight once if present (not once per occurrence).
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
_RULES: Dict[TaskType, List[Tuple[int, str, List[str]]]] = {
    TaskType.CODE_GENERATION: [
        (3, "phrase", ["write a", "create a", "build a", "make a", "generate a",
                       "code to", "implement a", "write me a"]),
        (2, "word", ["implement", "develop", "program", "script"]),
    ],
    TaskType.CODE_COMPLETION: [
        (3, "phrase", ["complete the", "finish the", "continue the", "fill in",
                       "what comes next", "complete this", "complete the following"]),
        (2, "word", ["autocomplete", "complete", "finish", "continue"]),
    ],
    TaskType.CODE_EXPLANATION: [
        (3, "phrase", ["what does this code", "how does this code",
                       "walk through this code", "break down this code",
                       "explain this code", "explain the code", "code explanation",
                       "explain the following code", "what does this function"]),
    ],
    TaskType.CODE_REVIEW: [
        (2, "word", ["review", "debug", "refactor", "lint", "optimize"]),
        (2, "phrase", ["check for errors", "find bugs", "fix the bug", "code review"]),
        (1, "word", ["fix", "improve"]),
    ],
    TaskType.CREATIVE_WRITING: [
        (2, "word", ["story", "poem", "haiku", "novel", "narrative", "fiction",
                     "fantasy", "lyrics", "screenplay", "dialogue"]),
        (2, "phrase", ["write a story", "write a poem", "short story"]),
        (1, "word", ["character", "plot", "imagine", "scene", "romance", "mystery"]),
    ],
    TaskType.REASONING: [
        (3, "phrase", ["explain why", "why is", "why does", "how does",
                       "what causes", "pros and cons"]),
        (2, "word", ["analyze", "compare", "contrast", "evaluate", "assess",
                     "examine", "reasoning"]),
        (1, "word", ["impact", "advantages", "disadvantages", "benefits", "drawbacks"]),
    ],
    TaskType.MATHEMATICS: [
        (2, "word", ["calculate", "compute", "solve", "equation", "formula",
                     "derivative", "integral", "algebra", "geometry", "calculus",
                     "trigonometry", "probability", "percentage", "fraction",
                     "statistics", "factorial", "median", "variance"]),
    ],
    TaskType.TRANSLATION: [
        (3, "word", ["translate", "translation"]),
        (2, "phrase", ["in spanish", "in french", "in german", "in italian",
                       "in portuguese", "in russian", "in chinese", "in japanese",
                       "in korean", "in arabic", "in hindi", "to spanish",
                       "to french", "to german", "to japanese"]),
    ],
    TaskType.SUMMARIZATION: [
        (3, "word", ["summarize", "summary"]),
        (2, "phrase", ["tl;dr", "in short", "sum up", "to summarize", "in summary",
                       "key points", "give me an overview"]),
        (1, "word", ["overview", "brief"]),
    ],
}

# Code-domain vocabulary — a weak signal that a request is code-related. Adds a
# small boost to every CODE_* task so e.g. "write a function to calculate X"
# routes to code, not maths.
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
_CODE_DOMAIN = [
    "python", "javascript", "typescript", "java", "kotlin", "swift", "rust",
    "golang", "ruby", "php", "html", "css", "sql", "bash", "function", "class",
    "method", "variable", "array", "loop", "api", "endpoint", "code", "def",
    "import", "compile", "syntax", "algorithm", "recursion", "json", "regex",
]
_CODE_TASKS = [
    TaskType.CODE_GENERATION,
    TaskType.CODE_COMPLETION,
    TaskType.CODE_EXPLANATION,
    TaskType.CODE_REVIEW,
]

# Deterministic tie-break order (more specific tasks win ties).
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
_TASK_PRIORITY = [
    TaskType.VISION,
    TaskType.CODE_GENERATION,
    TaskType.CODE_COMPLETION,
    TaskType.CODE_REVIEW,
    TaskType.CODE_EXPLANATION,
    TaskType.MATHEMATICS,
    TaskType.TRANSLATION,
    TaskType.SUMMARIZATION,
    TaskType.CREATIVE_WRITING,
    TaskType.REASONING,
    TaskType.CHAT,
]

_ARITHMETIC_RE = re.compile(r"\d+\s*[+\-*/^]\s*\d+")


def _word_score(patterns: List[str], text: str) -> int:
    """Number of patterns present as whole words."""
    return sum(1 for p in patterns if re.search(r"\b" + re.escape(p) + r"\b", text))


def _phrase_score(patterns: List[str], text: str) -> int:
    """Number of patterns present as substrings."""
    return sum(1 for p in patterns if p in text)


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
class CapabilityAnalyzer:
    """Analyzes requests to determine task type via weighted keyword scoring."""

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    @staticmethod
    def _extract_content(messages: List[Dict[str, Any]]) -> Tuple[str, bool]:
        """
        Flatten message content to text and detect whether any image is present.

        Handles both plain-string content and OpenAI multimodal content, where
        `content` is a list of parts such as
        ``{"type": "text", "text": ...}`` and
        ``{"type": "image_url", "image_url": {...}}``.

        Returns:
            (combined_text, has_image)
        """
        texts: List[str] = []
        has_image = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        texts.append(str(part))
                        continue
                    part_type = part.get("type")
                    if part_type == "text":
                        texts.append(part.get("text", ""))
                    elif part_type in ("image_url", "image", "input_image"):
                        has_image = True
            elif content is not None:
                texts.append(str(content))
        return " ".join(texts), has_image

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def classify(self, messages: List[Dict[str, Any]]) -> Classification:  # noqa: C901
        """
        Classify a request into a task type with a confidence score.

        Uses weighted, word-boundary keyword scoring plus structural signals
        (image presence -> vision, arithmetic expressions -> maths) and a
        code-domain boost so incidental keywords don't misroute code requests.
        """
        if not messages:
            return Classification(TaskType.CHAT, 0.4, {})

        full_text, has_image = self._extract_content(messages)
        text = full_text.lower()

        # A request carrying an image is a vision task regardless of the text.
        if has_image:
            return Classification(TaskType.VISION, 0.99, {"vision": 1.0})

        # Score every task type from its weighted rules.
        scores: Dict[TaskType, float] = {}
        for task, rules in _RULES.items():
            total = 0
            for weight, kind, patterns in rules:
                hits = (_word_score if kind == "word" else _phrase_score)(patterns, text)
                total += weight * hits
            if total:
                scores[task] = total

        # Structural: arithmetic expressions are a strong maths signal.
        if _ARITHMETIC_RE.search(text):
            scores[TaskType.MATHEMATICS] = scores.get(TaskType.MATHEMATICS, 0) + 3

        # Code-domain boost: nudge all code tasks when code vocabulary appears.
        domain = min(_word_score(_CODE_DOMAIN, text), 3)
        if domain:
            for task in _CODE_TASKS:
                if task in scores or domain:
                    scores[task] = scores.get(task, 0) + domain

        if not scores or max(scores.values()) == 0:
            return Classification(TaskType.CHAT, 0.4, {})

        top_score = max(scores.values())
        # Deterministic tie-break by priority order.
        winners = [t for t in _TASK_PRIORITY if scores.get(t, 0) == top_score]
        winner = winners[0] if winners else TaskType.CHAT

        # Confidence: winner's share of total signal, with a margin bonus.
        total_signal = sum(scores.values()) + 1  # +1 baseline for chat
        confidence = min(0.99, max(0.4, top_score / total_signal + 0.1))

        readable = {t.value: s for t, s in scores.items()}
        logger.debug("capability scores", scores=readable, winner=winner.value)
        return Classification(winner, round(confidence, 2), readable)

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def analyze_request(self, messages: List[Dict[str, str]]) -> TaskType:
        """Return just the detected task type (see ``classify`` for detail)."""
        return self.classify(messages).task_type


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
class ModelRegistry:
    """Registry of available models."""

    def __init__(self):
        self.models: Dict[str, ModelCapability] = {}
        self._initialize_default_models()

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def _initialize_default_models(self):
        """Initialize with default NVIDIA NIM models (build.nvidia.com IDs).

        These IDs were verified as servable against the live NIM endpoint. If
        your account has access to additional models (e.g. a dedicated code
        model), register them here to specialize routing further.
        """
        # Nemotron Super 49B - strong reasoning, math, summarization, Q&A
        self.models["nvidia/llama-3.3-nemotron-super-49b-v1"] = ModelCapability(
            model_id="nvidia/llama-3.3-nemotron-super-49b-v1",
            name="Llama-3.3-Nemotron-Super-49B",
            provider="nvidia",
            version="1.0",
            supported_tasks=[
                TaskType.REASONING,
                TaskType.MATHEMATICS,
                TaskType.SUMMARIZATION,
                TaskType.QUESTION_ANSWERING,
                TaskType.TRANSLATION,
                TaskType.CHAT,
            ],
            latency_ms=700,
            throughput_tps=35.0,
            quality_score=0.92,
            reliability_score=0.9,
            context_window=32768,
            supports_streaming=True,
            input_cost_per_1k=0.0009,
            output_cost_per_1k=0.0009,
            description="NVIDIA Nemotron Super 49B for reasoning and math",
            tags=["general-purpose", "reasoning", "math"],
        )

        # Llama 3.1 70B - capable generalist; handles code and creative writing.
        # (No dedicated code model is available to this account, so code tasks
        # route here.)
        self.models["meta/llama-3.1-70b-instruct"] = ModelCapability(
            model_id="meta/llama-3.1-70b-instruct",
            name="Llama-3.1-70B-Instruct",
            provider="nvidia",
            version="1.0",
            supported_tasks=[
                TaskType.CODE_GENERATION,
                TaskType.CODE_COMPLETION,
                TaskType.CODE_EXPLANATION,
                TaskType.CODE_REVIEW,
                TaskType.CREATIVE_WRITING,
                TaskType.CHAT,
            ],
            latency_ms=600,
            throughput_tps=40.0,
            quality_score=0.88,
            reliability_score=0.88,
            context_window=32768,
            supports_streaming=True,
            supports_function_calling=True,
            input_cost_per_1k=0.0009,
            output_cost_per_1k=0.0009,
            description="Meta Llama 3.1 70B generalist for code and content",
            tags=["general-purpose", "code", "creative"],
        )

        # Llama 3.1 8B - fast, low-latency conversational model.
        self.models["meta/llama-3.1-8b-instruct"] = ModelCapability(
            model_id="meta/llama-3.1-8b-instruct",
            name="Llama-3.1-8B-Instruct",
            provider="nvidia",
            version="1.0",
            supported_tasks=[TaskType.CHAT],
            latency_ms=250,
            throughput_tps=90.0,
            quality_score=0.80,
            reliability_score=0.9,
            context_window=32768,
            supports_streaming=True,
            input_cost_per_1k=0.0002,
            output_cost_per_1k=0.0002,
            description="Meta Llama 3.1 8B for fast conversational responses",
            tags=["fast", "chat", "lightweight"],
        )

        # Llama 3.2 90B Vision - multimodal image understanding
        self.models["meta/llama-3.2-90b-vision-instruct"] = ModelCapability(
            model_id="meta/llama-3.2-90b-vision-instruct",
            name="Llama-3.2-90B-Vision-Instruct",
            provider="nvidia",
            version="1.0",
            supported_tasks=[TaskType.VISION],
            latency_ms=1200,
            throughput_tps=20.0,
            quality_score=0.87,
            reliability_score=0.85,
            context_window=32768,
            supports_streaming=True,
            supports_vision=True,
            input_cost_per_1k=0.0011,
            output_cost_per_1k=0.0011,
            description="Meta Llama 3.2 90B Vision for image understanding",
            tags=["vision", "multimodal", "image-analysis"],
        )

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def get_model(self, model_id: str) -> Optional[ModelCapability]:
        """
        Look up a model by its identifier.

        Args:
            model_id: The model identifier to look up

        Returns:
            ModelCapability: The matching model, or None if not registered
        """
        return self.models.get(model_id)

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def select_best_model(self, task_type: TaskType) -> Optional[ModelCapability]:
        """
        Select the best model for a given task type.

        Args:
            task_type: The task type to find a model for

        Returns:
            ModelCapability: The best model for the task, or None if no suitable model
        """
        # Get models that support the task type
        suitable_models = [
            model for model in self.models.values()
            if task_type in model.supported_tasks
        ]

        if not suitable_models:
            # Fallback to general models (CHAT task type)
            suitable_models = [
                model for model in self.models.values()
                if TaskType.CHAT in model.supported_tasks
            ]

        if not suitable_models:
            # Last resort: use any available model
            suitable_models = list(self.models.values())

        if not suitable_models:
            return None

        # Score each candidate and return the best. Scoring blends static
        # quality/reliability with a live latency signal so the router adapts
        # to real observed performance (the "latency tracker").
        return max(suitable_models, key=self._score_model)

    # @spec[PROJECT_PROFILE.md#Requirements]
    def rank_models(self, task_type: TaskType) -> List[ModelCapability]:
        """
        Return all suitable models for a task, best first.

        Used to build a fallback chain: if the top model fails upstream, the
        gateway can retry the next-ranked model that supports the same task.
        """
        suitable = [m for m in self.models.values() if task_type in m.supported_tasks]
        if not suitable:
            suitable = [m for m in self.models.values() if TaskType.CHAT in m.supported_tasks]
        if not suitable:
            suitable = list(self.models.values())
        return sorted(suitable, key=self._score_model, reverse=True)

    # @spec[PROJECT_PROFILE.md#Requirements]
    def _score_model(self, model: ModelCapability) -> float:
        """
        Compute a routing score for a model (higher is better).

        Combines static quality and reliability with a latency penalty derived
        from the live latency tracker when samples exist, otherwise the model's
        declared latency. Latency is normalised against a 2000ms reference.
        """
        live_latency = metrics.get_avg_latency_ms(model.model_id)
        latency_ms = live_latency if live_latency is not None else float(model.latency_ms)
        # Normalise to a 0..1 penalty (clamped); lower latency -> smaller penalty.
        latency_penalty = min(latency_ms / 2000.0, 1.0)

        score = (
            0.5 * model.quality_score
            + 0.3 * model.reliability_score
            + 0.2 * (1.0 - latency_penalty)
        )
        # Optional cost-aware routing: penalise pricier models. Reference of
        # $0.002/1k tokens maps to a full penalty; disabled when cost_weight = 0.
        if settings.cost_weight:
            avg_cost = (model.input_cost_per_1k + model.output_cost_per_1k) / 2.0
            cost_penalty = min(avg_cost / 0.002, 1.0)
            score -= settings.cost_weight * cost_penalty
        return score


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
class RequestRouter:
    """Main router that combines capability analysis and model selection."""

    def __init__(self):
        self.capability_analyzer = CapabilityAnalyzer()
        self.model_registry = ModelRegistry()
        self._decision_history: List[Dict[str, Any]] = []

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    async def route_request(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> RoutingDecision:
        """
        Route a request to the most appropriate model based on the task.

        Args:
            messages: The conversation messages
            model: Optional specific model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            RoutingDecision: The routing decision with selected model and reasoning
        """
        start_time = time.time()
        request_id = str(int(time.time() * 1000))  # Simple request ID

        # Analyze the request to determine task type (with a confidence score).
        classification = self.capability_analyzer.classify(messages)
        task_type = classification.task_type
        confidence = classification.confidence

        # If a specific model is requested, use it if available
        selected_model = None
        if model:
            specific_model = self.model_registry.get_model(model)
            if specific_model:
                selected_model = specific_model
                # An explicit model choice is a certain routing decision.
                confidence = 1.0
                # Determine task type from the model's capabilities
                if specific_model.supported_tasks:
                    task_type = specific_model.supported_tasks[0]

        # If no specific model was requested or found, select the best model for the task
        if not selected_model:
            selected_model = self.model_registry.select_best_model(task_type)

        if not selected_model:
            confidence = min(confidence, 0.5)

        # Generate reasoning
        reasoning = f"Selected {selected_model.name if selected_model else 'no model'} for {task_type.value} task"
        if selected_model and selected_model.supported_tasks:
            reasoning += f" based on capabilities: {', '.join([t.value for t in selected_model.supported_tasks])}"

        # Create the decision
        decision = RoutingDecision(
            request_id=request_id,
            task_type=task_type,
            selected_model=selected_model,
            confidence=confidence,
            reasoning=reasoning
        )

        # Surface the decision on the live routing log for the TUI/metrics.
        metrics.log_routing(
            request_id=request_id,
            task_type=task_type.value,
            model_id=selected_model.model_id if selected_model else None,
            confidence=confidence,
        )

        # Add to history for statistics
        self._decision_history.append({
            "timestamp": time.time(),
            "decision": decision,
            "processing_time": time.time() - start_time
        })

        # Keep only the last 100 decisions
        if len(self._decision_history) > 100:
            self._decision_history = self._decision_history[-100:]

        return decision

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def get_routing_stats(self) -> dict:
        """
        Get routing statistics.

        Returns:
            dict: Statistics about routing decisions
        """
        if not self._decision_history:
            return {
                "total_decisions": 0,
                "task_type_distribution": {},
                "model_usage": {},
                "average_confidence": 0.0,
                "recent_decisions": []
            }

        # Count by task type
        task_type_counts = {}
        model_usage = {}
        total_confidence = 0.0

        for entry in self._decision_history:
            decision = entry["decision"]
            task_type = decision.task_type
            model_id = decision.selected_model.model_id if decision.selected_model else None
            confidence = decision.confidence

            # Count task types
            task_type_str = task_type.value if hasattr(task_type, 'value') else str(task_type)
            task_type_counts[task_type_str] = task_type_counts.get(task_type_str, 0) + 1

            # Count model usage
            if model_id:
                model_usage[model_id] = model_usage.get(model_id, 0) + 1

            # Sum confidence for average
            total_confidence += confidence

        # Calculate average confidence
        avg_confidence = total_confidence / len(self._decision_history) if self._decision_history else 0.0

        # Get recent decisions (last 5)
        recent_decisions = []
        for entry in self._decision_history[-5:]:
            decision = entry["decision"]
            model_info = decision.selected_model
            model_str = "none"
            if model_info and model_info.model_id:
                model_tasks = [t.value for t in model_info.supported_tasks] if model_info.supported_tasks else []
                model_str = f"{model_info.model_id} ({','.join(model_tasks)})" if model_tasks else model_info.model_id

            recent_decisions.append({
                "request_id": decision.request_id,
                "timestamp": entry["timestamp"],
                "task_type": decision.task_type.value if hasattr(decision.task_type, 'value') else str(decision.task_type),
                "model": model_str,
                "confidence": round(decision.confidence, 2)
            })

        return {
            "total_decisions": len(self._decision_history),
            "task_type_distribution": task_type_counts,
            "model_usage": model_usage,
            "average_confidence": round(avg_confidence, 2),
            "recent_decisions": recent_decisions
        }


# Create a singleton instance
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
router = RequestRouter()
