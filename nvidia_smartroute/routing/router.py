# @spec[PROJECT_PROFILE.md]
"""
Model capability routing and NVIDIA NIM integration for NVIDIA-SmartRoute-CLI.
"""

import asyncio
import json
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import structlog

from ..metrics import metrics

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
    cost_per_token: float = 0.0  # Cost per token (if applicable)
    
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
class CapabilityAnalyzer:
    """Analyzes requests to determine task type."""

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
    def analyze_request(self, messages: List[Dict[str, str]]) -> TaskType:
        """
        Analyze the request to determine the task type.
        
        Args:
            messages: The conversation messages
            
        Returns:
            TaskType: The detected task type
        """
        if not messages:
            return TaskType.CHAT

        # Extract text and detect image parts. OpenAI multimodal messages carry
        # `content` as a list of parts (text / image_url), so plain string
        # joining would raise on those payloads.
        full_text, has_image = self._extract_content(messages)
        full_text = full_text.lower()

        # A request carrying an image is a vision task regardless of the text.
        if has_image:
            return TaskType.VISION
        
        # Check for code-related keywords
        code_indicators = [
            "function", "class", "def ", "return ", "if ", "for ", "while ",
            "import ", "from ", "== ", "!=", "< ", "> ", "+ ", "- ", "* ", "/",
            "python", "javascript", "java", "cpp", "c#", "ruby", "php", "html",
            "css", "sql", "api", "endpoint", "variable", "loop", "array"
        ]
        
        # Check for code generation indicators
        code_gen_indicators = [
            "write a", "create a", "implement", "code to",
            "program", "script", "build a", "make a", "develop"
        ]
        
        # Check for code completion indicators
        code_comp_indicators = [
            "complete", "finish", "continue", "...", "what comes next"
        ]
        
        # Check for code explanation indicators
        code_exp_indicators = [
            "what does this code do", "how does this code work",
            "walk through this code", "break down this code",
            "explain this code", "explain the code", "code explanation"
        ]
        # Check for code review indicators
        code_rev_indicators = [
            "review", "check for errors", "debug", "fix", "improve",
            "optimize", "refactor", "lint"
        ]
        
        # Check for creative writing indicators
        creative_indicators = [
            "story", "poem", "novel", "character", "plot", "setting", "narrative",
            "describe", "imagine", "create", "write", "fiction", "fantasy",
            "scifi", "mystery", "romance", "dialogue", "scene"
        ]
        
        # Check for analytical thinking indicators
        analysis_indicators = [
            "analyze", "compare", "contrast", "evaluate", "assess", "examine",
            "explain why", "how does", "what causes", "impact", "effect",
            "pros and cons", "advantages", "disadvantages", "benefits", "drawbacks"
        ]
        
        # Check for mathematics indicators
        # Note: "sum" is intentionally omitted because it is a substring of
        # "summarize"/"summary" and would misclassify summarization requests.
        math_indicators = [
            "calculate", "compute", "solve", "equation", "formula", "derivative",
            "integral", "limit", "average", "median", "mode",
            "statistics", "probability", "percentage", "fraction", "algebra",
            "geometry", "trigonometry", "calculus"
        ]
        
        # Check for translation indicators
        trans_indicators = [
            "translate", "translation", "in spanish", "in french", "in german",
            "in italian", "in portuguese", "in russian", "in chinese", "in japanese",
            "in korean", "in arabic", "in hindi"
        ]
        
        # Check for summarization indicators
        sum_indicators = [
            "summarize", "summary", "brief", "overview", "tl;dr", "in short",
            "to summarize", "in summary", "sum up"
        ]
        
        # Score each category
        code_score = sum(1 for indicator in code_indicators if indicator in full_text)
        code_gen_score = sum(2 for indicator in code_gen_indicators if indicator in full_text)
        code_comp_score = sum(1 for indicator in code_comp_indicators if indicator in full_text)
        code_exp_score = sum(1 for indicator in code_exp_indicators if indicator in full_text)
        code_rev_score = sum(1 for indicator in code_rev_indicators if indicator in full_text)
        creative_score = sum(1 for indicator in creative_indicators if indicator in full_text)
        analysis_score = sum(1 for indicator in analysis_indicators if indicator in full_text)
        math_score = sum(1 for indicator in math_indicators if indicator in full_text)
        # Arithmetic expressions (e.g. "2+2", "10 * 3") are a strong math signal
        if re.search(r"\d+\s*[+\-*/^]\s*\d+", full_text):
            math_score += 2
        trans_score = sum(1 for indicator in trans_indicators if indicator in full_text)
        sum_score = sum(1 for indicator in sum_indicators if indicator in full_text)
        
        # Determine the highest scoring category
        scores = {
            "code_generation": code_gen_score,
            "code_completion": code_comp_score,
            "code_explanation": code_exp_score,
            "code_review": code_rev_score,
            "creative_writing": creative_score,
            "reasoning": analysis_score,
            "mathematics": math_score,
            "translation": trans_score,
            "summarization": sum_score,
            "chat": 1  # Default baseline
        }
        logger.debug("capability scores", scores=scores)

        # Find the category with the highest score
        max_category = max(scores, key=scores.get)
        max_score = scores[max_category]
        
        # Map to TaskType enum
        if max_category == "code_generation" and max_score > 0:
            return TaskType.CODE_GENERATION
        elif max_category == "code_completion" and max_score > 0:
            return TaskType.CODE_COMPLETION
        elif max_category == "code_explanation" and max_score > 0:
            return TaskType.CODE_EXPLANATION
        elif max_category == "code_review" and max_score > 0:
            return TaskType.CODE_REVIEW
        elif max_category == "creative_writing" and max_score > 0:
            return TaskType.CREATIVE_WRITING
        elif max_category == "reasoning" and max_score > 0:
            return TaskType.REASONING
        elif max_category == "mathematics" and max_score > 0:
            return TaskType.MATHEMATICS
        elif max_category == "translation" and max_score > 0:
            return TaskType.TRANSLATION
        elif max_category == "summarization" and max_score > 0:
            return TaskType.SUMMARIZATION
        else:
            # Default to chat
            return TaskType.CHAT


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

        return (
            0.5 * model.quality_score
            + 0.3 * model.reliability_score
            + 0.2 * (1.0 - latency_penalty)
        )


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
        
        # Analyze the request to determine task type
        task_type = self.capability_analyzer.analyze_request(messages)
        
        # If a specific model is requested, use it if available
        selected_model = None
        if model:
            specific_model = self.model_registry.get_model(model)
            if specific_model:
                selected_model = specific_model
                # Determine task type from the model's capabilities
                if specific_model.supported_tasks:
                    task_type = specific_model.supported_tasks[0]
        
        # If no specific model was requested or found, select the best model for the task
        if not selected_model:
            selected_model = self.model_registry.select_best_model(task_type)
        
        # Calculate confidence (simplified)
        confidence = 0.8 if selected_model else 0.5
        
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