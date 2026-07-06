# @spec[PROJECT_PROFILE.md]
"""
Configuration management for NVIDIA-SmartRoute-CLI using Pydantic Settings.
"""

from functools import lru_cache
from typing import Optional, List
from pydantic import Field, AliasChoices, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    # Server settings
    # @spec[GATEWAY_API.md#Requirements]
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    # @spec[GATEWAY_API.md#Requirements]
    # The gateway is specified to listen on port 9000 (0.0.0.0:9000).
    port: int = Field(default=9000, ge=1, le=65535, description="Port to bind to")
    # @spec[GATEWAY_API.md#Requirements]
    workers: int = Field(default=1, ge=1, description="Number of worker processes")
    # @spec[GATEWAY_API.md#Requirements]
    reload: bool = Field(default=False, description="Enable auto-reload")
    # @spec[OBSERVABILITY.md#Requirements]
    log_level: str = Field(default="info", description="Logging level")
    # @spec[OBSERVABILITY.md#Requirements]
    log_json: bool = Field(
        default=False, description="Emit JSON logs instead of console-formatted"
    )
    # @spec[GATEWAY_API.md#Requirements]
    pid_file: str = Field(
        default=".nvidia-smartroute.pid",
        description="Path to the gateway PID file (used by start/stop)",
    )
    # @spec[GATEWAY_API.md#Requirements]
    debug: bool = Field(default=False, description="Enable debug mode")

    # NVIDIA NIM API settings
    # @spec[SECURITY_AND_KEYS.md#Requirements]
    # Accept both NVIDIA_NIM_API_KEY and the shorter NVIDIA_API_KEY used in .env.
    nvidia_nim_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("nvidia_nim_api_key", "nvidia_api_key"),
        description="NVIDIA NIM API key for accessing models",
    )
    # @spec[SECURITY_AND_KEYS.md#Requirements]
    # Optional pool of additional keys (comma-separated). NIM free models cap
    # at ~40 req/min per key; rotating across keys raises aggregate throughput.
    nvidia_api_keys: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("nvidia_api_keys", "nvidia_nim_api_keys"),
        description="Comma-separated pool of NVIDIA API keys for rotation",
    )
    # @spec[GATEWAY_API.md#Requirements]
    # Default to the OpenAI-compatible NIM endpoint. Accept NVIDIA_BASE_URL too.
    nvidia_nim_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        validation_alias=AliasChoices("nvidia_nim_base_url", "nvidia_base_url"),
        description="Base URL for NVIDIA NIM API",
    )
    # @spec[GATEWAY_API.md#Requirements]
    # NIM models can be slow on cold start; allow a generous read timeout so
    # upstream calls aren't cut off prematurely.
    request_timeout: float = Field(
        default=120.0, gt=0, description="Upstream NIM read timeout in seconds"
    )

    # Model routing settings
    # @spec[ROUTING.md#Requirements]
    default_model: str = Field(
        default="meta/llama-3.1-70b-instruct",
        description="Default model to use when routing is not specified",
    )
    # @spec[ROUTING.md#Requirements]
    enable_routing: bool = Field(
        default=True,
        description="Enable intelligent model routing based on task type",
    )
    # @spec[ROUTING.md#Requirements]
    # "static"  -> quality/latency/cost scoring (default)
    # "adaptive" -> epsilon-greedy bandit that learns the best model per task
    routing_strategy: str = Field(
        default="static", description="Model selection strategy: static | adaptive"
    )
    # @spec[ROUTING.md#Requirements]
    bandit_epsilon: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Exploration rate for adaptive routing (0..1)",
    )
    # @spec[MODEL_DISCOVERY.md#Requirements]
    # Discovered models file (written by `nvidia-smartroute discover`). When
    # present, the router loads these on top of the built-in defaults.
    models_file: str = Field(
        default="discovered_models.json",
        description="Path to discovered model capabilities (optional)",
    )
    # @spec[MODEL_DISCOVERY.md#Requirements]
    default_embedding_model: str = Field(
        default="nvidia/nv-embedqa-e5-v5",
        description="Model used for /v1/embeddings when none is specified",
    )

    # Dynamic Agent Autoscale settings
    # @spec[GATEWAY_API.md#Requirements]
    enable_autoscale: bool = Field(
        default=True,
        description="Enable spawning sub-agents for complex multi-step tasks",
    )
    # @spec[GATEWAY_API.md#Requirements]
    max_concurrent_agents: int = Field(
        default=10, ge=1, description="Maximum number of concurrent sub-agents"
    )
    # @spec[GATEWAY_API.md#Requirements]
    # Run follow-up sub-agents (tester/reviewer) one at a time. Default True so
    # they don't compete on the same slow free-tier model and time out; set
    # False to parallelize when models are fast / keys are plentiful.
    autoscale_sequential: bool = Field(
        default=True,
        description="Run follow-up sub-agents sequentially instead of concurrently",
    )
    # @spec[GATEWAY_API.md#Requirements]
    agent_timeout: int = Field(
        default=300, ge=1, description="Timeout (seconds) for a sub-agent task"
    )

    # Response cache
    # @spec[GATEWAY_API.md#Requirements]
    enable_cache: bool = Field(
        default=True, description="Cache identical non-streaming chat responses"
    )
    # @spec[GATEWAY_API.md#Requirements]
    cache_ttl: int = Field(
        default=300, ge=1,
        validation_alias=AliasChoices("cache_ttl", "model_cache_ttl"),
        description="Response cache TTL in seconds",
    )
    # @spec[GATEWAY_API.md#Requirements]
    cache_max_entries: int = Field(
        default=1000, ge=1, description="Maximum number of cached responses"
    )

    # Cost & budget
    # @spec[COST.md#Requirements]
    daily_budget_usd: float = Field(
        default=0.0, ge=0, description="Daily spend cap in USD (0 = unlimited)"
    )
    # @spec[COST.md#Requirements]
    cost_weight: float = Field(
        default=0.0, ge=0,
        description="Weight of model cost in routing (0 = ignore cost)",
    )

    # Reliability
    # @spec[ROUTING.md#Requirements]
    enable_model_fallback: bool = Field(
        default=True,
        description="On upstream model failure, retry the next-best model",
    )
    # @spec[ROUTING.md#Requirements]
    max_model_fallbacks: int = Field(
        default=2, ge=0, description="Max alternative models to try on failure"
    )
    # @spec[ROUTING.md#Requirements]
    circuit_breaker_enabled: bool = Field(
        default=True, description="Take repeatedly-failing models out of rotation"
    )
    # @spec[ROUTING.md#Requirements]
    circuit_failure_threshold: int = Field(
        default=3, ge=1, description="Consecutive failures before a model trips open"
    )
    # @spec[ROUTING.md#Requirements]
    circuit_reset_seconds: int = Field(
        default=30, ge=1, description="Cooldown before probing a tripped model"
    )

    # Metrics persistence (survive restarts)
    # @spec[OBSERVABILITY.md#Requirements]
    persist_metrics: bool = Field(
        default=False, description="Persist metrics counters to disk across restarts"
    )
    # @spec[OBSERVABILITY.md#Requirements]
    metrics_file: str = Field(
        default=".nvidia-smartroute-metrics.json",
        description="Path to the persisted metrics file",
    )
    # @spec[OBSERVABILITY.md#Requirements]
    metrics_save_interval: int = Field(
        default=60, ge=5, description="Seconds between periodic metrics saves"
    )

    # Concurrency / backpressure (smooths bursts against upstream rate limits)
    # @spec[GATEWAY_API.md#Requirements]
    enable_concurrency_limit: bool = Field(
        default=True, description="Bound concurrent upstream requests with a queue"
    )
    # @spec[GATEWAY_API.md#Requirements]
    max_inflight_requests: int = Field(
        default=32, ge=1, description="Max simultaneous upstream chat requests"
    )
    # @spec[GATEWAY_API.md#Requirements]
    max_queued_requests: int = Field(
        default=64, ge=0, description="Max requests waiting for a slot before 503"
    )
    # @spec[GATEWAY_API.md#Requirements]
    queue_timeout: float = Field(
        default=30.0, gt=0, description="Max seconds to wait for a slot before 503"
    )

    # TUI settings
    # @spec[OBSERVABILITY.md#Requirements]
    tui_refresh_rate: float = Field(
        default=1.0, gt=0, description="TUI dashboard refresh interval in seconds"
    )

    # Security settings
    # @spec[SECURITY_AND_KEYS.md#Requirements]
    # Stored as a raw string (e.g. "*", or "https://a.com,https://b.com") and
    # exposed as a parsed list via the `cors_origins` property.
    allowed_origins: str = Field(
        default="*",
        validation_alias=AliasChoices("allowed_origins", "cors_origins"),
        description="Comma-separated list of allowed CORS origins",
    )
    # @spec[SECURITY_AND_KEYS.md#Requirements]
    api_key_header: str = Field(
        default="X-API-Key", description="Header name for API key authentication"
    )
    # @spec[SECURITY_AND_KEYS.md#Requirements]
    require_api_key: bool = Field(
        default=False, description="Require a valid client API key on /v1/* endpoints"
    )
    # @spec[SECURITY_AND_KEYS.md#Requirements]
    gateway_api_keys: Optional[str] = Field(
        default=None, description="Comma-separated client API keys accepted by the gateway"
    )

    # Rate limiting (inbound, applied to /v1/* endpoints)
    # @spec[SECURITY_AND_KEYS.md#Requirements]
    enable_rate_limit: bool = Field(
        default=True, description="Enforce inbound rate limiting on /v1/* endpoints"
    )
    # @spec[SECURITY_AND_KEYS.md#Requirements]
    rate_limit_requests: int = Field(
        default=100, description="Number of requests allowed per time window"
    )
    # @spec[SECURITY_AND_KEYS.md#Requirements]
    rate_limit_window: int = Field(
        default=60, description="Time window for rate limiting in seconds"
    )

    # Upstream resilience
    # @spec[GATEWAY_API.md#Requirements]
    upstream_max_retries: int = Field(
        default=3, ge=0, description="Retries on upstream 429/5xx with backoff"
    )
    # @spec[GATEWAY_API.md#Requirements]
    upstream_backoff_base: float = Field(
        default=0.5, gt=0, description="Base seconds for exponential backoff"
    )
    # @spec[SECURITY_AND_KEYS.md#Requirements]
    # Per-key outbound budget (NIM free tier is ~40 requests/minute per key).
    rate_limit_per_key: int = Field(
        default=40, ge=1, description="Max upstream requests per key per window"
    )
    # @spec[SECURITY_AND_KEYS.md#Requirements]
    per_key_rate_window: int = Field(
        default=60, ge=1, description="Rolling window (seconds) for per-key budget"
    )
    # @spec[GATEWAY_API.md#Requirements]
    inline_remote_images: bool = Field(
        default=True,
        description="Fetch remote image URLs and inline them as base64 for vision",
    )
    # @spec[GATEWAY_API.md#Requirements]
    image_fetch_max_bytes: int = Field(
        default=5_000_000, gt=0, description="Max size of a fetched remote image"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("allowed_origins")
    @classmethod
    def _strip_origins(cls, v: str) -> str:
        """Normalise the raw origins string."""
        return v.strip() if isinstance(v, str) else v

    @property
    def cors_origins(self) -> List[str]:
        """Parse the raw allowed-origins string into a list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def gateway_api_key_set(self) -> set:
        """Set of client API keys the gateway accepts (inbound auth)."""
        if not self.gateway_api_keys:
            return set()
        return {k.strip() for k in self.gateway_api_keys.split(",") if k.strip()}

    @property
    def api_keys(self) -> List[str]:
        """
        The ordered, de-duplicated pool of NVIDIA API keys.

        Merges the single ``NVIDIA_API_KEY`` with the comma-separated
        ``NVIDIA_API_KEYS`` pool. Single-key setups keep working unchanged.
        """
        keys: List[str] = []
        if self.nvidia_api_keys:
            keys.extend(k.strip() for k in self.nvidia_api_keys.split(","))
        if self.nvidia_nim_api_key:
            keys.append(self.nvidia_nim_api_key.strip())
        seen = set()
        ordered: List[str] = []
        for k in keys:
            if k and k not in seen:
                seen.add(k)
                ordered.append(k)
        return ordered


# @spec[GATEWAY_API.md#Requirements]
@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# For backward compatibility
# @spec[GATEWAY_API.md#Requirements]
settings = get_settings()
