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
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    # The gateway is specified to listen on port 9000 (0.0.0.0:9000).
    port: int = Field(default=9000, ge=1, le=65535, description="Port to bind to")
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    workers: int = Field(default=1, ge=1, description="Number of worker processes")
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    reload: bool = Field(default=False, description="Enable auto-reload")
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    log_level: str = Field(default="info", description="Logging level")
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    log_json: bool = Field(
        default=False, description="Emit JSON logs instead of console-formatted"
    )
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    pid_file: str = Field(
        default=".nvidia-smartroute.pid",
        description="Path to the gateway PID file (used by start/stop)",
    )
    # @spec[PROJECT_PROFILE.md#Token Budget Class]
    debug: bool = Field(default=False, description="Enable debug mode")

    # NVIDIA NIM API settings
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    # Accept both NVIDIA_NIM_API_KEY and the shorter NVIDIA_API_KEY used in .env.
    nvidia_nim_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("nvidia_nim_api_key", "nvidia_api_key"),
        description="NVIDIA NIM API key for accessing models",
    )
    # @spec[PROJECT_PROFILE.md#Requirements]
    # Optional pool of additional keys (comma-separated). NIM free models cap
    # at ~40 req/min per key; rotating across keys raises aggregate throughput.
    nvidia_api_keys: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("nvidia_api_keys", "nvidia_nim_api_keys"),
        description="Comma-separated pool of NVIDIA API keys for rotation",
    )
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    # Default to the OpenAI-compatible NIM endpoint. Accept NVIDIA_BASE_URL too.
    nvidia_nim_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        validation_alias=AliasChoices("nvidia_nim_base_url", "nvidia_base_url"),
        description="Base URL for NVIDIA NIM API",
    )
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    # NIM models can be slow on cold start; allow a generous read timeout so
    # upstream calls aren't cut off prematurely.
    request_timeout: float = Field(
        default=120.0, gt=0, description="Upstream NIM read timeout in seconds"
    )

    # Model routing settings
    # @spec[PROJECT_PROFILE.md#Requirements]
    default_model: str = Field(
        default="meta/llama-3.1-70b-instruct",
        description="Default model to use when routing is not specified",
    )
    # @spec[PROJECT_PROFILE.md#Requirements]
    enable_routing: bool = Field(
        default=True,
        description="Enable intelligent model routing based on task type",
    )
    # @spec[PROJECT_PROFILE.md#Requirements]
    default_embedding_model: str = Field(
        default="nvidia/nv-embedqa-e5-v5",
        description="Model used for /v1/embeddings when none is specified",
    )

    # Dynamic Agent Autoscale settings
    # @spec[PROJECT_PROFILE.md#Requirements]
    enable_autoscale: bool = Field(
        default=True,
        description="Enable spawning sub-agents for complex multi-step tasks",
    )
    # @spec[PROJECT_PROFILE.md#Requirements]
    max_concurrent_agents: int = Field(
        default=10, ge=1, description="Maximum number of concurrent sub-agents"
    )
    # @spec[PROJECT_PROFILE.md#Requirements]
    # Run follow-up sub-agents (tester/reviewer) one at a time. Default True so
    # they don't compete on the same slow free-tier model and time out; set
    # False to parallelize when models are fast / keys are plentiful.
    autoscale_sequential: bool = Field(
        default=True,
        description="Run follow-up sub-agents sequentially instead of concurrently",
    )
    # @spec[PROJECT_PROFILE.md#Requirements]
    agent_timeout: int = Field(
        default=300, ge=1, description="Timeout (seconds) for a sub-agent task"
    )

    # TUI settings
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    tui_refresh_rate: float = Field(
        default=1.0, gt=0, description="TUI dashboard refresh interval in seconds"
    )

    # Security settings
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    # Stored as a raw string (e.g. "*", or "https://a.com,https://b.com") and
    # exposed as a parsed list via the `cors_origins` property.
    allowed_origins: str = Field(
        default="*",
        validation_alias=AliasChoices("allowed_origins", "cors_origins"),
        description="Comma-separated list of allowed CORS origins",
    )
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    api_key_header: str = Field(
        default="X-API-Key", description="Header name for API key authentication"
    )
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    require_api_key: bool = Field(
        default=False, description="Require a valid client API key on /v1/* endpoints"
    )
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    gateway_api_keys: Optional[str] = Field(
        default=None, description="Comma-separated client API keys accepted by the gateway"
    )

    # Rate limiting (inbound, applied to /v1/* endpoints)
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    enable_rate_limit: bool = Field(
        default=True, description="Enforce inbound rate limiting on /v1/* endpoints"
    )
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    rate_limit_requests: int = Field(
        default=100, description="Number of requests allowed per time window"
    )
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    rate_limit_window: int = Field(
        default=60, description="Time window for rate limiting in seconds"
    )

    # Upstream resilience
    # @spec[PROJECT_PROFILE.md#Requirements]
    upstream_max_retries: int = Field(
        default=3, ge=0, description="Retries on upstream 429/5xx with backoff"
    )
    # @spec[PROJECT_PROFILE.md#Requirements]
    upstream_backoff_base: float = Field(
        default=0.5, gt=0, description="Base seconds for exponential backoff"
    )
    # @spec[PROJECT_PROFILE.md#Requirements]
    # Per-key outbound budget (NIM free tier is ~40 requests/minute per key).
    rate_limit_per_key: int = Field(
        default=40, ge=1, description="Max upstream requests per key per window"
    )
    # @spec[PROJECT_PROFILE.md#Requirements]
    per_key_rate_window: int = Field(
        default=60, ge=1, description="Rolling window (seconds) for per-key budget"
    )
    # @spec[PROJECT_PROFILE.md#Requirements]
    inline_remote_images: bool = Field(
        default=True,
        description="Fetch remote image URLs and inline them as base64 for vision",
    )
    # @spec[PROJECT_PROFILE.md#Requirements]
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


# @spec[PROJECT_PROFILE.md#Token Budget Class]
@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# For backward compatibility
# @spec[PROJECT_PROFILE.md#Token Budget Class]
settings = get_settings()
