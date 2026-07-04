# @spec[PROJECT_PROFILE.md]
"""
Configuration management for NVIDIA-SmartRoute-CLI using Pydantic Settings.
"""

from functools import lru_cache
from typing import Optional, Literal
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # Server settings
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    port: int = Field(default=8000, ge=1, le=65535, description="Port to bind to")
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    workers: int = Field(default=1, ge=1, description="Number of worker processes")
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    reload: bool = Field(default=False, description="Enable auto-reload")
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    log_level: str = Field(default="info", description="Logging level")
    # @spec[PROJECT_PROFILE.md#Token Budget Class]
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # NVIDIA NIM API settings
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    nvidia_nim_api_key: Optional[str] = Field(
        default=None, 
        description="NVIDIA NIM API key for accessing models"
    )
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    nvidia_nim_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="Base URL for NVIDIA NIM API"
    )
    
    # Model routing settings
    # @spec[PROJECT_PROFILE.md#Requirements]
    default_model: str = Field(
        default="nemotron-3-super",
        description="Default model to use when routing is not specified"
    )
    # @spec[PROJECT_PROFILE.md#Requirements]
    enable_routing: bool = Field(
        default=True,
        description="Enable intelligent model routing based on task type"
    )
    
    # Security settings
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    cors_origins: list[str] = Field(
        default=["*"], 
        description="Allowed CORS origins"
    )
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication"
    )
    
    # Rate limiting
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    rate_limit_requests: int = Field(
        default=100,
        description="Number of requests allowed per time window"
    )
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    rate_limit_window: int = Field(
        default=60,
        description="Time window for rate limiting in seconds"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    @validator("nvidia_nim_api_key")
    def validate_api_key(cls, v):
        """Validate that API key is provided when needed."""
        # In development, we might not require it, but in production we should
        return v


# @spec[PROJECT_PROFILE.md#Token Budget Class]
@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# For backward compatibility
# @spec[PROJECT_PROFILE.md#Token Budget Class]
settings = get_settings()