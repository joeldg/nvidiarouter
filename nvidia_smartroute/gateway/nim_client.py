# @spec[GATEWAY_API.md#Requirements]
"""
NVIDIA NIM API client: key rotation, retry/backoff, and the chat/embeddings/
models calls. Uses the shared runtime HTTP client so the app lifespan owns its
lifecycle.
"""

import asyncio
from typing import Dict, List, Optional, Union

import structlog

from ..config import settings
from ..keypool import key_pool, KeyPoolExhaustedError, _mask as _mask_key
from . import runtime

logger = structlog.get_logger()


# @spec[GATEWAY_API.md#Requirements]
class NIMClient:
    """Client for interacting with NVIDIA NIM API."""

    def __init__(self, base_url: str, key_pool):
        self.base_url = base_url.rstrip("/")
        self.key_pool = key_pool

    def _headers(self, api_key: Optional[str]) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _acquire_key(self) -> str:
        """Get a key with budget, waiting briefly if all are momentarily full."""
        # If no keys are configured, proceed unauthenticated (dev/local).
        if not self.key_pool.has_keys():
            return ""
        attempts = settings.upstream_max_retries + 1
        for attempt in range(attempts):
            key, wait = self.key_pool.acquire()
            if key is not None:
                return key
            if attempt < attempts - 1:
                # Cap the wait so a request can't hang indefinitely.
                await asyncio.sleep(min(wait, 5.0))
                continue
            raise KeyPoolExhaustedError("all API keys are rate-limited; retry later")
        raise KeyPoolExhaustedError("all API keys are rate-limited; retry later")

    # @spec[GATEWAY_API.md#Requirements]
    async def _post_with_retries(self, url: str, payload: dict) -> dict:
        """POST with key rotation + backoff on 429/5xx, honoring Retry-After."""
        attempts = settings.upstream_max_retries + 1
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            key = await self._acquire_key()
            response = await runtime.http_client.post(
                url, json=payload, headers=self._headers(key)
            )
            if response.status_code < 400:
                return response.json()

            retry_after = response.headers.get("Retry-After")
            delay = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else settings.upstream_backoff_base * (2 ** attempt)
            )

            # On 429, cool this key down and fail over to another key.
            if response.status_code == 429:
                if key:
                    self.key_pool.record_cooldown(key, delay or settings.per_key_rate_window)
                if attempt < attempts - 1:
                    logger.warning(
                        "upstream 429; rotating key",
                        key=_mask_key(key),
                        attempt=attempt + 1,
                        attempts=attempts,
                    )
                    continue
            # On 5xx, back off and retry (possibly a different key).
            elif response.status_code >= 500 and attempt < attempts - 1:
                logger.warning(
                    "upstream error; retrying",
                    status=response.status_code,
                    delay=round(delay, 1),
                    attempt=attempt + 1,
                    attempts=attempts,
                )
                await asyncio.sleep(delay)
                continue

            # Non-retryable (4xx other than 429), or retries exhausted.
            try:
                response.raise_for_status()
            except Exception as exc:
                last_exc = exc
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("request failed without a response")

    # @spec[GATEWAY_API.md#Requirements]
    async def chat_completions(
        self,
        model: str,
        messages: list,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> dict:
        """Send chat completion request to NVIDIA NIM."""
        payload = {"model": model, "messages": messages, "stream": stream}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(kwargs)
        return await self._post_with_retries(f"{self.base_url}/chat/completions", payload)

    # @spec[GATEWAY_API.md#Requirements]
    async def embeddings(
        self,
        model: str,
        input: Union[str, List[str]],
        encoding_format: str = "float",
        **kwargs,
    ) -> dict:
        """Send embedding request to NVIDIA NIM."""
        payload = {"model": model, "input": input, "encoding_format": encoding_format}
        # Forward model-specific params (e.g. input_type, truncate).
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        return await self._post_with_retries(f"{self.base_url}/embeddings", payload)

    # @spec[GATEWAY_API.md#Requirements]
    async def models(self) -> dict:
        """Get available models from NVIDIA NIM."""
        key = await self._acquire_key()
        response = await runtime.http_client.get(
            f"{self.base_url}/models", headers=self._headers(key)
        )
        response.raise_for_status()
        return response.json()


# Process-wide NIM client.
# @spec[GATEWAY_API.md#Requirements]
nim_client = NIMClient(base_url=settings.nvidia_nim_base_url, key_pool=key_pool)
