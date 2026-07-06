# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
"""
Streaming chat completions: SSE relay from NVIDIA NIM with key rotation on
connect-time 429s, plus token/latency accounting for the streamed path.
"""

import json
import time
import uuid
from typing import AsyncGenerator, Optional

import structlog

from ..config import settings
from ..metrics import metrics
from ..keypool import key_pool, _mask as _mask_key
from . import runtime
from .nim_client import nim_client
from .recording import record_stream_usage

logger = structlog.get_logger()


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
def format_streaming_chunk(
    content: str, model: str, index: int = 0, finish_reason: Optional[str] = None
) -> dict:
    """Format a streaming chunk in OpenAI format."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": index, "delta": {}, "finish_reason": finish_reason}],
    }
    if content:
        chunk["choices"][0]["delta"]["content"] = content
    return chunk


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
async def stream_chat_completion(
    model: str,
    messages: list,
    request_id: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """Stream a chat completion from NVIDIA NIM."""
    start = time.time()
    errored = False
    try:
        async for line in stream_nim_request(
            model, messages, True, max_tokens, temperature, **kwargs
        ):
            yield line
    except Exception as e:
        errored = True
        logger.error(f"Error in stream_chat_completion: {e}")
        error_chunk = {
            "error": {
                "message": str(e) or repr(e),
                "type": "internal_error",
                "param": None,
                "code": 500,
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    finally:
        # Record latency for the streaming path (tokens tallied by the relay).
        if errored:
            metrics.record_error(model)
        else:
            metrics.record_latency(model, (time.time() - start) * 1000.0)
        yield "data: [DONE]\n\n"


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
async def stream_nim_request(  # noqa: C901
    model: str,
    messages: list,
    stream: bool,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """Make a streaming request to NVIDIA NIM API."""
    payload = {"model": model, "messages": messages, "stream": stream}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    payload.update(kwargs)

    url = f"{settings.nvidia_nim_base_url}/chat/completions"

    # Rotate keys on 429 when opening the stream (can't retry mid-stream).
    attempts = settings.upstream_max_retries + 1
    for attempt in range(attempts):
        key = await nim_client._acquire_key()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        async with runtime.http_client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code == 429 and attempt < attempts - 1:
                await response.aread()
                retry_after = response.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else settings.per_key_rate_window
                )
                if key:
                    key_pool.record_cooldown(key, delay)
                logger.warning("upstream 429 on stream; rotating key", key=_mask_key(key))
                continue

            response.raise_for_status()
            # Track the latest usage seen; upstream may send it cumulatively
            # across chunks, so record it once at the end (not per chunk).
            final_usage: dict = {}
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]  # Remove "data: " prefix
                if data.strip() == "[DONE]":
                    record_stream_usage(model, final_usage)
                    yield "data: [DONE]\n\n"
                    return
                try:
                    # Parse the JSON data from NIM and forward it as-is.
                    json_data = json.loads(data)
                    usage = json_data.get("usage") if isinstance(json_data, dict) else None
                    if isinstance(usage, dict) and usage.get("total_tokens"):
                        final_usage = usage
                    yield f"data: {json.dumps(json_data)}\n\n"
                except json.JSONDecodeError:
                    # If it's not JSON, wrap it in a basic chunk.
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': data}}]})}\n\n"
            # Stream completed (no explicit [DONE]); record usage if seen.
            record_stream_usage(model, final_usage)
            return
