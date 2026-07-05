# @spec[PROJECT_PROFILE.md]
"""
FastAPI server for NVIDIA-SmartRoute-CLI providing OpenAI-compatible endpoints.
"""

import base64
import time
import uuid
import json
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List, AsyncGenerator, Union, Deque
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
import structlog

from ..config import settings
from ..metrics import metrics
from ..routing.router import router, TaskType, RoutingDecision
from ..agents import autoscale_engine
from ..keypool import key_pool, KeyPool, KeyPoolExhaustedError, _mask as _mask_key
from .. import logging_config  # noqa: F401  (configures structlog on import)

# Structured logging (unified with the router/agent layers).
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
logger = structlog.get_logger()

# HTTP client for NVIDIA NIM API (initialised in the lifespan handler).
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
http_client: Optional[httpx.AsyncClient] = None

# Background task set retained for shutdown cleanup.
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
background_tasks = set()


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and tear down the shared HTTP client."""
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.request_timeout, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    logger.info("HTTP client initialized for NVIDIA NIM API")
    try:
        yield
    finally:
        if http_client:
            await http_client.aclose()
        for task in list(background_tasks):
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        logger.info("Application shutting down")


# Create FastAPI app
# @spec[PROJECT_PROFILE.md#Token Budget Class]
app = FastAPI(
    title="NVIDIA-SmartRoute-CLI API",
    description="OpenAI-compatible API gateway for NVIDIA NIM models",
    version="0.1.0",
    docs_url="/docs" if not settings.debug else None,
    redoc_url="/redoc" if not settings.debug else None,
    lifespan=lifespan,
)

# Add CORS middleware
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
# Allowing credentials together with a wildcard origin is rejected by browsers,
# so only enable credentials when origins are explicitly configured.
_cors_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Inbound rate limiter (sliding window per client IP, applied to /v1/*).
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
_rate_windows: Dict[str, Deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Enforce a per-client sliding-window rate limit on /v1/* endpoints."""
    if settings.enable_rate_limit and request.url.path.startswith("/v1/"):
        client = request.client.host if request.client else "unknown"
        now = time.time()
        window = _rate_windows[client]
        cutoff = now - settings.rate_limit_window
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= settings.rate_limit_requests:
            retry_after = int(window[0] + settings.rate_limit_window - now) + 1
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_exceeded",
                        "code": 429,
                    }
                },
            )
        window.append(now)
    return await call_next(request)


# Inbound client API-key authentication (optional, applied to /v1/*).
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    """Require a valid client API key on /v1/* when auth is enabled."""
    if settings.require_api_key and request.url.path.startswith("/v1/"):
        allowed = settings.gateway_api_key_set
        # Accept the key from the configured header or a Bearer token.
        provided = request.headers.get(settings.api_key_header)
        if not provided:
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                provided = auth[len("Bearer "):].strip()
        if not allowed or not provided or provided not in allowed:
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": settings.api_key_header},
                content={
                    "error": {
                        "message": "Missing or invalid API key",
                        "type": "authentication_error",
                        "code": 401,
                    }
                },
            )
    return await call_next(request)


# Middleware for request logging and timing
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and their processing time."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Add request ID to request state for tracing
    request.state.request_id = request_id

    # Track active connections for the live dashboard.
    metrics.connection_opened()
    try:
        # Process request
        response = await call_next(request)
    finally:
        metrics.connection_closed()

    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log request details
    logger.info(
        "request completed",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=round(process_time, 4),
        client_host=request.client.host if request.client else None,
    )
    
    # Add custom headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Global exception handler
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "unhandled exception",
        request_id=request_id,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        exc_info=True,
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": 500,
                "request_id": request_id,
            }
        },
    )


# Health check endpoint
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and orchestration."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "0.1.0",
        "service": "nvidia-smartroute-cli",
    }


# Root endpoint
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "NVIDIA-SmartRoute-CLI",
        "version": "0.1.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "chat_completions": "/v1/chat/completions",
            "embeddings": "/v1/embeddings",
            "models": "/v1/models"
        }
    }


# Readiness check endpoint
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    # Check HTTP client readiness
    http_ready = http_client is not None and not http_client.is_closed
    
    # TODO: Add actual NVIDIA API connectivity check
    nvidia_api_ready = key_pool.has_keys()
    
    status = "ready" if (http_ready and nvidia_api_ready) else "not_ready"
    
    return {
        "status": status,
        "timestamp": time.time(),
        "checks": {
            "http_client": "ready" if http_ready else "not_initialized",
            "nvidia_api": "configured" if nvidia_api_ready else "not_configured",
        },
    }


# Live metrics endpoint (consumed by the TUI dashboard)
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.get("/metrics")
async def get_metrics():
    """Return live gateway metrics and routing statistics."""
    snapshot = metrics.snapshot()
    snapshot["routing_stats"] = router.get_routing_stats()
    snapshot["api_keys"] = key_pool.snapshot()
    return snapshot


# NVIDIA NIM API client
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
class NIMClient:
    """Client for interacting with NVIDIA NIM API."""

    def __init__(self, base_url: str, key_pool: "KeyPool"):
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
            raise KeyPoolExhaustedError(
                "all API keys are rate-limited; retry later"
            )
        raise KeyPoolExhaustedError("all API keys are rate-limited; retry later")

    # @spec[PROJECT_PROFILE.md#Requirements]
    async def _post_with_retries(self, url: str, payload: dict) -> dict:
        """POST with key rotation + backoff on 429/5xx, honoring Retry-After."""
        attempts = settings.upstream_max_retries + 1
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            key = await self._acquire_key()
            response = await http_client.post(
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

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    async def chat_completions(
        self,
        model: str,
        messages: list,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> dict:
        """Send chat completion request to NVIDIA NIM."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        # Add any additional parameters
        payload.update(kwargs)

        url = f"{self.base_url}/chat/completions"
        return await self._post_with_retries(url, payload)
    
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    async def embeddings(
        self,
        model: str,
        input: Union[str, List[str]],
        encoding_format: str = "float",
        **kwargs,
    ) -> dict:
        """Send embedding request to NVIDIA NIM."""
        payload = {
            "model": model,
            "input": input,
            "encoding_format": encoding_format,
        }
        # Forward model-specific params (e.g. input_type, truncate).
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        url = f"{self.base_url}/embeddings"
        return await self._post_with_retries(url, payload)

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    async def models(self) -> dict:
        """Get available models from NVIDIA NIM."""
        url = f"{self.base_url}/models"
        key = await self._acquire_key()
        response = await http_client.get(url, headers=self._headers(key))
        response.raise_for_status()
        return response.json()


# Initialize NIM client
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
nim_client = NIMClient(
    base_url=settings.nvidia_nim_base_url,
    key_pool=key_pool,
)


# Helper functions for response formatting
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
def format_chat_response(
    content: str, 
    model: str, 
    request_id: str,
    finish_reason: str = "stop"
) -> dict:
    """Format a chat completion response in OpenAI format."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # Would be calculated from actual tokenization
            "completion_tokens": 0,  # Would be calculated from actual tokenization
            "total_tokens": 0
        },
        "system_fingerprint": f"fp_{int(time.time())}",
        "_request_id": request_id  # Custom field for tracing
    }


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
def format_streaming_chunk(
    content: str, 
    model: str, 
    index: int = 0,
    finish_reason: Optional[str] = None
) -> dict:
    """Format a streaming chunk in OpenAI format."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": {},
                "finish_reason": finish_reason
            }
        ]
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
    **kwargs
) -> AsyncGenerator[str, None]:
    """Stream a chat completion from NVIDIA NIM."""
    start = time.time()
    errored = False
    try:
        async for line in _stream_nim_request(
            model, messages, True, max_tokens, temperature, **kwargs
        ):
            yield line
    except Exception as e:
        errored = True
        logger.error(f"Error in stream_chat_completion: {e}")
        # Yield an error chunk in OpenAI format
        error_chunk = {
            "error": {
                "message": str(e) or repr(e),
                "type": "internal_error",
                "param": None,
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    finally:
        # Record latency for the streaming path (tokens aren't tallied here).
        if errored:
            metrics.record_error(model)
        else:
            metrics.record_latency(model, (time.time() - start) * 1000.0)
        yield "data: [DONE]\n\n"


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
async def _stream_nim_request(
    model: str, 
    messages: list, 
    stream: bool,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> AsyncGenerator[str, None]:
    """Make a streaming request to NVIDIA NIM API."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream
    }
    
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
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"

        async with http_client.stream("POST", url, json=payload, headers=headers) as response:
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
            final_total_tokens = 0
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]  # Remove "data: " prefix
                if data.strip() == "[DONE]":
                    if final_total_tokens:
                        metrics.record_tokens(model, final_total_tokens)
                    yield "data: [DONE]\n\n"
                    return
                try:
                    # Parse the JSON data from NIM and forward it as-is.
                    json_data = json.loads(data)
                    usage = json_data.get("usage") if isinstance(json_data, dict) else None
                    if isinstance(usage, dict) and usage.get("total_tokens"):
                        final_total_tokens = int(usage["total_tokens"])
                    yield f"data: {json.dumps(json_data)}\n\n"
                except json.JSONDecodeError:
                    # If it's not JSON, wrap it in a basic chunk.
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': data}}]})}\n\n"
            # Stream completed (no explicit [DONE]); record usage if seen.
            if final_total_tokens:
                metrics.record_tokens(model, final_total_tokens)
            return


# Auto-inline remote images (NVIDIA vision NIM requires base64, not URLs)
# @spec[PROJECT_PROFILE.md#Requirements]
async def _fetch_as_data_url(url: str) -> Optional[str]:
    """Fetch a remote image and return it as a base64 data URL, or None."""
    try:
        resp = await http_client.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True
        )
        resp.raise_for_status()
        data = resp.content
        if len(data) > settings.image_fetch_max_bytes:
            logger.warning("remote image too large; leaving as URL", bytes=len(data))
            return None
        mime = (resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
                or "image/jpeg")
        return f"data:{mime};base64," + base64.b64encode(data).decode()
    except Exception as e:
        logger.warning("failed to inline remote image", url=url, error=str(e))
        return None


# @spec[PROJECT_PROFILE.md#Requirements]
async def _inline_remote_images(messages: list) -> list:
    """Replace remote image_url parts with inlined base64 data URLs."""
    if not settings.inline_remote_images:
        return messages
    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "image_url"
                    and isinstance(part.get("image_url"), dict)
                ):
                    url = part["image_url"].get("url", "")
                    if isinstance(url, str) and url.startswith(("http://", "https://")):
                        data_url = await _fetch_as_data_url(url)
                        if data_url:
                            part = {**part, "image_url": {**part["image_url"], "url": data_url}}
                new_content.append(part)
            msg = {**msg, "content": new_content}
        result.append(msg)
    return result


# Record request performance in background
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
async def _record_request_performance(
    request_id: str,
    routing_decision: RoutingDecision,
    start_time: float
):
    """Record request performance metrics in the background."""
    try:
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log performance metrics
        logger.info(
            "request performance recorded",
            request_id=request_id,
            response_time=round(response_time, 4),
            selected_model=routing_decision.selected_model.model_id if routing_decision.selected_model else None,
            task_type=routing_decision.task_type.value if routing_decision.task_type else None,
            confidence=routing_decision.confidence if routing_decision else None,
            routing_reasoning=getattr(routing_decision, "reasoning", None),
        )
    except Exception as e:
        logger.error(f"Error recording request performance: {e}")


# OpenAI-compatible endpoints with actual routing implementation

# @spec[PROJECT_PROFILE.md#Requirements]
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    """OpenAI-compatible chat completions endpoint with intelligent routing."""
    start_time = time.time()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        # Parse request body
        body = await request.json()
        messages = body.get("messages", [])
        model = body.get("model")  # Optional model override
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature")
        
        # Remove parameters we handle/pass separately so they aren't forwarded
        # twice (as explicit args and again via **body_without_model).
        body_without_model = {k: v for k, v in body.items()
                             if k not in ["messages", "model", "stream", "max_tokens", "temperature"]}
        
        logger.info(
            "chat completion request received",
            request_id=request_id,
            message_count=len(messages),
            stream=stream,
            model_hint=model,
        )
        
        # Inline any remote image URLs (NVIDIA vision NIM needs base64 data URLs).
        messages = await _inline_remote_images(messages)

        # Route the request to determine the best model
        routing_decision: RoutingDecision = await router.route_request(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **body_without_model
        )
        
        selected_model = routing_decision.selected_model
        if not selected_model:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": "No suitable model available for the requested task",
                        "type": "service_unavailable",
                        "code": 503,
                        "request_id": request_id,
                    }
                }
            )
        
        # Log the routing decision
        logger.info(
            "routing decision",
            request_id=request_id,
            selected_model=selected_model.model_id,
            task_type=routing_decision.task_type.value,
            confidence=round(routing_decision.confidence, 2),
            reasoning=getattr(routing_decision, "reasoning", "N/A"),
        )
        
        # Track performance in background
        background_tasks.add_task(
            _record_request_performance,
            request_id,
            routing_decision,
            start_time
        )
        
        # If streaming is requested, return a streaming response
        if stream:
            return StreamingResponse(
                stream_chat_completion(
                    model=selected_model.model_id,
                    messages=messages,
                    request_id=request_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **body_without_model
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id,
                    "X-Selected-Model": selected_model.model_id,
                    "X-Task-Type": routing_decision.task_type.value,
                    "X-Routing-Confidence": str(routing_decision.confidence)
                }
            )
        
        # Common tracing headers for the response.
        response_headers = {
            "X-Request-ID": request_id,
            "X-Selected-Model": selected_model.model_id,
            "X-Task-Type": routing_decision.task_type.value,
            "X-Routing-Confidence": str(routing_decision.confidence),
        }

        # Non-streaming response
        nim_start = time.time()
        try:
            # Dynamic Agent Autoscale Engine: complex multi-step code tasks are
            # fanned out to specialized sub-agents that each call NIM.
            if autoscale_engine.should_scale(routing_decision.task_type, messages):
                logger.info(
                    "autoscaling request to sub-agents",
                    request_id=request_id,
                    model=selected_model.model_id,
                )
                orchestrated = await autoscale_engine.orchestrate(
                    messages=messages,
                    model_id=selected_model.model_id,
                    nim_call=nim_client.chat_completions,
                )
                response_data = format_chat_response(
                    content=orchestrated["content"],
                    model=selected_model.model_id,
                    request_id=request_id,
                )
                # Report the aggregated token usage across all sub-agents.
                response_data["usage"] = orchestrated["usage"]
                response_data["_agents"] = orchestrated["agents"]
                response_headers["X-Autoscaled"] = "true"
                response_headers["X-Agent-Count"] = str(len(orchestrated["agents"]))
            else:
                response_data = await nim_client.chat_completions(
                    model=selected_model.model_id,
                    messages=messages,
                    stream=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **body_without_model
                )

            # Record live latency and token usage for the routing tracker/TUI.
            latency_ms = (time.time() - nim_start) * 1000.0
            metrics.record_latency(selected_model.model_id, latency_ms)
            usage = response_data.get("usage") if isinstance(response_data, dict) else None
            if isinstance(usage, dict) and usage.get("total_tokens"):
                metrics.record_tokens(selected_model.model_id, int(usage["total_tokens"]))

            # Return the upstream payload unchanged, with tracing metadata
            # exposed as response headers (not injected into the body).
            return JSONResponse(content=response_data, headers=response_headers)
        except KeyPoolExhaustedError as e:
            metrics.record_error(selected_model.model_id)
            logger.warning(f"All API keys rate-limited: {e}")
            raise HTTPException(
                status_code=503,
                headers={"Retry-After": str(settings.per_key_rate_window)},
                detail={
                    "error": {
                        "message": "All API keys are rate-limited; retry later",
                        "type": "rate_limit_exceeded",
                        "code": 503,
                        "request_id": request_id,
                    }
                },
            )
        except Exception as e:
            metrics.record_error(selected_model.model_id)
            logger.error(f"Error in chat_completions: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": str(e) or repr(e),
                        "type": "internal_error",
                        "code": 500,
                        "request_id": request_id,
                    }
                }
            )
    except HTTPException:
        # Already-formed HTTP errors (e.g. 503 key exhaustion) pass through.
        raise
    except Exception as e:
        # Catch any other exceptions that occur in the outer try block
        logger.error(f"Error in chat_completions: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e) or repr(e),
                    "type": "internal_error",
                    "code": 500,
                    "request_id": request_id,
                }
            }
        )


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """OpenAI-compatible embeddings endpoint."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    body = await request.json()

    input_data = body.get("input")
    if input_data is None or input_data == "" or input_data == []:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "'input' is required and must be a non-empty string or list",
                    "type": "invalid_request_error",
                    "code": 400,
                    "request_id": request_id,
                }
            },
        )

    # Default to the configured embedding model when none is provided.
    model = body.get("model") or settings.default_embedding_model
    encoding_format = body.get("encoding_format", "float")
    # NVIDIA embedqa models require input_type/truncate; default them sensibly.
    input_type = body.get("input_type", "query")
    truncate = body.get("truncate", "END")

    nim_start = time.time()
    try:
        response = await nim_client.embeddings(
            model=model,
            input=input_data,
            encoding_format=encoding_format,
            input_type=input_type,
            truncate=truncate,
        )
        # Record live metrics for the embedding model.
        metrics.record_latency(model, (time.time() - nim_start) * 1000.0)
        usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict) and usage.get("total_tokens"):
            metrics.record_tokens(model, int(usage["total_tokens"]))
        return JSONResponse(
            content=response,
            headers={"X-Request-ID": request_id, "X-Selected-Model": model},
        )
    except KeyPoolExhaustedError as e:
        metrics.record_error(model)
        logger.warning(f"All API keys rate-limited (embeddings): {e}")
        raise HTTPException(
            status_code=503,
            headers={"Retry-After": str(settings.per_key_rate_window)},
            detail={
                "error": {
                    "message": "All API keys are rate-limited; retry later",
                    "type": "rate_limit_exceeded",
                    "code": 503,
                    "request_id": request_id,
                }
            },
        )
    except Exception as e:
        metrics.record_error(model)
        logger.error(f"Error in embeddings: {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": str(e) or repr(e),
                    "type": "upstream_error",
                    "code": 502,
                    "request_id": request_id,
                }
            }
        )


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.get("/v1/models")
async def list_models(source: str = "router"):
    """
    OpenAI-compatible models endpoint.

    Returns the models this gateway actually routes to (the router registry).
    Pass ``?source=upstream`` to proxy NVIDIA's full NIM catalog instead.
    """
    if source == "upstream":
        try:
            return await nim_client.models()
        except Exception as e:
            logger.error(f"Error in list_models: {e}")
            raise HTTPException(
                status_code=502,
                detail={
                    "error": {
                        "message": str(e) or repr(e),
                        "type": "upstream_error",
                        "code": 502,
                    }
                },
            )

    created = int(time.time())
    data = []
    for model in router.model_registry.models.values():
        data.append(
            {
                "id": model.model_id,
                "object": "model",
                "created": created,
                "owned_by": model.provider,
                # Router-specific metadata (non-standard extensions).
                "supported_tasks": [t.value for t in model.supported_tasks],
                "context_window": model.context_window,
                "supports_vision": model.supports_vision,
                "supports_streaming": model.supports_streaming,
            }
        )
    return {"object": "list", "data": data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)