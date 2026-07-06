# @spec[PROJECT_PROFILE.md]
"""
FastAPI server for NVIDIA-SmartRoute-CLI providing OpenAI-compatible endpoints.
"""

import time
import uuid
import json
from pathlib import Path
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from typing import Optional, Dict, Deque

import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
import structlog

from ..config import settings
from ..metrics import metrics
from ..routing.router import router, RoutingDecision
from ..agents import autoscale_engine
from ..keypool import key_pool, KeyPoolExhaustedError
from ..cache import response_cache, make_key
from ..circuit import breaker
from ..concurrency import gate, QueueFullError
from ..cost import budget, compute_cost
from ..bandit import adaptive_router
from ..web import DASHBOARD_HTML
from .. import logging_config  # noqa: F401  (configures structlog on import)
from . import runtime
from .nim_client import NIMClient, nim_client  # noqa: F401  (re-exported)
from .images import fetch_as_data_url as _fetch_as_data_url, inline_remote_images as _inline_remote_images  # noqa: F401
from .recording import (  # noqa: F401
    record_cost as _record_cost,
    record_throughput as _record_throughput,
    record_stream_usage as _record_stream_usage,
)
from .streaming import (  # noqa: F401
    format_streaming_chunk,
    stream_chat_completion,
    stream_nim_request as _stream_nim_request,
)
from .completion import (  # noqa: F401
    should_fallback as _should_fallback,
    complete_with_fallback as _complete_with_fallback,
)

# Structured logging (unified with the router/agent layers).
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
logger = structlog.get_logger()

# Background task set retained for shutdown cleanup.
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
background_tasks = set()


# @spec[PROJECT_PROFILE.md#Requirements]
def _load_metrics() -> None:
    """Restore persisted metrics on startup, if enabled and present."""
    if not settings.persist_metrics:
        return
    path = Path(settings.metrics_file)
    if not path.exists():
        return
    try:
        metrics.load(json.loads(path.read_text()))
        logger.info("metrics restored", file=str(path))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("failed to restore metrics", error=str(e) or repr(e))


# @spec[PROJECT_PROFILE.md#Requirements]
def _save_metrics() -> None:
    """Persist metrics counters to disk, if enabled."""
    if not settings.persist_metrics:
        return
    try:
        Path(settings.metrics_file).write_text(json.dumps(metrics.dump()))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("failed to persist metrics", error=str(e) or repr(e))


# @spec[PROJECT_PROFILE.md#Requirements]
async def _periodic_metrics_save() -> None:
    """Background task: persist metrics on an interval."""
    while True:
        await asyncio.sleep(settings.metrics_save_interval)
        _save_metrics()


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and tear down the shared HTTP client and background tasks."""
    runtime.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.request_timeout, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    logger.info("HTTP client initialized for NVIDIA NIM API")

    _load_metrics()
    saver: Optional[asyncio.Task] = None
    if settings.persist_metrics:
        saver = asyncio.create_task(_periodic_metrics_save())

    try:
        yield
    finally:
        if saver:
            saver.cancel()
        _save_metrics()
        if runtime.http_client:
            await runtime.http_client.aclose()
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

    # Track active connections for the live dashboard. Only count real API
    # traffic (/v1/*) toward total requests — not health/metrics polling, which
    # would otherwise tick up once per dashboard refresh.
    metrics.connection_opened()
    if request.url.path.startswith("/v1/"):
        metrics.note_request()
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
            "dashboard": "/dashboard",
            "health": "/health",
            "chat_completions": "/v1/chat/completions",
            "embeddings": "/v1/embeddings",
            "models": "/v1/models"
        }
    }


# Web dashboard + playground (self-contained HTML)
# @spec[PROJECT_PROFILE.md#Requirements]
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the live web dashboard + prompt playground."""
    return HTMLResponse(content=DASHBOARD_HTML)


# @spec[PROJECT_PROFILE.md#Requirements]
@app.post("/explain")
async def explain(request: Request):
    """Route a prompt and return the answer *plus* why it routed that way."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    body = await request.json()
    messages = body.get("messages") or (
        [{"role": "user", "content": body["prompt"]}] if body.get("prompt") else []
    )
    if not messages or not any(m.get("content") for m in messages):
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "messages or prompt required",
                              "type": "invalid_request_error", "code": 400}},
        )
    max_tokens = body.get("max_tokens", 200)
    temperature = body.get("temperature", 0.2)

    messages = await _inline_remote_images(messages)
    classification = router.capability_analyzer.classify(messages)
    decision = await router.route_request(
        messages=messages, model=body.get("model"),
        max_tokens=max_tokens, temperature=temperature,
    )
    selected = decision.selected_model
    if not selected:
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": "No suitable model for the task",
                              "type": "service_unavailable", "code": 503}},
        )
    if not budget.allow():
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": "Daily budget exceeded",
                              "type": "budget_exceeded", "code": 503}},
        )

    started = time.time()
    try:
        data, used_model, fell_back = await _complete_with_fallback(
            decision.task_type, selected, messages, max_tokens, temperature, {}
        )
    except KeyPoolExhaustedError:
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": "All API keys are rate-limited",
                              "type": "rate_limit_exceeded", "code": 503}},
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={"error": {"message": str(e) or repr(e),
                              "type": "upstream_error", "code": 502}},
        )

    choice = (data.get("choices") or [{}])[0]
    answer = (choice.get("message") or {}).get("content")
    usage = data.get("usage") or {}
    cost = compute_cost(
        used_model, int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )
    return {
        "answer": answer,
        "routing": {
            "task_type": classification.task_type.value,
            "confidence": classification.confidence,
            "scores": classification.scores,
            "selected_model": used_model.model_id,
            "parameters_b": used_model.parameters_b,
            "fell_back": fell_back,
            "reasoning": decision.reasoning,
        },
        "usage": usage,
        "cost_usd": round(cost, 6),
        "latency_ms": round((time.time() - started) * 1000.0, 1),
        "request_id": request_id,
    }


# Readiness check endpoint
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    # Check HTTP client readiness
    http_ready = runtime.http_client is not None and not runtime.http_client.is_closed

    # TODO: Add actual NVIDIA API connectivity check
    nvidia_api_ready = key_pool.has_keys()

    open_models = breaker.open_models()

    status = "ready" if (http_ready and nvidia_api_ready) else "not_ready"

    return {
        "status": status,
        "timestamp": time.time(),
        "checks": {
            "http_client": "ready" if http_ready else "not_initialized",
            "nvidia_api": "configured" if nvidia_api_ready else "not_configured",
            "unhealthy_models": open_models,
        },
    }


# Live metrics endpoint (consumed by the TUI dashboard)
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
def _full_snapshot() -> dict:
    """Assemble the full live-metrics snapshot from every subsystem."""
    snapshot = metrics.snapshot()
    snapshot["routing_stats"] = router.get_routing_stats()
    snapshot["api_keys"] = key_pool.snapshot()
    snapshot["cache"] = response_cache.snapshot()
    snapshot["circuits"] = breaker.snapshot()
    snapshot["concurrency"] = gate.snapshot()
    snapshot["budget"] = budget.snapshot()
    snapshot["adaptive_routing"] = adaptive_router.snapshot()
    return snapshot


@app.get("/metrics")
async def get_metrics():
    """Return live gateway metrics and routing statistics (JSON)."""
    return _full_snapshot()


# @spec[PROJECT_PROFILE.md#Requirements]
@app.get("/metrics/prometheus")
async def get_metrics_prometheus():
    """Prometheus text exposition of the live metrics (for scraping)."""
    from ..prometheus import render_prometheus

    return PlainTextResponse(
        render_prometheus(_full_snapshot()),
        media_type="text/plain; version=0.0.4",
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
async def chat_completions(request: Request, background_tasks: BackgroundTasks):  # noqa: C901
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

        # Response cache: identical non-streaming requests are served from cache,
        # skipping routing and the upstream NIM call entirely.
        cache_key = None
        if settings.enable_cache and not stream:
            cache_key = make_key({k: v for k, v in body.items() if k != "stream"})
            cached = response_cache.get(cache_key)
            if cached is not None:
                return JSONResponse(
                    content=cached,
                    headers={"X-Request-ID": request_id, "X-Cache": "HIT"},
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

        # Budget guardrail: refuse new upstream work once the daily cap is hit.
        if not budget.allow():
            raise HTTPException(
                status_code=503,
                headers={"Retry-After": "3600"},
                detail={
                    "error": {
                        "message": "Daily budget exceeded; try again later",
                        "type": "budget_exceeded",
                        "code": 503,
                        "request_id": request_id,
                    }
                },
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

        # Non-streaming response.
        # Backpressure: bound concurrent upstream requests; shed load when the
        # queue is full or the wait exceeds the timeout.
        acquired = False
        if settings.enable_concurrency_limit:
            try:
                await gate.acquire()
                acquired = True
            except QueueFullError:
                raise HTTPException(
                    status_code=503,
                    headers={"Retry-After": "1"},
                    detail={
                        "error": {
                            "message": "Server busy; please retry shortly",
                            "type": "overloaded",
                            "code": 503,
                            "request_id": request_id,
                        }
                    },
                )

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
                autoscale_latency_ms = (time.time() - nim_start) * 1000.0
                metrics.record_latency(selected_model.model_id, autoscale_latency_ms)
                metrics.record_tokens(
                    selected_model.model_id,
                    int(orchestrated["usage"].get("total_tokens") or 0),
                )
                _record_cost(selected_model, orchestrated["usage"])
                _record_throughput(
                    selected_model.model_id, orchestrated["usage"], autoscale_latency_ms
                )
            else:
                # Call the selected model, failing over to the next-best model
                # for the task on hard upstream errors (404 / 5xx / transport).
                response_data, used_model, fell_back = await _complete_with_fallback(
                    routing_decision.task_type,
                    selected_model,
                    messages,
                    max_tokens,
                    temperature,
                    body_without_model,
                )
                if fell_back:
                    response_headers["X-Selected-Model"] = used_model.model_id
                    response_headers["X-Model-Fallback"] = "true"

            # Cache the successful response for identical future requests.
            if cache_key is not None:
                response_cache.set(cache_key, response_data)
            response_headers["X-Cache"] = "MISS"

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
        finally:
            if acquired:
                gate.release()
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
                "parameters_b": model.parameters_b,
                "context_window": model.context_window,
                "supports_vision": model.supports_vision,
                "supports_streaming": model.supports_streaming,
            }
        )
    return {"object": "list", "data": data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
