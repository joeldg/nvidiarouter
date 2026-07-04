# @spec[PROJECT_PROFILE.md]
"""
FastAPI server for NVIDIA-SmartRoute-CLI providing OpenAI-compatible endpoints.
"""

import logging
import time
import uuid
import json
from typing import Optional, Dict, Any, List, AsyncGenerator
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio

from ..config import settings
from ..routing.router import router, TaskType, RoutingDecision

# Configure logging
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
logger = logging.getLogger(__name__)

# Create FastAPI app
# @spec[PROJECT_PROFILE.md#Token Budget Class]
app = FastAPI(
    title="NVIDIA-SmartRoute-CLI API",
    description="OpenAI-compatible API gateway for NVIDIA NIM models",
    version="0.1.0",
    docs_url="/docs" if not settings.debug else None,
    redoc_url="/redoc" if not settings.debug else None,
)

# Add CORS middleware
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP client for NVIDIA NIM API
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
http_client: Optional[httpx.AsyncClient] = None

# Background task for updating model performance
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
background_tasks = set()


# Middleware for request logging and timing
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and their processing time."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Add request ID to request state for tracing
    request.state.request_id = request_id
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log request details
    logger.info(
        f"Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "process_time": round(process_time, 4),
            "client_host": request.client.host if request.client else None,
        }
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
        f"Unhandled exception: {exc}",
        extra={
            "request_id": request_id,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
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


# Startup and shutdown events
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.on_event("startup")
async def startup_event():
    """Initialize HTTP client on startup."""
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )
    logger.info("HTTP client initialized for NVIDIA NIM API")


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up HTTP client on shutdown."""
    global http_client
    if http_client:
        await http_client.aclose()
    # Cancel any remaining background tasks
    for task in background_tasks:
        task.cancel()
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)
    logger.info("Application shutting down")


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
    nvidia_api_ready = bool(settings.nvidia_nim_api_key)  # Simple check for now
    
    status = "ready" if (http_ready and nvidia_api_ready) else "not_ready"
    
    return {
        "status": status,
        "timestamp": time.time(),
        "checks": {
            "http_client": "ready" if http_ready else "not_initialized",
            "nvidia_api": "configured" if nvidia_api_ready else "not_configured",
        },
    }


# NVIDIA NIM API client
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
class NIMClient:
    """Client for interacting with NVIDIA NIM API."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
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
        
        response = await http_client.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    async def embeddings(
        self, 
        model: str, 
        input: str | list[str],
        encoding_format: str = "float"
    ) -> dict:
        """Send embedding request to NVIDIA NIM."""
        payload = {
            "model": model,
            "input": input,
            "encoding_format": encoding_format
        }
        
        url = f"{self.base_url}/embeddings"
        
        response = await http_client.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    async def models(self) -> dict:
        """Get available models from NVIDIA NIM."""
        url = f"{self.base_url}/models"
        
        response = await http_client.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()


# Initialize NIM client
# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
nim_client = NIMClient(
    base_url=settings.nvidia_nim_base_url,
    api_key=settings.nvidia_nim_api_key
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
    try:
        async for line in await _stream_nim_request(
            model, messages, True, max_tokens, temperature, **kwargs
        ):
            yield line
    except Exception as e:
        logger.error(f"Error in stream_chat_completion: {e}")
        # Yield an error chunk in OpenAI format
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "internal_error",
                "param": None,
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    finally:
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
    
    auth_header = {}
    if settings.nvidia_nim_api_key:
        auth_header = {
            "Authorization": f"Bearer {settings.nvidia_nim_api_key}"
        }
    
    async with http_client.stream(
        "POST", 
        url, 
        json=payload, 
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            **auth_header
        }
    ) as response:
        response.raise_for_status()
        
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                if data.strip() == "[DONE]":
                    yield f"data: [DONE]\n\n"
                    break
                else:
                    try:
                        # Parse the JSON data from NIM
                        json_data = json.loads(data)
                        # Forward it as-is (it should already be in OpenAI format)
                        yield f"data: {json.dumps(json_data)}\n\n"
                    except json.JSONDecodeError:
                        # If it's not JSON, wrap it in a basic chunk
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': data}}]})}\n\n"


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
            f"Request performance recorded",
            extra={
                "request_id": request_id,
                "response_time": round(response_time, 4),
                "selected_model": routing_decision.selected_model.model_id if routing_decision.selected_model else None,
                "task_type": routing_decision.task_type.value if routing_decision.task_type else None,
                "confidence": routing_decision.confidence if routing_decision else None,
                "routing_reasoning": getattr(routing_decision, 'reasoning', None)
            }
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
        
        # Remove parameters we handle separately
        body_without_model = {k: v for k, v in body.items() 
                             if k not in ["model", "stream", "max_tokens", "temperature"]}
        
        logger.info(
            f"Chat completion request received",
            extra={
                "request_id": request_id,
                "message_count": len(messages),
                "stream": stream,
                "model_hint": model
            }
        )
        
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
            f"Routing decision: {selected_model.model_id} "
            f"(confidence: {routing_decision.confidence:.2f}) for task {routing_decision.task_type.value}",
            extra={
                "request_id": request_id,
                "selected_model": selected_model.model_id,
                "task_type": routing_decision.task_type.value,
                "confidence": routing_decision.confidence,
                "reasoning": getattr(routing_decision, 'reasoning', 'N/A')
            }
        )
        
        # Track performance in background
        def record_performance():
            background_tasks.add_task(
                _record_request_performance, 
                request_id, 
                routing_decision, 
                start_time
            )
        
        background_tasks.add_task(record_performance)
        
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
        
        # Non-streaming response
        try:
            response_data = await nim_client.chat_completions(
                model=selected_model.model_id,
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
                **body_without_model
            )
            
            # Add custom headers for tracing
            response_headers = {
                "X-Request-ID": request_id,
                "X-Selected-Model": selected_model.model_id,
                "X-Task-Type": routing_decision.task_type.value,
                "X-Routing-Confidence": str(routing_decision.confidence)
            }
            
            # Add the headers to the response
            for key, value in response_headers.items():
                response_data[key] = value
            
            return response_data
        except Exception as e:
            logger.error(f"Error in chat_completions: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": str(e),
                        "type": "internal_error",
                        "code": 500,
                        "request_id": request_id,
                    }
                }
            )
    except Exception as e:
        # Catch any other exceptions that occur in the outer try block
        logger.error(f"Error in chat_completions: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
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
    # For simplicity, we're not implementing routing for embeddings yet
    # In a full implementation, this would use the router to select an appropriate model
    body = await request.json()
    model = body.get("model")
    input_data = body.get("input")
    encoding_format = body.get("encoding_format", "float")
    
    try:
        response = await nim_client.embeddings(
            model=model,
            input=input_data,
            encoding_format=encoding_format
        )
        return response
    except Exception as e:
        logger.error(f"Error in embeddings: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": 500,
                }
            }
        )


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint."""
    try:
        response = await nim_client.models()
        return response
    except Exception as e:
        logger.error(f"Error in list_models: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": 500,
                }
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)