"""FastAPI server application for Swarm Agent multi-provider gateway."""

from __future__ import annotations
from contextlib import asynccontextmanager
import json
import logging
from typing import AsyncIterator, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from swarm.config import GatewayConfig, get_config
from swarm.core.exceptions import (
    ModelNotFoundError,
    ProviderAuthError,
    ProviderPoolExhaustedError,
)
from swarm.core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorDetail,
    ErrorResponse,
    ModelListResponse,
)
from swarm.router.pool import ProviderPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("swarm.main")

_global_pool: Optional[ProviderPool] = None


def get_global_pool(config: Optional[GatewayConfig] = None) -> ProviderPool:
    global _global_pool
    if _global_pool is None:
        _global_pool = ProviderPool(config or get_config())
    return _global_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI gateway server."""
    config = get_config()
    pool = get_global_pool(config)
    app.state.router = pool
    logger.info(f"Swarm Gateway initialized with {len(pool.providers)} providers.")
    yield
    logger.info("Swarm Gateway shutting down, closing provider connections...")
    await pool.aclose()


def create_app(config: Optional[GatewayConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = config or get_config()

    app = FastAPI(
        title="Swarm Multi-Provider Gateway",
        description="High-concurrency, OpenAI-compatible proxy gateway aggregating free-tier quotas across 8 AI providers.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_router() -> ProviderPool:
        if hasattr(app.state, "router") and app.state.router:
            return app.state.router
        return get_global_pool(cfg)

    @app.exception_handler(ProviderPoolExhaustedError)
    async def pool_exhausted_handler(request: Request, exc: ProviderPoolExhaustedError):
        logger.warning(f"All providers exhausted for model '{exc.model}': {exc.message}")
        content = {
            "error": {
                "message": exc.message,
                "type": "provider_pool_exhausted_error",
                "param": "model",
                "code": "quota_exhausted_all_providers",
            }
        }
        return JSONResponse(status_code=429, content=content)

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
        content = {
            "error": {
                "message": exc.message,
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        }
        return JSONResponse(status_code=400, content=content)

    @app.exception_handler(ProviderAuthError)
    async def auth_error_handler(request: Request, exc: ProviderAuthError):
        content = {
            "error": {
                "message": exc.message,
                "type": "authentication_error",
                "param": None,
                "code": "invalid_api_key",
            }
        }
        return JSONResponse(status_code=401, content=content)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        router = get_router()

        if request.stream:
            async def event_stream() -> AsyncIterator[str]:
                try:
                    async for chunk in router.route_stream_chat_completion(request):
                        chunk_json = chunk.model_dump_json(exclude_none=True)
                        yield f"data: {chunk_json}\n\n"
                    yield "data: [DONE]\n\n"
                except ProviderPoolExhaustedError as e:
                    err_dict = {
                        "error": {
                            "message": e.message,
                            "type": "provider_pool_exhausted_error",
                            "param": "model",
                            "code": "quota_exhausted_all_providers",
                        }
                    }
                    yield f"data: {json.dumps(err_dict)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream; charset=utf-8",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream; charset=utf-8",
                },
            )

        # Non-streaming response
        response = await router.route_chat_completion(request)
        return response

    @app.get("/health")
    async def health_check():
        router = get_router()
        return router.get_health()

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models():
        router = get_router()
        return router.get_models()

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    uvicorn.run("swarm.main:app", host=cfg.server.host, port=cfg.server.port, reload=False)
