"""Application lifecycle and transport configuration."""

import math
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from .http import APIError, RequestLimitsMiddleware
from .openai import router
from .openai.errors import error_response, exception_response

if TYPE_CHECKING:
    from ..engine.config import EngineConfig


@dataclass(frozen=True, slots=True)
class ServerConfig:
    served_model_name: str | None = None
    api_key: str | None = None
    max_request_bytes: int = 8 * 1024 * 1024
    max_batch_size: int = 32
    send_timeout: float = 30.0

    def __post_init__(self) -> None:
        if self.max_request_bytes <= 0 or self.max_batch_size <= 0:
            raise ValueError("Request and batch limits must be positive")
        if not math.isfinite(self.send_timeout) or self.send_timeout <= 0:
            raise ValueError("send_timeout must be finite and positive")
        if self.api_key == "" or self.served_model_name == "":
            raise ValueError("API key and served model name must not be empty")


def create_app(
    engine_config: "EngineConfig", config: ServerConfig | None = None
) -> FastAPI:
    config = config or ServerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from ..engine import AsyncEngine

        engine = await AsyncEngine.create(engine_config)
        app.state.engine = engine
        app.state.model_name = config.served_model_name or engine.model_name
        app.state.created = int(time.time())
        try:
            yield
        finally:
            await engine.aclose()

    app = FastAPI(title="Zephyr", lifespan=lifespan)
    app.state.config = config
    app.add_middleware(
        RequestLimitsMiddleware,
        max_bytes=config.max_request_bytes,
        api_key=config.api_key,
        error_response=error_response,
        public_paths=("/health",),
    )
    for error_type in (APIError, RequestValidationError, HTTPException, Exception):
        app.add_exception_handler(error_type, exception_response)

    @app.get("/health")
    async def health(request: Request):
        engine = getattr(request.app.state, "engine", None)
        if engine is None or not engine.is_running:
            raise APIError("The inference engine is unavailable", 503)
        return {"status": "ok"}

    app.include_router(router)
    return app
