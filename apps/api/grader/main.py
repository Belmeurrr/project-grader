import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import sentry_sdk
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from grader import __version__
from grader.logging import configure_logging, get_logger
from grader.routers import cert, health, submissions
from grader.services.rate_limit import (
    RateLimitExceeded,
    limiter,
    rate_limit_exceeded_handler,
)
from grader.settings import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # Sentry must come before logging so any failure during configure_logging
    # itself is captured. It's opt-in: when SENTRY_DSN is unset, we skip the
    # init entirely — no network calls, no global hub installed.
    if settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            traces_sample_rate=0.1,
            environment=settings.env,
            integrations=[
                FastApiIntegration(),
                CeleryIntegration(),
                SqlalchemyIntegration(),
            ],
        )
    configure_logging()
    log = get_logger(__name__)
    log.info("api.startup", version=__version__, sentry_enabled=bool(settings.sentry_dsn))
    yield
    log.info("api.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Project Grader API",
        version=__version__,
        description="AI trading card grading and authentication.",
        lifespan=lifespan,
    )

    # slowapi: in-memory limiter. Single-process for now; we'll swap
    # ``storage_uri`` to Redis when we go multi-instance (Phase-2).
    # ``app.state.limiter`` is the contract slowapi reads inside its
    # decorator/middleware, so don't drop this assignment.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timer — emits one structured log line per request with the
    # duration in ms. Sits AFTER CORS in registration order, which (in
    # FastAPI/Starlette) means it runs INSIDE CORS in the dispatch chain:
    # CORS sees the request first, the timer wraps the actual route. This
    # is what we want — preflight OPTIONS handled by CORS shouldn't show
    # up as application traffic, and timing measures real route work.
    @app.middleware("http")
    async def request_timer(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        structlog.get_logger("grader.http").info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response

    app.include_router(health.router)
    app.include_router(submissions.router)
    app.include_router(cert.router)
    return app


app = create_app()
