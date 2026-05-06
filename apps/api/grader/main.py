from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    configure_logging()
    log = get_logger(__name__)
    log.info("api.startup", version=__version__)
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
    app.include_router(health.router)
    app.include_router(submissions.router)
    app.include_router(cert.router)
    return app


app = create_app()
