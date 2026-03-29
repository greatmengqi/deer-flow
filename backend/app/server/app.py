"""Unified DeerFlow server — replaces both langgraph-cli and gateway.

Serves two sets of routes on one port:
  - LangGraph Platform API  (/threads, /assistants, …)  — for the SDK
  - Gateway REST API         (/api/models, /api/skills, …) — for the frontend

Set DEERFLOW_USE_GATEWAY_ROUTERS=true to use the original gateway router
implementations instead of the server-layer ones (A/B testing).
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from deerflow.agents.checkpointer import get_checkpointer
from deerflow.client import DeerFlowClient
from deerflow.config.app_config import get_app_config

from . import deps
from .routers import assistants, runs, threads

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Feature flag: use original gateway routers vs server-layer implementations
_USE_GATEWAY = os.getenv("DEERFLOW_USE_GATEWAY_ROUTERS", "").lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load config (same as gateway)
    try:
        get_app_config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.exception("Failed to load configuration: %s", e)
        raise

    # Init DeerFlowClient
    checkpointer = get_checkpointer()
    deps.client = DeerFlowClient(checkpointer=checkpointer)
    logger.info("DeerFlow unified server started (router_mode=%s)", "gateway" if _USE_GATEWAY else "server")

    # Start IM channel service if configured
    try:
        from app.channels.service import start_channel_service

        svc = await start_channel_service()
        logger.info("Channel service started: %s", svc.get_status())
    except Exception:
        logger.debug("No IM channels configured or channel service failed to start")

    yield

    # Shutdown
    try:
        from app.channels.service import stop_channel_service

        await stop_channel_service()
    except Exception:
        pass
    logger.info("DeerFlow unified server stopped")


app = FastAPI(title="DeerFlow Server", lifespan=lifespan)

# ── LangGraph Platform API (no /api prefix) ─────────────────────────────────
app.include_router(threads.router)
app.include_router(runs.router)
app.include_router(assistants.router)

# ── REST API (/api prefix) ──────────────────────────────────────────────────
if _USE_GATEWAY:
    # A/B: original gateway implementations (import from deerflow.* directly)
    from app.gateway.routers import agents as gw_agents
    from app.gateway.routers import artifacts, channels, mcp, memory, models, skills, suggestions, uploads
    from app.gateway.routers import threads as gw_threads
else:
    # Default: server-layer implementations (delegate to DeerFlowClient)
    # These don't have server-layer equivalents yet — always from gateway
    from app.gateway.routers import agents as gw_agents
    from app.gateway.routers import artifacts, channels, suggestions, uploads
    from app.gateway.routers import threads as gw_threads

    from .routers import mcp, memory, models, skills  # noqa: F811

app.include_router(models.router)
app.include_router(mcp.router)
app.include_router(memory.router)
app.include_router(skills.router)
app.include_router(artifacts.router)
app.include_router(uploads.router)
app.include_router(gw_threads.router)
app.include_router(gw_agents.router)
app.include_router(suggestions.router)
app.include_router(channels.router)


@app.get("/ok")
async def ok() -> dict[str, Any]:
    return {"ok": True}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "healthy", "service": "deer-flow-server"}
