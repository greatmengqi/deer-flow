"""DeerFlow unified server.

Modes (controlled by DEERFLOW_STANDALONE env var):
  - false (default): LangGraph API + server-layer REST only.
    Gateway runs as separate process on port 8001.
  - true: Mounts everything — no separate gateway needed.
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
from .routers import assistants, mcp, memory, models, runs, skills, threads

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_STANDALONE = os.getenv("DEERFLOW_STANDALONE", "").lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_app_config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.exception("Failed to load configuration: %s", e)
        raise

    checkpointer = get_checkpointer()
    deps.client = DeerFlowClient(checkpointer=checkpointer)
    logger.info("DeerFlow server started (standalone=%s)", _STANDALONE)

    if _STANDALONE:
        try:
            from app.channels.service import start_channel_service

            svc = await start_channel_service()
            logger.info("Channel service started: %s", svc.get_status())
        except Exception:
            logger.debug("No IM channels configured or channel service failed to start")

    yield

    if _STANDALONE:
        try:
            from app.channels.service import stop_channel_service

            await stop_channel_service()
        except Exception:
            pass
    logger.info("DeerFlow server stopped")


app = FastAPI(title="DeerFlow Server", lifespan=lifespan)

# ── LangGraph Platform API ──────────────────────────────────────────────────
app.include_router(threads.router)
app.include_router(runs.router)
app.include_router(assistants.router)

# ── Server-layer REST API (always mounted, delegates to DeerFlowClient) ─────
app.include_router(models.router)
app.include_router(mcp.router)
app.include_router(memory.router)
app.include_router(skills.router)

# ── Gateway-only routes (mounted only in standalone mode) ────────────────────
if _STANDALONE:
    from app.gateway.routers import agents as gw_agents
    from app.gateway.routers import artifacts, channels, suggestions, uploads
    from app.gateway.routers import threads as gw_threads

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
