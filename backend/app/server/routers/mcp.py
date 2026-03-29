"""MCP configuration API — server-layer implementation using DeerFlowClient."""

import logging

from fastapi import APIRouter, HTTPException

from ..deps import get_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/mcp/config")
async def get_mcp_config():
    return get_client().get_mcp_config()


@router.put("/mcp/config")
async def update_mcp_config(request: dict):
    try:
        servers = request.get("mcp_servers", {})
        return get_client().update_mcp_config(servers)
    except Exception as e:
        logger.exception("Failed to update MCP configuration")
        raise HTTPException(status_code=500, detail=f"Failed to update MCP configuration: {e}")
