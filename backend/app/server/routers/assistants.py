"""Minimal assistants endpoint for LangGraph SDK compatibility."""

from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/assistants")


@router.post("/search")
async def search_assistants(request: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [
        {
            "assistant_id": "lead_agent",
            "graph_id": "lead_agent",
            "name": "DeerFlow Lead Agent",
            "metadata": {},
            "config": {},
            "version": 1,
        }
    ]
