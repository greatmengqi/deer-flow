"""Memory API — server-layer implementation using DeerFlowClient."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..deps import get_client

router = APIRouter(prefix="/api")


class FactCreateRequest(BaseModel):
    content: str = Field(..., min_length=1)
    category: str = Field(default="context")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class FactPatchRequest(BaseModel):
    content: str | None = Field(default=None, min_length=1)
    category: str | None = Field(default=None)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


@router.get("/memory")
async def get_memory():
    return get_client().get_memory()


@router.post("/memory/reload")
async def reload_memory():
    return get_client().reload_memory()


@router.delete("/memory")
async def clear_memory():
    try:
        return get_client().clear_memory()
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to clear memory data.") from exc


@router.post("/memory/facts")
async def create_fact(request: FactCreateRequest):
    try:
        return get_client().create_memory_fact(content=request.content, category=request.category, confidence=request.confidence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/memory/facts/{fact_id}")
async def delete_fact(fact_id: str):
    try:
        return get_client().delete_memory_fact(fact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Fact '{fact_id}' not found.") from exc


@router.patch("/memory/facts/{fact_id}")
async def update_fact(fact_id: str, request: FactPatchRequest):
    try:
        return get_client().update_memory_fact(fact_id=fact_id, content=request.content, category=request.category, confidence=request.confidence)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Fact '{fact_id}' not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/memory/config")
async def get_memory_config():
    return get_client().get_memory_config()


@router.get("/memory/status")
async def get_memory_status():
    return get_client().get_memory_status()
