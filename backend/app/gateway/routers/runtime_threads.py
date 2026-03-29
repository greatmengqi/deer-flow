"""Thread CRUD + state endpoints compatible with LangGraph Platform API."""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.gateway.deps import get_store

router = APIRouter(prefix="/threads")


# ── Request schemas ──────────────────────────────────────────────────────────


class CreateThreadRequest(BaseModel):
    thread_id: str | None = None
    metadata: dict[str, Any] | None = None
    if_exists: str | None = None


class SearchThreadsRequest(BaseModel):
    metadata: dict[str, Any] | None = None
    ids: list[str] | None = None
    limit: int = 50
    offset: int = 0
    status: str | None = None
    sort_by: str = "updated_at"
    sort_order: str = "desc"
    select: list[str] | None = None
    values: dict[str, Any] | None = None


class UpdateStateRequest(BaseModel):
    values: dict[str, Any]
    checkpoint_id: str | None = None
    checkpoint: dict[str, Any] | None = None
    as_node: str | None = None


class GetHistoryRequest(BaseModel):
    limit: int = 10


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_checkpoint(thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "checkpoint_ns": "",
        "checkpoint_id": checkpoint_id or str(uuid.uuid4()),
    }


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("")
async def create_thread(request: CreateThreadRequest | None = None) -> dict[str, Any]:
    store = get_store()
    req = request or CreateThreadRequest()
    record = store.create(thread_id=req.thread_id, metadata=req.metadata)
    return record.to_dict()


@router.post("/search")
async def search_threads(request: SearchThreadsRequest | None = None) -> list[dict[str, Any]]:
    store = get_store()
    req = request or SearchThreadsRequest()
    return store.search(
        limit=req.limit,
        offset=req.offset,
        sort_by=req.sort_by,
        sort_order=req.sort_order,
        select=req.select,
    )


@router.get("/{thread_id}")
async def get_thread(thread_id: str) -> dict[str, Any]:
    store = get_store()
    record = store.get(thread_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return record.to_dict()


@router.delete("/{thread_id}")
async def delete_thread(thread_id: str):
    store = get_store()
    store.delete(thread_id)
    # LangGraph SDK expects 204 or silent success; never 404.
    return None


@router.get("/{thread_id}/state")
async def get_thread_state(thread_id: str) -> dict[str, Any]:
    store = get_store()
    record = store.get(thread_id)
    if record is None:
        # Return empty state instead of 404 — the SDK may probe before a run.
        return {
            "values": {},
            "next": [],
            "tasks": [],
            "metadata": {},
            "checkpoint": None,
            "parent_checkpoint": None,
        }
    return {
        "values": record.values,
        "next": [],
        "tasks": [],
        "metadata": record.metadata,
        "created_at": record.created_at,
        "checkpoint": _make_checkpoint(thread_id),
        "parent_checkpoint": None,
    }


@router.post("/{thread_id}/state")
async def update_thread_state(thread_id: str, request: UpdateStateRequest) -> dict[str, Any]:
    store = get_store()
    store.get_or_create(thread_id)
    store.update_values(thread_id, request.values)
    return {"checkpoint": _make_checkpoint(thread_id)}


@router.post("/{thread_id}/history")
async def get_thread_history(thread_id: str, request: GetHistoryRequest | None = None) -> list[dict[str, Any]]:
    store = get_store()
    record = store.get(thread_id)
    if record is None or not record.values:
        return []
    return [
        {
            "values": record.values,
            "next": [],
            "tasks": [],
            "metadata": record.metadata,
            "created_at": record.created_at,
            "checkpoint": _make_checkpoint(thread_id),
            "parent_checkpoint": None,
        }
    ]
