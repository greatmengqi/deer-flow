"""Run streaming endpoint compatible with LangGraph Platform API.

Bridges DeerFlowClient.stream() (sync generator) to SSE over HTTP,
matching the wire format expected by @langchain/langgraph-sdk.
"""

import asyncio
import json
import logging
import threading
import uuid
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..deps import get_client, get_store

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request schema ───────────────────────────────────────────────────────────


class RunStreamRequest(BaseModel):
    assistant_id: str = "lead_agent"
    input: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    stream_mode: list[str] | str | None = None
    stream_subgraphs: bool = False
    stream_resumable: bool = False
    interrupt_before: list[str] | None = None
    interrupt_after: list[str] | None = None
    checkpoint: dict[str, Any] | None = None
    checkpoint_id: str | None = None
    multitask_strategy: str | None = None
    on_completion: str | None = None
    on_disconnect: str | None = None
    webhook: str | None = None
    after_seconds: int | None = None
    if_not_exists: str | None = None
    feedback_keys: list[str] | None = None
    checkpoint_during: bool | None = None
    durability: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _sse(event: str, data: Any) -> str:
    """Format one SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extract_text(input_data: dict[str, Any]) -> str:
    """Pull user text from the SDK input payload."""
    messages = input_data.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    content = last.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _build_overrides(req: RunStreamRequest) -> dict[str, Any]:
    """Build DeerFlowClient.stream() kwargs from the request."""
    overrides: dict[str, Any] = {}
    ctx = req.context or {}
    if "thinking_enabled" in ctx:
        overrides["thinking_enabled"] = ctx["thinking_enabled"]
    if "is_plan_mode" in ctx:
        overrides["plan_mode"] = ctx["is_plan_mode"]
    if "subagent_enabled" in ctx:
        overrides["subagent_enabled"] = ctx["subagent_enabled"]
    if "model_name" in ctx:
        overrides["model_name"] = ctx["model_name"]
    cfg = req.config or {}
    if "recursion_limit" in cfg:
        overrides["recursion_limit"] = cfg["recursion_limit"]
    return overrides


def _requested_modes(req: RunStreamRequest) -> set[str]:
    if req.stream_mode is None:
        return {"values", "messages-tuple"}
    if isinstance(req.stream_mode, str):
        return {req.stream_mode}
    return set(req.stream_mode)


# ── Endpoint ─────────────────────────────────────────────────────────────────


@router.post("/threads/{thread_id}/runs/stream")
async def stream_run(thread_id: str, request: RunStreamRequest):
    client = get_client()
    store = get_store()

    store.get_or_create(thread_id)
    store.set_busy(thread_id)

    run_id = str(uuid.uuid4())
    message = _extract_text(request.input) if request.input else ""
    overrides = _build_overrides(request)
    modes = _requested_modes(request)

    async def event_generator():
        queue: asyncio.Queue[Any] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        cancel = threading.Event()

        def _run():
            try:
                for ev in client.stream(message, thread_id=thread_id, stream_mode="dual", **overrides):
                    if cancel.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, ev)
            except Exception as exc:
                if not cancel.is_set():
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        fut = loop.run_in_executor(None, _run)
        last_values: dict[str, Any] | None = None
        prev_title: str | None = None

        # Emit metadata event (SDK uses this for run bootstrap)
        yield _sse("metadata", {"run_id": run_id, "thread_id": thread_id})

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    logger.exception("Stream error for thread %s", thread_id)
                    yield _sse("error", {"message": "Internal server error", "code": "INTERNAL_ERROR"})
                    yield _sse("end", None)
                    break

                ev = item
                if ev.type == "messages-tuple" and "messages-tuple" in modes:
                    # Skip middleware-internal model calls (title, memory, etc.)
                    if "Middleware" in ev.metadata.get("langgraph_node", ""):
                        continue
                    yield _sse("messages", [ev.data, ev.metadata])

                elif ev.type == "values" and "values" in modes:
                    last_values = ev.data
                    yield _sse("values", ev.data)

                    # Emit updates event when title changes (including clear)
                    title = ev.data.get("title")
                    if title != prev_title and "updates" in modes:
                        yield _sse("updates", {"agent": {"title": title}})
                        prev_title = title

            yield _sse("end", None)

        finally:
            cancel.set()
            if last_values:
                store.update_values(thread_id, last_values)
            store.set_idle(thread_id)
            await fut

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )
