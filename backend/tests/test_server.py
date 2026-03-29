"""Tests for the unified LangGraph-compatible server."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# ── ThreadStore unit tests ───────────────────────────────────────────────────


class TestThreadStore:
    @pytest.fixture()
    def store(self):
        from app.server.store import ThreadStore

        return ThreadStore(max_threads=5)

    def test_create_generates_uuid(self, store):
        record = store.create()
        assert record.thread_id
        assert record.status == "idle"
        assert record.values == {}

    def test_create_with_explicit_id(self, store):
        record = store.create(thread_id="my-thread")
        assert record.thread_id == "my-thread"

    def test_get_existing(self, store):
        store.create(thread_id="t1")
        assert store.get("t1") is not None
        assert store.get("t1").thread_id == "t1"

    def test_get_missing(self, store):
        assert store.get("nonexistent") is None

    def test_get_or_create_existing(self, store):
        store.create(thread_id="t1", metadata={"key": "val"})
        record = store.get_or_create("t1")
        assert record.metadata == {"key": "val"}

    def test_get_or_create_new(self, store):
        record = store.get_or_create("new-thread")
        assert record.thread_id == "new-thread"
        assert store.get("new-thread") is not None

    def test_search_returns_all(self, store):
        store.create(thread_id="a")
        store.create(thread_id="b")
        results = store.search()
        assert len(results) == 2

    def test_search_pagination(self, store):
        for i in range(5):
            store.create(thread_id=f"t{i}")
        results = store.search(limit=2, offset=1)
        assert len(results) == 2

    def test_search_select_fields(self, store):
        store.create(thread_id="t1")
        results = store.search(select=["thread_id", "status"])
        assert "thread_id" in results[0]
        assert "status" in results[0]
        assert "values" not in results[0]

    def test_delete(self, store):
        store.create(thread_id="t1")
        assert store.delete("t1") is True
        assert store.get("t1") is None

    def test_delete_missing(self, store):
        assert store.delete("nonexistent") is False

    def test_update_values(self, store):
        store.create(thread_id="t1")
        store.update_values("t1", {"title": "Hello"})
        assert store.get("t1").values["title"] == "Hello"

    def test_update_values_merges(self, store):
        store.create(thread_id="t1")
        store.update_values("t1", {"title": "Hello"})
        store.update_values("t1", {"artifacts": ["a.txt"]})
        vals = store.get("t1").values
        assert vals["title"] == "Hello"
        assert vals["artifacts"] == ["a.txt"]

    def test_set_busy_idle(self, store):
        store.create(thread_id="t1")
        store.set_busy("t1")
        assert store.get("t1").status == "busy"
        store.set_idle("t1")
        assert store.get("t1").status == "idle"

    def test_concurrent_busy_idle(self, store):
        store.create(thread_id="t1")
        store.set_busy("t1")
        store.set_busy("t1")  # two concurrent runs
        store.set_idle("t1")  # first finishes
        assert store.get("t1").status == "busy"  # still busy
        store.set_idle("t1")  # second finishes
        assert store.get("t1").status == "idle"

    def test_eviction_at_capacity(self, store):
        # max_threads=5, create 5 then one more
        for i in range(5):
            store.create(thread_id=f"t{i}")
        store.create(thread_id="t5")
        # Should still have at most 5
        assert len(store.search(limit=100)) == 5
        # Newest should exist
        assert store.get("t5") is not None

    def test_eviction_skips_busy(self, store):
        # Fill with busy threads
        for i in range(5):
            store.create(thread_id=f"t{i}")
            store.set_busy(f"t{i}")
        # Create one more — can't evict any, so goes over capacity
        store.create(thread_id="overflow")
        assert store.get("overflow") is not None

    def test_to_dict_always_includes_thread_id(self, store):
        store.create(thread_id="t1")
        record = store.get("t1")
        d = record.to_dict(select=["status"])
        assert "thread_id" in d


# ── API endpoint tests ───────────────────────────────────────────────────────


@pytest.fixture()
def client():
    """Create test client with the LangGraph-compatible routes."""
    from fastapi import FastAPI

    from app.server import deps
    from app.server.routers import assistants, runs, threads
    from app.server.store import ThreadStore

    # Reset singletons
    deps.store = ThreadStore()
    deps.client = MagicMock()

    app = FastAPI()
    app.include_router(threads.router)
    app.include_router(runs.router)
    app.include_router(assistants.router)

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "deer-flow-server"}

    return TestClient(app)


class TestThreadEndpoints:
    def test_create_thread(self, client):
        resp = client.post("/threads")
        assert resp.status_code == 200
        data = resp.json()
        assert "thread_id" in data
        assert data["status"] == "idle"

    def test_create_thread_with_id(self, client):
        resp = client.post("/threads", json={"thread_id": "custom-id"})
        assert resp.json()["thread_id"] == "custom-id"

    def test_search_threads(self, client):
        client.post("/threads")
        client.post("/threads")
        resp = client.post("/threads/search", json={"limit": 10})
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_thread(self, client):
        tid = client.post("/threads").json()["thread_id"]
        resp = client.get(f"/threads/{tid}")
        assert resp.status_code == 200
        assert resp.json()["thread_id"] == tid

    def test_get_thread_404(self, client):
        resp = client.get("/threads/nonexistent")
        assert resp.status_code == 404

    def test_delete_thread(self, client):
        tid = client.post("/threads").json()["thread_id"]
        resp = client.delete(f"/threads/{tid}")
        assert resp.status_code == 200

    def test_get_state_empty(self, client):
        resp = client.get("/threads/unknown/state")
        assert resp.status_code == 200
        assert resp.json()["values"] == {}

    def test_update_state(self, client):
        tid = client.post("/threads").json()["thread_id"]
        resp = client.post(f"/threads/{tid}/state", json={"values": {"title": "New"}})
        assert resp.status_code == 200
        assert "checkpoint" in resp.json()
        # Verify
        state = client.get(f"/threads/{tid}/state").json()
        assert state["values"]["title"] == "New"

    def test_history_empty(self, client):
        resp = client.post("/threads/unknown/history", json={"limit": 1})
        assert resp.json() == []

    def test_history_with_state(self, client):
        tid = client.post("/threads").json()["thread_id"]
        client.post(f"/threads/{tid}/state", json={"values": {"title": "Test"}})
        resp = client.post(f"/threads/{tid}/history", json={"limit": 1})
        data = resp.json()
        assert len(data) == 1
        assert data[0]["values"]["title"] == "Test"


class TestAssistantEndpoints:
    def test_search_assistants(self, client):
        resp = client.post("/assistants/search")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["assistant_id"] == "lead_agent"


class TestRunStreamEndpoint:
    def test_stream_returns_sse(self, client):
        from deerflow.client import StreamEvent

        # Mock DeerFlowClient.stream() to yield known events (dual mode)
        mock_client = MagicMock()
        mock_client.stream.return_value = iter(
            [
                StreamEvent(type="messages-tuple", data={"type": "ai", "content": "hi", "id": "m1"}, metadata={"langgraph_checkpoint_ns": ""}),
                StreamEvent(type="values", data={"title": "Test", "messages": [], "artifacts": []}),
                StreamEvent(type="end", data={"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}),
            ]
        )

        from app.server import deps

        deps.client = mock_client

        resp = client.post(
            "/threads/test-t/runs/stream",
            json={
                "assistant_id": "lead_agent",
                "input": {"messages": [{"type": "human", "content": "hello"}]},
                "stream_mode": ["values", "messages-tuple"],
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "/threads/test-t/runs/" in resp.headers["content-location"]

        # Parse SSE events
        body = resp.text
        events = []
        for block in body.strip().split("\n\n"):
            lines = block.strip().split("\n")
            if len(lines) >= 2:
                event_name = lines[0].replace("event: ", "")
                event_data = json.loads(lines[1].replace("data: ", ""))
                events.append((event_name, event_data))

        event_types = [e[0] for e in events]
        assert "values" in event_types
        assert "messages" in event_types

        # Messages event should be [msg, metadata] tuple
        msg_event = next(e for e in events if e[0] == "messages")
        assert isinstance(msg_event[1], list)
        assert len(msg_event[1]) == 2
        assert msg_event[1][0]["type"] == "ai"

    def test_stream_extracts_text_from_content_blocks(self, client):
        from deerflow.client import StreamEvent

        mock_client = MagicMock()
        mock_client.stream.return_value = iter(
            [
                StreamEvent(type="end", data={}),
            ]
        )

        from app.server import deps

        deps.client = mock_client

        client.post(
            "/threads/t1/runs/stream",
            json={
                "input": {"messages": [{"type": "human", "content": [{"type": "text", "text": "extracted"}]}]},
            },
        )

        # Verify the text was extracted and passed to client.stream()
        mock_client.stream.assert_called_once()
        assert mock_client.stream.call_args[0][0] == "extracted"

    def test_stream_passes_config_overrides(self, client):
        from deerflow.client import StreamEvent

        mock_client = MagicMock()
        mock_client.stream.return_value = iter([StreamEvent(type="end", data={})])

        from app.server import deps

        deps.client = mock_client

        client.post(
            "/threads/t1/runs/stream",
            json={
                "input": {"messages": [{"type": "human", "content": "hi"}]},
                "context": {"thinking_enabled": False, "is_plan_mode": True},
                "config": {"recursion_limit": 50},
            },
        )

        kwargs = mock_client.stream.call_args[1]
        assert kwargs["thinking_enabled"] is False
        assert kwargs["plan_mode"] is True
        assert kwargs["recursion_limit"] == 50


class TestHealthEndpoints:
    def test_ok(self, client):
        resp = client.get("/ok")
        assert resp.json() == {"ok": True}

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "healthy"
