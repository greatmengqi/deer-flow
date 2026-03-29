"""Shared singletons for the LangGraph-compatible server.

Separated from app.py to avoid circular imports between app and routers.
"""

from app.gateway.store import ThreadStore
from deerflow.client import DeerFlowClient

# Module-level singletons, initialized in app.py lifespan
store = ThreadStore()
client: DeerFlowClient | None = None


def get_store() -> ThreadStore:
    return store


def get_client() -> DeerFlowClient:
    if client is None:
        raise RuntimeError("DeerFlowClient not initialized — server still starting?")
    return client
