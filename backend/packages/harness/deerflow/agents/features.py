"""Declarative feature flags for create_deep_agent.

Pure data class — no I/O, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain.agents.middleware import AgentMiddleware


@dataclass
class AgentFeatures:
    """Declarative feature flags for ``create_deep_agent``.

    Each feature accepts:
    - ``True``: use the built-in default middleware
    - ``False``: disable
    - An ``AgentMiddleware`` instance: use this custom implementation instead
    """

    sandbox: bool | AgentMiddleware = True
    memory: bool | AgentMiddleware = False
    summarization: bool | AgentMiddleware = False
    subagent: bool | AgentMiddleware = False
    vision: bool | AgentMiddleware = False
    auto_title: bool | AgentMiddleware = False
