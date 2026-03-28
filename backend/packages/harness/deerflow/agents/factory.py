"""Configuration-free factory for DeerFlow agents.

``create_deep_agent`` accepts plain Python arguments — no YAML files, no
global singletons.  It is the SDK-level entry point sitting between the raw
``langchain.agents.create_agent`` primitive and the config-driven
``make_lead_agent`` application factory.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

from deerflow.agents.features import AgentFeatures
from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from deerflow.agents.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware
from deerflow.agents.middlewares.tool_error_handling_middleware import ToolErrorHandlingMiddleware
from deerflow.agents.thread_state import ThreadState
from deerflow.tools.builtins import ask_clarification_tool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_deep_agent(
    model: BaseChatModel,
    tools: list[BaseTool] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: list[AgentMiddleware] | None = None,
    features: AgentFeatures | None = None,
    state_schema: type | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    name: str = "default",
) -> CompiledStateGraph:
    """Create a DeerFlow agent.  Pure arguments — no config files read.

    Parameters
    ----------
    model:
        Chat model instance.
    tools:
        User-provided tools.  Feature-injected tools are appended automatically.
    system_prompt:
        System message.  ``None`` uses a minimal default.
    middleware:
        **Full takeover** — if provided, this exact list is used.
        Cannot be combined with *features*.
    features:
        Declarative feature flags.  Ignored when *middleware* is given.
    state_schema:
        LangGraph state type.  Defaults to ``ThreadState``.
    checkpointer:
        Optional persistence backend.
    name:
        Agent name (passed to middleware that cares, e.g. ``MemoryMiddleware``).

    Raises
    ------
    ValueError
        If both *middleware* and *features* are provided.
    """
    if middleware is not None and features is not None:
        raise ValueError("Cannot specify both 'middleware' and 'features'.  Use one or the other.")

    effective_tools: list[BaseTool] = list(tools or [])
    effective_state = state_schema or ThreadState

    if middleware is not None:
        effective_middleware = list(middleware)
    else:
        feat = features or AgentFeatures()
        effective_middleware, extra_tools = _assemble_from_features(feat, name=name)
        # Deduplicate by tool name — user-provided tools take priority.
        existing_names = {t.name for t in effective_tools}
        for t in extra_tools:
            if t.name not in existing_names:
                effective_tools.append(t)
                existing_names.add(t.name)

    return create_agent(
        model=model,
        tools=effective_tools or None,
        middleware=effective_middleware,
        system_prompt=system_prompt,
        state_schema=effective_state,
        checkpointer=checkpointer,
        name=name,
    )


# ---------------------------------------------------------------------------
# Internal: feature-driven middleware assembly
# ---------------------------------------------------------------------------


def _assemble_from_features(
    feat: AgentFeatures,
    *,
    name: str = "default",
) -> tuple[list[AgentMiddleware], list[BaseTool]]:
    """Build an ordered middleware chain + extra tools from *feat*.

    Middleware order follows sequential append (no priority numbers):
      1. Sandbox infrastructure (ThreadData → Uploads → Sandbox)
      2. Always-on error handling (DanglingToolCall, ToolError)
      3. Feature middlewares (summarization, auto_title, memory, vision, subagent)
      4. Clarification (always last)

    Each feature value is handled as:
      - ``False``: skip
      - ``True``: create the built-in default middleware
      - ``AgentMiddleware`` instance: use directly (custom replacement)
    """
    chain: list[AgentMiddleware] = []
    extra_tools: list[BaseTool] = []

    # --- Sandbox infrastructure ---
    if feat.sandbox is not False:
        if isinstance(feat.sandbox, AgentMiddleware):
            chain.append(feat.sandbox)
        else:
            from deerflow.agents.middlewares.thread_data_middleware import ThreadDataMiddleware
            from deerflow.agents.middlewares.uploads_middleware import UploadsMiddleware
            from deerflow.sandbox.middleware import SandboxMiddleware

            chain.append(ThreadDataMiddleware(lazy_init=True))
            chain.append(UploadsMiddleware())
            chain.append(SandboxMiddleware(lazy_init=True))

    # --- Always-on error handling ---
    chain.append(DanglingToolCallMiddleware())
    chain.append(ToolErrorHandlingMiddleware())

    # --- Summarization ---
    if feat.summarization is not False:
        if isinstance(feat.summarization, AgentMiddleware):
            chain.append(feat.summarization)
        else:
            from langchain.agents.middleware import SummarizationMiddleware

            chain.append(SummarizationMiddleware())

    # --- Auto Title ---
    if feat.auto_title is not False:
        if isinstance(feat.auto_title, AgentMiddleware):
            chain.append(feat.auto_title)
        else:
            from deerflow.agents.middlewares.title_middleware import TitleMiddleware

            chain.append(TitleMiddleware())

    # --- Memory ---
    if feat.memory is not False:
        if isinstance(feat.memory, AgentMiddleware):
            chain.append(feat.memory)
        else:
            from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware

            chain.append(MemoryMiddleware(agent_name=name))

    # --- Vision ---
    if feat.vision is not False:
        if isinstance(feat.vision, AgentMiddleware):
            chain.append(feat.vision)
        else:
            from deerflow.agents.middlewares.view_image_middleware import ViewImageMiddleware

            chain.append(ViewImageMiddleware())
        from deerflow.tools.builtins import view_image_tool

        extra_tools.append(view_image_tool)

    # --- Subagent ---
    if feat.subagent is not False:
        if isinstance(feat.subagent, AgentMiddleware):
            chain.append(feat.subagent)
        else:
            from deerflow.agents.middlewares.subagent_limit_middleware import SubagentLimitMiddleware

            chain.append(SubagentLimitMiddleware())
        from deerflow.tools.builtins import task_tool

        extra_tools.append(task_tool)

    # --- Clarification (always last) ---
    chain.append(ClarificationMiddleware())
    extra_tools.append(ask_clarification_tool)

    return chain, extra_tools
