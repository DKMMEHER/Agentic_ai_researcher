"""LangGraph workflow builder for the research agent.

Provides a `build_graph()` factory function that constructs the complete
multi-agent sequence graph with tools, models, state management, and checkpointing.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from ai_researcher.agent.guardrails import (
    MAX_RESEARCHER_ITERATIONS,
    MAX_WRITER_ITERATIONS,
    log_iteration_limit_reached,
)
from ai_researcher.agent.prompts import load_prompt
from ai_researcher.agent.state import AgentState
from ai_researcher.agent.supervisor import _call_supervisor
from ai_researcher.config import Settings, get_settings
from ai_researcher.logging import get_logger
from ai_researcher.tools import get_researcher_tools, get_writer_tools

logger = get_logger(__name__)


def _create_models(settings: Settings):
    """Create and configure the LLM with tools bound for both agents.

    Args:
        settings: Application settings.

    Returns:
        Tuple of (researcher_model, writer_model) with tools bound exactly once.
    """
    researcher_tools = get_researcher_tools()
    writer_tools = get_writer_tools()

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq

    if settings.model_name.startswith("gemini"):
        base_model = ChatGoogleGenerativeAI(
            model=settings.model_name,
            google_api_key=settings.gemini_api_key,
            temperature=settings.model_temperature,
        )
    else:
        base_model = ChatGroq(  # type: ignore
            model=settings.model_name,
            api_key=settings.groq_api_key,  # type: ignore
            temperature=settings.model_temperature,
        )

    researcher_model = base_model.bind_tools(researcher_tools)
    writer_model = base_model.bind_tools(writer_tools)

    logger.info(
        "Created models using %s. Researcher has %d tools, Writer has %d tools.",
        settings.model_name,
        len(researcher_tools),
        len(writer_tools),
    )
    return researcher_model, writer_model


def _filter_system_messages(messages: list) -> list:
    """Helper to remove any lingering system messages from history."""
    filtered = []
    for m in messages:
        if isinstance(m, dict):
            if m.get("role") == "system":
                continue
        elif getattr(m, "type", "") == "system":
            continue
        filtered.append(m)
    return filtered


def _call_researcher(state: AgentState, config: RunnableConfig) -> dict:
    """Node function: invoke the Researcher LLM."""
    settings = get_settings()
    model, _ = _create_models(settings)
    sys_msg = SystemMessage(content=load_prompt("researcher"))

    filtered_messages = _filter_system_messages(state.get("messages", []))
    response = model.invoke([sys_msg, *filtered_messages])

    iters = state.get("researcher_iterations", 0) + 1
    return {
        "messages": [response],
        "current_agent": "researcher",
        "researcher_iterations": iters,
    }


def _call_writer(state: AgentState, config: RunnableConfig) -> dict:
    """Node function: invoke the Writer LLM."""
    settings = get_settings()
    _, model = _create_models(settings)
    sys_msg = SystemMessage(content=load_prompt("writer"))

    filtered_messages = _filter_system_messages(state.get("messages", []))
    response = model.invoke([sys_msg, *filtered_messages])

    iters = state.get("writer_iterations", 0) + 1
    return {
        "messages": [response],
        "current_agent": "writer",
        "writer_iterations": iters,
    }


def _should_continue_supervisor(state: AgentState) -> Literal["researcher", "__end__"]:
    """Edge function: decide whether supervisor routes to researcher or ends."""
    intent = state.get("intent", "research_paper")
    if intent == "direct_chat":
        return END  # type: ignore
    return "researcher"


def _should_continue_researcher(
    state: AgentState,
) -> Literal["researcher_tools", "guardrail_handler", "human_review", "__end__"]:
    """Edge function: decide whether researcher routes to tools or to human review."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and getattr(
        last_message, "tool_calls", None
    ):
        if state.get("researcher_iterations", 0) >= MAX_RESEARCHER_ITERATIONS:
            log_iteration_limit_reached(
                "researcher", state.get("researcher_iterations", 0)
            )
            return "guardrail_handler"
        return "researcher_tools"

    if state.get("intent") == "quick_research":
        return END  # type: ignore

    return "human_review"


def _human_review(state: AgentState) -> dict:
    """Node function: process human approval decision and route accordingly.

    This node runs after the graph resumes from the human review interrupt.
    The caller must set 'human_approval' in state via graph.update_state()
    before resuming the graph.
    """
    approval = state.get("human_approval", "pending")
    logger.info("Human review node — approval=%s", approval)

    if approval == "revise":
        instructions = state.get("revision_instructions", "Please do more research.")
        revision_msg = HumanMessage(
            content=f"[REVISION REQUEST from user]: {instructions}"
        )
        return {
            "messages": [revision_msg],
            "current_agent": "researcher",
            "human_approval": "pending",
            "researcher_iterations": 0,  # Reset loop allowance for revision
        }
    elif approval == "abort":
        return {"current_agent": "done"}

    # Default: approved — proceed to writer
    return {"current_agent": "writer"}


def _route_after_review(
    state: AgentState,
) -> Literal["writer", "researcher", "__end__"]:
    """Edge function: route based on human review decision."""
    current = state.get("current_agent", "writer")
    if current == "researcher":
        return "researcher"
    elif current == "done":
        return END  # type: ignore
    return "writer"


def _should_continue_writer(
    state: AgentState,
) -> Literal["writer_tools", "guardrail_handler", "__end__"]:
    """Edge function: decide whether writer routes to tools or ends."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and getattr(
        last_message, "tool_calls", None
    ):
        if state.get("writer_iterations", 0) >= MAX_WRITER_ITERATIONS:
            log_iteration_limit_reached("writer", state.get("writer_iterations", 0))
            return "guardrail_handler"
        return "writer_tools"

    return END  # type: ignore


def _guardrail_handler(state: AgentState) -> dict:
    """Node function: injects a visible system message when guardrails are triggered."""
    agent = state.get("current_agent", "unknown")
    iters = state.get(f"{agent}_iterations", 0)

    # Check if there was a technical error in the last message
    last_msg = state["messages"][-1]
    error_context = ""
    if hasattr(last_msg, "content") and "ERROR:" in str(last_msg.content):
        error_context = f"\n\nLast reported error:\n{last_msg.content}"

    limit = (
        MAX_RESEARCHER_ITERATIONS if agent == "researcher" else MAX_WRITER_ITERATIONS
    )

    msg = SystemMessage(
        content=(
            f"🛡️ **GUARDRAIL TRIGGERED**: The {agent.capitalize()} reached its "
            f"safety iteration limit ({iters}/{limit}).\n"
            f"To prevent an infinite loop and excessive token usage, the process has been halted. "
            f"{error_context}\n\nPlease review the output so far or try a different approach."
        )
    )
    return {"messages": [msg], "current_agent": "done"}


def build_graph(
    settings: Settings | None = None,
    thread_id: str | None = None,
    checkpointer=None,
) -> tuple[CompiledStateGraph, dict]:
    """Build and compile the multi-agent sequential graph.

    Args:
        settings: Application settings. Uses default if not provided.
        thread_id: Conversation thread ID. Uses settings default if not provided.
        checkpointer: An already-initialized checkpointer instance.

    Returns:
        Tuple of (compiled_graph, config_dict) ready for invocation.
    """
    if settings is None:
        settings = get_settings()

    if thread_id is None:
        thread_id = settings.thread_id

    # Create models and tools
    _researcher_model, _writer_model = _create_models(settings)

    researcher_tools_node = ToolNode(get_researcher_tools())
    writer_tools_node = ToolNode(get_writer_tools())

    # Build the sequential state graph
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", _call_supervisor)
    workflow.add_node("researcher", _call_researcher)
    workflow.add_node("researcher_tools", researcher_tools_node)
    workflow.add_node("human_review", _human_review)
    workflow.add_node("writer", _call_writer)
    workflow.add_node("writer_tools", writer_tools_node)
    workflow.add_node("guardrail_handler", _guardrail_handler)

    # Define the Dynamic Routing
    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges("supervisor", _should_continue_supervisor)

    workflow.add_conditional_edges("researcher", _should_continue_researcher)
    workflow.add_edge("researcher_tools", "researcher")

    workflow.add_conditional_edges("human_review", _route_after_review)

    workflow.add_conditional_edges("writer", _should_continue_writer)
    workflow.add_edge("writer_tools", "writer")
    workflow.add_edge("guardrail_handler", END)

    # Use provided checkpointer or fall back to the configured default (SQLite)
    if checkpointer is None:
        from ai_researcher.agent.checkpointer import get_checkpointer

        checkpointer = get_checkpointer(
            backend=settings.checkpoint_backend,  # type: ignore[arg-type]
            db_url=settings.checkpoint_db_url,  # type: ignore
        )

    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )

    # Config dict for thread-safe state management
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {},
    }

    logger.info(
        "HITL graph built successfully (thread_id=%s, interrupt_before=human_review)",
        thread_id,
    )
    return graph, config
