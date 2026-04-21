"""Agent state definition for the LangGraph workflow."""

from typing import Annotated
import operator

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """State schema for the research agent graph.

    Attributes:
        messages: Conversation message history, managed by LangGraph's
                  add_messages reducer for automatic message accumulation.
        research_summary: The plain-text output produced by the Researcher
                          to be handed off to the Writer.
        current_agent: Tracks which agent is currently active ("researcher" or "writer").
        human_approval: Human-in-the-loop control flag. Values:
                        "pending" (awaiting review), "approved" (proceed to Writer),
                        "revise" (loop back to Researcher), "abort" (end workflow).
        revision_instructions: Free-text feedback from the user when requesting
                               a revision of the research.
        researcher_iterations: Counter for how many times the researcher has executed.
        writer_iterations: Counter for how many times the writer has executed.
        intent: The classification of the user's request (e.g., "research_paper", "quick_research", "direct_chat").
        research_notes: A continuous scratchpad of notes saved by the researcher.
    """

    messages: Annotated[list, add_messages]
    research_summary: str
    current_agent: str
    human_approval: str
    revision_instructions: str
    researcher_iterations: int
    writer_iterations: int
    guardrail_reason: str
    intent: str
    research_notes: Annotated[list[str], operator.add]
