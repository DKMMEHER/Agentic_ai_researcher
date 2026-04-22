"""API request and response schemas for the FastAPI backend."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Request to start or continue a research thread."""

    question: str = Field(
        description="The user's research query or follow-up question."
    )
    thread_id: str | None = Field(
        default=None, description="Unique ID for the conversation thread."
    )


class ActionRequest(BaseModel):
    """Request to perform a HITL action (approve/revise/abort)."""

    thread_id: str = Field(description="Unique ID for the conversation thread.")
    action: Literal["approved", "revise", "abort"] = Field(
        description="The human review decision."
    )
    instructions: str | None = Field(
        default=None, description="Revision instructions if action is 'revise'."
    )


class StreamEvent(BaseModel):
    """A single event emitted over the SSE stream."""

    event_type: Literal["token", "status", "value", "error", "done"] = Field(
        description="Type of stream event."
    )
    payload: Any = Field(description="The data associated with the event.")
    node: str | None = Field(
        default=None, description="The graph node that emitted this event."
    )
