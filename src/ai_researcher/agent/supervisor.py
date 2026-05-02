"""Supervisor agent node for dynamic intent routing."""

from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from ai_researcher.agent.prompts import load_prompt
from ai_researcher.agent.state import AgentState
from ai_researcher.config import get_settings
from ai_researcher.logging import get_logger

logger = get_logger(__name__)


class SupervisorOutput(BaseModel):
    """Structured output for user intent classification."""

    intent: Literal["research_paper", "quick_research", "direct_chat"] = Field(
        description="The classified intent of the user request."
    )
    chat_response: str = Field(
        default="",
        description="A direct response to the user. ONLY fill this if intent is 'direct_chat'.",
    )


def _call_supervisor(state: AgentState) -> dict:
    """Classifies user intent and routes the conversation appropriately.

    This node acts as the traffic controller at the start of the graph.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq

    settings = get_settings()

    # Use a fast model for classification (Gemini Flash is preferred)
    if settings.model_name.startswith("gemini"):
        model = ChatGoogleGenerativeAI(
            model=settings.model_name,  # or force "gemini-1.5-flash"
            google_api_key=settings.gemini_api_key,
            temperature=0,  # Deterministic for routing
        )
    else:
        model = ChatGroq(  # type: ignore
            model=settings.model_name,
            api_key=settings.groq_api_key,  # type: ignore
            temperature=0,
        )

    # Bind structured output with include_raw to preserve token telemetry
    structured_model = model.with_structured_output(SupervisorOutput, include_raw=True)

    # Load supervisor system prompt
    sys_prompt = load_prompt("supervisor")
    messages = [SystemMessage(content=sys_prompt)] + state["messages"]

    logger.info(
        "Supervisor classifying query... (state has %d messages, total %d with system prompt)",
        len(state["messages"]),
        len(messages),
    )
    usage = None
    prediction = None
    try:
        response = structured_model.invoke(messages)
        prediction = response.get("parsed")  # type: ignore
        raw_msg = response.get("raw")  # type: ignore

        if prediction:
            intent = prediction.intent
        else:
            logger.warning(
                "Supervisor parsed prediction is None. Defaulting to 'research_paper'."
            )
            intent = "research_paper"

        if getattr(raw_msg, "usage_metadata", None):
            usage = raw_msg.usage_metadata  # type: ignore
    except Exception as e:
        logger.error(
            "Supervisor classification failed: %s. Defaulting to 'research_paper'.", e
        )
        intent = "research_paper"

    logger.info("Supervisor decided intent: '%s'", intent)

    # Update the state with the determined intent
    update = {"intent": intent, "current_agent": "supervisor"}

    # Use the pre-generated chat response if available (Single-call architecture)
    if intent == "direct_chat":
        content = (
            prediction.chat_response or "How can I help you with your research today?"  # type: ignore
        )

        # Reconstruct the AIMessage explicitly passing the token usage
        new_msg = AIMessage(content=content)
        if usage:
            new_msg.usage_metadata = usage

        update["messages"] = [new_msg]

    return update
