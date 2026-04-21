"""Research Note-Taking / Scratchpad interface tool."""

from langchain_core.tools import tool

from ai_researcher.logging import get_logger

logger = get_logger(__name__)

@tool
def save_research_note(note: str) -> str:
    """Save an important finding, statistic, or thought to your persistent scratchpad.
    Use this whenever you find a critical piece of information that you want to remember
    for drafting the final paper, or if you want to track your current multi-step plan.
    
    Args:
        note: A detailed string capturing the finding, fact, or thought.
    """
    logger.info("Researcher saved a note to the scratchpad: '%s...'", note[:50])
    
    # The actual appending to AgentState happens dynamically in the graph's
    # _call_researcher node by intercepting this tool call.
    return "Note successfully saved to your scratchpad. It will be injected into your memory on your next turn."
