"""Guardrails and safety limits for the AI agent workflow."""

from ai_researcher.logging import get_logger

logger = get_logger(__name__)

# Maximum number of times agents are allowed to iterate (call tools) 
# before being forcefully halted.
MAX_RESEARCHER_ITERATIONS = 4
MAX_WRITER_ITERATIONS = 4


def log_iteration_limit_reached(agent_name: str, current_iterations: int) -> None:
    """Log a critical warning when an agent hits its iteration limit.
    
    Args:
        agent_name: Name of the agent ("researcher" or "writer")
        current_iterations: The integer count of iterations completed
    """
    limit = MAX_RESEARCHER_ITERATIONS if agent_name == "researcher" else MAX_WRITER_ITERATIONS
    logger.warning(
        "GUARDRAIL TRIGGERED: %s hit iteration limit (%d/%d). "
        "Forcing route away from tools to prevent infinite loop.",
        agent_name.capitalize(),
        current_iterations,
        limit,
    )
