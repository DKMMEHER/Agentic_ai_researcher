"""Web search capabilities for the AI Researcher."""

import json

from duckduckgo_search import DDGS
from langchain_core.tools import tool
from tavily import TavilyClient  # type: ignore

from ai_researcher.config import get_settings
from ai_researcher.exceptions import WebSearchError
from ai_researcher.logging import get_logger

logger = get_logger(__name__)


@tool
def duckduckgo_search(query: str) -> str:
    """Searches the web via DuckDuckGo for general knowledge and recent events.
    Use this to look up definitions, trending topics, or general context.

    Args:
        query: The search query string.
    """
    logger.info("Performing DuckDuckGo search for: '%s'", query)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            logger.warning("No DuckDuckGo results found for query: '%s'", query)
            return "No results found."

        logger.info("Found %d results on DuckDuckGo.", len(results))
        return json.dumps(results, indent=2)

    except Exception as e:
        logger.exception("DuckDuckGo search failed for query: '%s'", query)
        raise WebSearchError(
            message=f"DuckDuckGo search failed: {e!s}", query=query, engine="duckduckgo"
        ) from e


@tool
def tavily_search(query: str) -> str:
    """Searches the web via Tavily for deep, AI-optimized research information.
    Use this when you need highly accurate, comprehensive answers from the internet.

    Args:
        query: The search query string.
    """
    logger.info("Performing Tavily search for: '%s'", query)

    settings = get_settings()
    if not settings.tavily_api_key:
        error_msg = (
            "TAVILY_API_KEY is missing from the environment. "
            "Please explicitly tell the user that they must provide a TAVILY_API_KEY in their .env file to use this tool."
        )
        logger.error(error_msg)
        return error_msg

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        # We use a relatively small search to conserve tokens and API credits
        response = client.search(query, search_depth="basic", max_results=5)

        logger.info("Found %d results on Tavily.", len(response.get("results", [])))
        return json.dumps(response.get("results", []), indent=2)

    except Exception as e:
        logger.exception("Tavily search failed for query: '%s'", query)
        raise WebSearchError(
            message=f"Tavily search failed: {e!s}", query=query, engine="tavily"
        ) from e
