"""Wikipedia search tool."""

import wikipedia  # type: ignore
from langchain_core.tools import tool
from requests.exceptions import RequestException

from ai_researcher.logging import get_logger

logger = get_logger(__name__)


@tool
def wikipedia_search(query: str) -> str:
    """Searches Wikipedia for broad, general-knowledge information.
    Use this when you need background knowledge, history, or factual summaries about a topic
    before diving into specific academic papers.

    Args:
        query: The search topic.
    """
    logger.info("Searching Wikipedia for: '%s'", query)

    try:
        # First, search to get the most relevant page titles
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            logger.warning("No Wikipedia results found for: '%s'", query)
            return f"No results found on Wikipedia for '{query}'."

        # Try to pull the summary for the top result
        top_page_title = search_results[0]

        try:
            # Attempt to get the page summary (auto_suggest=False prevents it from getting confused)
            summary = wikipedia.summary(
                top_page_title, sentences=10, auto_suggest=False
            )

            result = f"--- WIKIPEDIA SUMMARY: {top_page_title} ---\n"
            result += summary + "\n\n"
            result += (
                f"Other related pages you could search: {', '.join(search_results[1:])}"
            )
            return result  # type: ignore

        except wikipedia.exceptions.DisambiguationError as e:
            # Handle ambiguous search terms (e.g. "Mercury" -> Planet, Element, God)
            logger.warning(
                "Wikipedia DisambiguationError for: '%s'. Options: %s",
                top_page_title,
                e.options[:5],
            )
            options = ", ".join(e.options[:5])
            return f"The term '{top_page_title}' is ambiguous. Did you mean one of these? {options}"

        except wikipedia.exceptions.PageError:
            logger.warning("Wikipedia PageError for: '%s'", top_page_title)
            return f"Could not fetch the specific page for '{top_page_title}'."

    except RequestException as e:
        logger.error("Wikipedia network error: %s", e)
        return "API Error: Failed to connect to Wikipedia."
    except Exception as e:
        logger.error("Wikipedia processing error: %s", e)
        return f"Processing Error: Wikipedia search failed ({e!s})."
