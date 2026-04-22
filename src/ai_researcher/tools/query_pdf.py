"""Vector database querying tool."""

from langchain_core.tools import tool

from ai_researcher.exceptions import ToolError
from ai_researcher.logging import get_logger
from ai_researcher.tools.db import get_vector_store

logger = get_logger(__name__)


class QueryPDFError(ToolError):
    def __init__(self, message: str = "", query: str = "", *args, **kwargs):
        self.query = query
        super().__init__(message, *args, **kwargs)


@tool
def query_pdf(url: str, search_query: str, max_results: int = 4) -> str:
    """Semantically searches an ingested PDF for specific information.
    The PDF MUST be ingested using the `read_pdf` tool first!

    Args:
        url: The exact URL of the PDF you are querying (used as a strict filter).
        search_query: The natural language question (e.g. "What is the training dataset?").
        max_results: Max paragraphs to return (default 4).
    """
    logger.info("Querying Vector DB for '%s' restricted to PDF %s", search_query, url)

    try:
        vector_store = get_vector_store()

        # Search the database natively, filtering strictly to chunks from THIS url
        results = vector_store.similarity_search(
            search_query, k=max_results, filter={"source": url}
        )

        if not results:
            return "No results found for your query. Make sure you used EXACTLY the same URL provided to read_pdf."

        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append(f"--- Search Result {i} ---\n{doc.page_content}")

        return "\n\n".join(formatted_results)

    except Exception as e:
        logger.exception("Failed to query vector database.")
        raise QueryPDFError(f"Vector search failed: {e!s}", query=search_query) from e
