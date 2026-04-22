"""Long-Document Map-Reduce Summarizer tool."""

from langchain.chains.summarize import load_summarize_chain
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from ai_researcher.config import get_settings
from ai_researcher.logging import get_logger
from ai_researcher.tools.db import get_vector_store

logger = get_logger(__name__)


@tool
def summarize_long_document(url: str) -> str:
    """Summarizes an entire lengthy document (like a PDF textbook or long paper)
    into a comprehensive executive summary.

    IMPORTANT: You MUST have already called the `read_pdf` tool on this URL BEFORE
    using this summarizer, otherwise it will have no data to summarize.

    Args:
        url: The URL of the PDF document that was previously ingested.
    """
    settings = get_settings()
    logger.info("Initializing Map-Reduce Summarizer for URL: %s", url)

    try:
        vector_store = get_vector_store()

        # We need to manually pull the chunks out of Chroma.
        # Since Chroma in LangChain doesn't easily let us fetch just by exact metadata match
        # without a similarity query, we perform a dummy similarity search that fetches a huge amount
        # and pre-filters by source=url. We request k=1000 since it's a huge document.
        docs = vector_store.similarity_search("summary", k=1000, filter={"source": url})

        if not docs:
            logger.warning("No document chunks found for URL: '%s'", url)
            return (
                f"Error: No text found for '{url}'. "
                "Did you forget to call `read_pdf` on it first?"
            )

        logger.info(
            "Retrieved %d document chunks for Map-Reduce processing.", len(docs)
        )

        # Instantiate a raw LLM without tools for pure reasoning
        if settings.model_name.startswith("gemini"):
            llm = ChatGoogleGenerativeAI(
                model=settings.model_name,
                google_api_key=settings.gemini_api_key,
                temperature=0.3,  # Low temperature for accurate summarization
            )
        else:
            llm = ChatGroq(  # type: ignore
                model=settings.model_name,
                api_key=settings.groq_api_key,  # type: ignore
                temperature=0.3,
            )

        # Execute the Map-Reduce Chain
        logger.info("Starting map-reduce chain execution...")
        chain = load_summarize_chain(llm, chain_type="map_reduce")

        # Run the chain on all document chunks
        # This will internally map (summarize each chunk) and reduce (combine into final)
        result = chain.invoke(docs)  # type: ignore

        final_summary = result.get("output_text", "")

        if not final_summary:
            return "Error: Map-Reduce chain failed to generate a summary."

        logger.info("Map-reduce summarization completed successfully.")
        return f"--- EXECUTIVE SUMMARY OF {url} ---\n\n{final_summary}"

    except Exception as e:
        logger.exception("Map-Reduce Summarizer failed for URL: '%s'", url)
        return f"Processing Error: Could not summarize the document ({e!s})."
