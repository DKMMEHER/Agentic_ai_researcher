"""Long-Document Map-Reduce Summarizer tool using LCEL."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from ai_researcher.config import get_settings
from ai_researcher.logging import get_logger
from ai_researcher.tools.db import get_vector_store

logger = get_logger(__name__)

# --- Prompts ---
MAP_PROMPT = ChatPromptTemplate.from_template(
    "Write a concise summary of the following section of a research document:\n\n"
    "{context}\n\n"
    "CONCISE SUMMARY:"
)

REDUCE_PROMPT = ChatPromptTemplate.from_template(
    "The following are summaries of different sections of a research document:\n\n"
    "{summaries}\n\n"
    "Based on these summaries, write a comprehensive executive summary that captures "
    "the main themes, key findings, and technical contributions of the entire document.\n\n"
    "EXECUTIVE SUMMARY:"
)


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
    logger.info("Initializing LCEL Map-Reduce Summarizer for URL: %s", url)

    try:
        vector_store = get_vector_store()
        # Retrieve chunks for this URL
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

        # Instantiate LLM
        if settings.model_name.startswith("gemini"):
            llm = ChatGoogleGenerativeAI(
                model=settings.model_name,
                google_api_key=settings.gemini_api_key,
                temperature=0.3,
            )
        else:
            llm = ChatGroq(  # type: ignore
                model=settings.model_name,
                api_key=settings.groq_api_key,  # type: ignore
                temperature=0.3,
            )

        # --- LCEL Map Step ---
        map_chain = MAP_PROMPT | llm | StrOutputParser()

        logger.info("Executing Map step (summarizing chunks)...")
        # Process chunks in batches to avoid rate limits or context window issues
        summaries = []
        for i, doc in enumerate(docs):
            if i % 10 == 0:
                logger.info("  Processing chunk %d/%d...", i + 1, len(docs))
            summary = map_chain.invoke({"context": doc.page_content})
            summaries.append(summary)

        # --- LCEL Reduce Step ---
        logger.info("Executing Reduce step (combining summaries)...")
        combined_summaries_text = "\n\n".join(summaries)
        reduce_chain = REDUCE_PROMPT | llm | StrOutputParser()

        final_summary = reduce_chain.invoke({"summaries": combined_summaries_text})

        if not final_summary:
            return "Error: Failed to generate a summary."

        logger.info("Map-reduce summarization completed successfully.")
        return f"--- EXECUTIVE SUMMARY OF {url} ---\n\n{final_summary}"

    except Exception as e:
        logger.exception("Map-Reduce Summarizer failed for URL: '%s'", url)
        return f"Processing Error: Could not summarize the document ({e!s})."
