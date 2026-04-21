"""Tools module — provides LangChain tools for the research agent.

Use `get_tools()` to get the list of all available tools.
"""

from langchain_core.tools import BaseTool

from ai_researcher.tools.arxiv import arxiv_search
from ai_researcher.tools.pdf_reader import read_pdf
from ai_researcher.tools.latex_renderer import render_latex_pdf
from ai_researcher.tools.web_search import tavily_search
from ai_researcher.tools.pubmed import pubmed_search
from ai_researcher.tools.query_pdf import query_pdf
from ai_researcher.tools.semantic_scholar import (
    semantic_scholar_search,
    semantic_scholar_citations,
    semantic_scholar_references,
)
from ai_researcher.tools.scratchpad import save_research_note
from ai_researcher.tools.summarizer import summarize_long_document
from ai_researcher.tools.youtube import youtube_transcript_reader
from ai_researcher.tools.google_scholar import google_scholar_search
from ai_researcher.tools.wikipedia_tool import wikipedia_search

def get_researcher_tools() -> list[BaseTool]:
    """Get all tools available to the researcher agent."""
    return [
        arxiv_search,
        read_pdf,
        tavily_search,
        pubmed_search,
        semantic_scholar_search,
        semantic_scholar_citations,
        semantic_scholar_references,
        save_research_note,
        summarize_long_document,
        youtube_transcript_reader,
        query_pdf,
        google_scholar_search,
        wikipedia_search,
    ]

def get_writer_tools() -> list[BaseTool]:
    """Get all tools available to the writer agent."""
    return [render_latex_pdf]


__all__ = ["get_researcher_tools", "get_writer_tools", "arxiv_search", "read_pdf", "render_latex_pdf", "tavily_search", "pubmed_search", "semantic_scholar_search", "youtube_transcript_reader", "query_pdf"]

