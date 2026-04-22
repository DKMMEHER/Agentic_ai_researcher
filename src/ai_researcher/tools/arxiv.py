"""arXiv paper search tool.

Searches the arXiv API for recently published papers on a given topic
and returns structured results.
"""

import re
import xml.etree.ElementTree as ET

import requests
from langchain_core.tools import tool

from ai_researcher.config import get_settings
from ai_researcher.exceptions import ArxivSearchError
from ai_researcher.logging import get_logger
from ai_researcher.models.schemas import SearchResult

logger = get_logger(__name__)

# arXiv Atom XML namespaces
_ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _sanitize_query(topic: str) -> str:
    """Sanitize and format a search query for the arXiv API.

    Args:
        topic: Raw topic string from the user.

    Returns:
        URL-safe query string.

    Raises:
        ArxivSearchError: If the topic is empty after sanitization.
    """
    # Remove special characters that break arXiv queries
    cleaned = re.sub(r'[()"\']', "", topic.strip())
    if not cleaned:
        raise ArxivSearchError(
            message="Search topic is empty after sanitization.",
            query=topic,
        )

    # Join words with '+' for arXiv API
    query = "+".join(cleaned.lower().split())
    return query


def _parse_arxiv_xml(xml_content: str) -> list[dict]:
    """Parse the XML content from an arXiv API response.

    Args:
        xml_content: Raw XML string from the arXiv API.

    Returns:
        List of parsed paper dictionaries.
    """
    entries = []
    root = ET.fromstring(xml_content)

    for entry in root.findall("atom:entry", _ARXIV_NS):
        authors = [
            author.findtext("atom:name", namespaces=_ARXIV_NS)
            for author in entry.findall("atom:author", _ARXIV_NS)
            if author.findtext("atom:name", namespaces=_ARXIV_NS)
        ]

        categories = [
            cat.attrib.get("term")
            for cat in entry.findall("atom:category", _ARXIV_NS)
            if cat.attrib.get("term")
        ]

        pdf_link = None
        for link in entry.findall("atom:link", _ARXIV_NS):
            if link.attrib.get("type") == "application/pdf":
                pdf_link = link.attrib.get("href")
                break

        title = entry.findtext("atom:title", namespaces=_ARXIV_NS) or ""
        summary = entry.findtext("atom:summary", namespaces=_ARXIV_NS) or ""

        entries.append(
            {
                "title": " ".join(title.split()),  # Normalize whitespace
                "summary": summary.strip(),
                "authors": authors,
                "categories": categories,
                "pdf": pdf_link,
            }
        )

    return entries


def _search_arxiv_papers(topic: str, max_results: int = 5) -> list[dict]:
    """Execute a search against the arXiv API.

    Args:
        topic: The search topic.
        max_results: Maximum number of results to return.

    Returns:
        List of parsed paper dictionaries.

    Raises:
        ArxivSearchError: If the API request fails.
    """
    query = _sanitize_query(topic)
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=all:{query}"
        f"&max_results={max_results}"
        "&sortBy=submittedDate"
        "&sortOrder=descending"
    )

    logger.info("Searching arXiv: %s", url)

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ArxivSearchError(
            message=f"arXiv API request failed: {e}",
            query=topic,
        ) from e

    entries = _parse_arxiv_xml(resp.text)
    logger.info("Found %d papers for query '%s'", len(entries), topic)
    return entries


@tool
def arxiv_search(topic: str) -> dict:
    """Search for recently uploaded arXiv papers on a given topic.

    Args:
        topic: The topic to search for papers about.

    Returns:
        Dictionary with search results including paper titles, authors,
        summaries, categories, and PDF links.
    """
    settings = get_settings()
    logger.info("arXiv search tool invoked for topic: %s", topic)

    entries = _search_arxiv_papers(topic, max_results=settings.max_arxiv_results)

    if not entries:
        raise ArxivSearchError(
            message=f"No papers found for topic: {topic}",
            query=topic,
        )

    result = SearchResult.from_entries(query=topic, entries=entries)
    logger.info(
        "Returning %d papers for topic '%s'",
        result.total_results,
        topic,
    )
    return result.model_dump()
