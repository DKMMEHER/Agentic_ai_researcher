"""Google Scholar search tool via Serper API."""

import json
import requests

from langchain_core.tools import tool

from ai_researcher.config import get_settings
from ai_researcher.exceptions import ToolError
from ai_researcher.logging import get_logger

logger = get_logger(__name__)

@tool
def google_scholar_search(query: str, max_results: int = 5) -> str:
    """Searches Google Scholar for academic papers, books, and patents using the Serper API.
    Use this to find top-quality academic sources across all disciplines when arXiv or PubMed are insufficient.
    
    Args:
        query: The academic search query string.
        max_results: Max number of papers to retrieve (default 5).
    """
    settings = get_settings()
    api_key = settings.serper_api_key
    
    if not api_key:
        logger.warning("SERPER_API_KEY is not configured.")
        return "API Error: Google Scholar (Serper) API key is missing. Please set SERPER_API_KEY in the environment."
        
    logger.info("Searching Google Scholar for: '%s'", query)
    
    url = "https://google.serper.dev/scholar"
    
    payload = json.dumps({
        "q": query,
        "num": max_results
    })
    
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("organic", [])
        
        if not results:
            logger.warning("No Google Scholar results found for query: '%s'", query)
            return "No results found on Google Scholar."
            
        # Format the output clearly for the LLM
        formatted_results = []
        for i, item in enumerate(results, 1):
            title = item.get("title", "No Title")
            link = item.get("link", "No Link")
            snippet = item.get("snippet", "No abstract available.")
            publication_info = item.get("publicationInfo", "Unknown publication")
            
            # Format nicely
            entry = f"[{i}] {title}\n"
            entry += f"Publication Info: {publication_info}\n"
            entry += f"Abstract/Snippet: {snippet}\n"
            entry += f"Link: {link}"
            formatted_results.append(entry)
            
        logger.info("Retrieved %d results from Google Scholar.", len(results))
        return "\n\n".join(formatted_results)
        
    except requests.RequestException as e:
        logger.exception("Google Scholar HTTP request failed for query: '%s'", query)
        return f"API Error: Google Scholar request failed ({e!s}). Please rely on arXiv or PubMed."
    except Exception as e:
        logger.exception("Google Scholar processing failed for query: '%s'", query)
        return f"Processing Error: Could not parse Google Scholar data ({e!s})."
