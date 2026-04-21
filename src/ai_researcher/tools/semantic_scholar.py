"""Semantic Scholar API integration tool."""

import json
import urllib.parse
import requests

from langchain_core.tools import tool

from ai_researcher.exceptions import ToolError
from ai_researcher.logging import get_logger

logger = get_logger(__name__)

class SemanticScholarError(ToolError):
    def __init__(self, message: str = "", query: str = "", *args, **kwargs):
        self.query = query
        super().__init__(message, *args, **kwargs)

@tool
def semantic_scholar_search(query: str, max_results: int = 5) -> str:
    """Searches Semantic Scholar for highly cited, reputable academic papers.
    Use this to find top-quality papers and verify their citation counts and influence.
    
    Args:
        query: The academic search query string.
        max_results: Max number of papers to retrieve (default 5).
    """
    logger.info("Searching Semantic Scholar for: '%s'", query)
    
    try:
        fields = "title,authors,year,citationCount,influentialCitationCount,abstract,url"
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(query)}&limit={max_results}&fields={fields}"
        
        headers = {"Accept": "application/json"}
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("data", [])
        
        if not results:
            logger.warning("No Semantic Scholar results found for query: '%s'", query)
            return "No results found on Semantic Scholar."
            
        logger.info("Retrieved %d highly-cited results from Semantic Scholar.", len(results))
        return json.dumps(results, indent=2)
        
    except requests.RequestException as e:
        logger.exception("Semantic Scholar HTTP request failed for query: '%s'", query)
        raise SemanticScholarError(message=f"HTTP Error: {e!s}", query=query) from e
    except Exception as e:
        logger.exception("Semantic Scholar processing failed for query: '%s'", query)
        raise SemanticScholarError(message=f"Processing Error: {e!s}", query=query) from e

@tool
def semantic_scholar_citations(paper_id: str, max_results: int = 5) -> str:
    """Finds the underlying papers that CITED the given paper.
    Use this for snowball research to find newer updates and related works.
    
    Args:
        paper_id: The ID of the target paper (e.g. arXiv:1706.03762, DOI, or S2 paper ID).
        max_results: Max number of citations to retrieve (default 5, max 10 to protect memory).
    """
    logger.info("Fetching Semantic Scholar citations for: '%s'", paper_id)
    
    try:
        fields = "title,authors,year,citationCount,abstract,url"
        url = f"https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(paper_id)}/citations?limit={max_results}&fields={fields}"
        
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        results = [item.get("citingPaper") for item in data.get("data", []) if item.get("citingPaper")]
        
        if not results:
            return f"No citations found for the paper '{paper_id}'."
            
        return json.dumps(results, indent=2)
        
    except requests.RequestException as e:
        logger.exception("Semantic Scholar API failed fetching citations for: '%s'", paper_id)
        if getattr(e.response, "status_code", None) == 404:
             return f"API Error: Paper ID '{paper_id}' not found in Semantic Scholar."
        return f"API Error: Request failed ({e!s})."


@tool
def semantic_scholar_references(paper_id: str, max_results: int = 5) -> str:
    """Finds the foundational papers that were REFERENCED/CITED BY the given paper.
    Use this for snowball research to find foundational or background papers.
    
    Args:
        paper_id: The ID of the target paper (e.g. arXiv:1706.03762, DOI, or S2 paper ID).
        max_results: Max number of references to retrieve.
    """
    logger.info("Fetching Semantic Scholar references for: '%s'", paper_id)
    
    try:
        fields = "title,authors,year,citationCount,abstract,url"
        url = f"https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(paper_id)}/references?limit={max_results}&fields={fields}"
        
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        results = [item.get("citedPaper") for item in data.get("data", []) if item.get("citedPaper")]
        
        if not results:
            return f"No references found for the paper '{paper_id}'."
            
        return json.dumps(results, indent=2)
        
    except requests.RequestException as e:
        logger.exception("Semantic Scholar API failed fetching references for: '%s'", paper_id)
        if getattr(e.response, "status_code", None) == 404:
             return f"API Error: Paper ID '{paper_id}' not found in Semantic Scholar."
        return f"API Error: Request failed ({e!s})."
