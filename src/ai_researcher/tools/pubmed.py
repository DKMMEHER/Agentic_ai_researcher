"""PubMed API search tool."""

import json
import urllib.parse
import xml.etree.ElementTree as ET

import requests
from langchain_core.tools import tool

from ai_researcher.exceptions import ToolError
from ai_researcher.logging import get_logger

logger = get_logger(__name__)


class PubMedSearchError(ToolError):
    def __init__(self, message: str = "", query: str = "", *args, **kwargs):
        self.query = query
        super().__init__(message, *args, **kwargs)


@tool
def pubmed_search(query: str, max_results: int = 5) -> str:
    """Searches PubMed (NCBI Entrez) for biomedical and life sciences literature.
    Returns highly reputable papers including titles, authors, journals, and abstracts.

    Args:
        query: The search query string.
        max_results: Max number of papers to return (default 5).
    """
    logger.info("Searching PubMed for: '%s'", query)

    try:
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmax={max_results}&retmode=json"

        search_resp = requests.get(search_url, timeout=10)
        search_resp.raise_for_status()
        id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])

        if not id_list:
            logger.warning("No PubMed results found for query: '%s'", query)
            return "No results found on PubMed."

        ids_str = ",".join(id_list)
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={ids_str}&retmode=xml"

        fetch_resp = requests.get(fetch_url, timeout=15)
        fetch_resp.raise_for_status()

        root = ET.fromstring(fetch_resp.content)
        papers = []

        for article in root.findall(".//PubmedArticle"):
            paper_data = {}

            title_node = article.find(".//ArticleTitle")
            if title_node is not None and title_node.text:
                paper_data["title"] = title_node.text

            abstract_texts = []
            for abstract_node in article.findall(".//AbstractText"):
                if abstract_node.text:
                    abstract_texts.append(abstract_node.text)
            if abstract_texts:
                paper_data["abstract"] = " ".join(abstract_texts)

            journal_node = article.find(".//Title")
            if journal_node is not None and journal_node.text:
                paper_data["journal"] = journal_node.text

            authors = []
            for author_node in article.findall(".//Author"):
                last_name = author_node.find("LastName")
                fore_name = author_node.find("ForeName")
                if last_name is not None and fore_name is not None:
                    authors.append(f"{fore_name.text} {last_name.text}")
            if authors:
                paper_data["authors"] = authors

            pmid_node = article.find(".//PMID")
            if pmid_node is not None and pmid_node.text:
                paper_data["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_node.text}/"

            papers.append(paper_data)

        logger.info("Found %d results perfectly parsed from PubMed.", len(papers))
        return json.dumps(papers, indent=2)

    except requests.RequestException as e:
        logger.exception("PubMed HTTP request failed for query: '%s'", query)
        raise PubMedSearchError(f"HTTP Error: {e!s}", query=query) from e
    except Exception as e:
        logger.exception("PubMed XML parsing failed for query: '%s'", query)
        raise PubMedSearchError(f"Processing Error: {e!s}", query=query) from e
