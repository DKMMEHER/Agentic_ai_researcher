"""Tests for the PubMed search tool."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_researcher.tools.pubmed import PubMedSearchError, pubmed_search


@pytest.fixture
def sample_pubmed_search_json():
    """Sample PubMed ESearch JSON response."""
    return {"esearchresult": {"idlist": ["12345", "67890"]}}


@pytest.fixture
def sample_pubmed_fetch_xml():
    """Sample PubMed EFetch XML response."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Deep Learning in Biomedicine</ArticleTitle>
        <Journal>
          <Title>Nature Medicine</Title>
        </Journal>
        <Abstract>
          <AbstractText>Comprehensive review of deep learning.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Doe</LastName>
            <ForeName>John</ForeName>
          </Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


class TestPubMedSearch:
    """Tests for PubMed search integrated tool functionality."""

    @patch("ai_researcher.tools.pubmed.requests.get")
    def test_successful_search(
        self, mock_get, sample_pubmed_search_json, sample_pubmed_fetch_xml
    ):
        """Test a complete successful PubMed search and fetch cycle."""
        # Setup mocks for two consecutive requests.get calls
        mock_search_resp = MagicMock()
        mock_search_resp.ok = True
        mock_search_resp.json.return_value = sample_pubmed_search_json
        mock_search_resp.raise_for_status = MagicMock()

        mock_fetch_resp = MagicMock()
        mock_fetch_resp.ok = True
        mock_fetch_resp.content = sample_pubmed_fetch_xml.encode("utf-8")
        mock_fetch_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_search_resp, mock_fetch_resp]

        result_str = pubmed_search.invoke({"query": "deep learning"})
        results = json.loads(result_str)

        assert len(results) == 1
        assert results[0]["title"] == "Deep Learning in Biomedicine"
        assert results[0]["journal"] == "Nature Medicine"
        assert "John Doe" in results[0]["authors"]
        assert "12345" in results[0]["url"]

    @patch("ai_researcher.tools.pubmed.requests.get")
    def test_no_results_found(self, mock_get):
        """Test behavior when PubMed returns no IDs."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = mock_resp

        result = pubmed_search.invoke({"query": "nonexistent query"})
        assert result == "No results found on PubMed."

    @patch("ai_researcher.tools.pubmed.requests.get")
    def test_http_error_raises(self, mock_get):
        """Test that HTTP failures raise PubMedSearchError."""
        import requests

        mock_get.side_effect = requests.RequestException("Network Failure")

        with pytest.raises(PubMedSearchError) as excinfo:
            pubmed_search.invoke({"query": "test"})
        assert "HTTP Error" in str(excinfo.value)
