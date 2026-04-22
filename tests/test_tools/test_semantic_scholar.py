"""Tests for the Semantic Scholar search tool."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_researcher.tools.semantic_scholar import (
    SemanticScholarError,
    semantic_scholar_citations,
    semantic_scholar_references,
    semantic_scholar_search,
)


@pytest.fixture
def sample_semantic_scholar_json():
    """Sample Semantic Scholar API JSON response."""
    return {
        "total": 1,
        "offset": 0,
        "data": [
            {
                "paperId": "abcdef",
                "url": "https://www.semanticscholar.org/paper/abcdef",
                "title": "Scaling Laws for Neural Language Models",
                "abstract": "We study the dependence of language model performance on model parameter count...",
                "year": 2020,
                "citationCount": 1500,
                "influentialCitationCount": 300,
                "authors": [
                    {"authorId": "1", "name": "Jared Kaplan"},
                    {"authorId": "2", "name": "Sam McCandlish"},
                ],
            }
        ],
    }


@pytest.fixture
def sample_citations_json():
    """Sample JSON for citations/references response."""
    return {
        "total": 1,
        "data": [
            {
                "citingPaper": {
                    "paperId": "cite123",
                    "title": "A newer paper",
                    "year": 2023,
                }
            }
        ],
    }


@pytest.fixture
def sample_references_json():
    """Sample JSON for references response."""
    return {
        "total": 1,
        "data": [
            {
                "citedPaper": {
                    "paperId": "ref123",
                    "title": "An older paper",
                    "year": 2010,
                }
            }
        ],
    }


class TestSemanticScholarSearch:
    """Tests for Semantic Scholar search tool."""

    @patch("ai_researcher.tools.semantic_scholar.requests.get")
    def test_successful_search(self, mock_get, sample_semantic_scholar_json):
        """Test a normal successful search returning multiple results."""
        # Setup mock response
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = sample_semantic_scholar_json
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result_str = semantic_scholar_search.invoke({"query": "scaling laws"})
        results = json.loads(result_str)

        assert len(results) == 1
        assert results[0]["title"] == "Scaling Laws for Neural Language Models"
        assert results[0]["citationCount"] == 1500
        assert results[0]["authors"][0]["name"] == "Jared Kaplan"
        assert mock_get.called

    @patch("ai_researcher.tools.semantic_scholar.requests.get")
    def test_no_results_found(self, mock_get):
        """Test behavior when Semantic Scholar returns no results."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"total": 0, "data": []}
        mock_get.return_value = mock_resp

        result = semantic_scholar_search.invoke({"query": "nonexistent query"})
        assert result == "No results found on Semantic Scholar."

    @patch("ai_researcher.tools.semantic_scholar.requests.get")
    def test_http_error_raises(self, mock_get):
        """Test that HTTP failures raise SemanticScholarError."""
        import requests

        mock_get.side_effect = requests.RequestException("API Offline")

        with pytest.raises(SemanticScholarError) as excinfo:
            semantic_scholar_search.invoke({"query": "test"})
        assert "HTTP Error" in str(excinfo.value)


class TestSemanticScholarCitations:
    """Tests for semantic_scholar_citations tool."""

    @patch("ai_researcher.tools.semantic_scholar.requests.get")
    def test_success(self, mock_get, sample_citations_json):
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_citations_json
        mock_get.return_value = mock_resp

        result = semantic_scholar_citations.invoke({"paper_id": "test-id"})
        data = json.loads(result)
        assert data[0]["title"] == "A newer paper"

    @patch("ai_researcher.tools.semantic_scholar.requests.get")
    def test_no_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        result = semantic_scholar_citations.invoke({"paper_id": "test-id"})
        assert "No citations found" in result

    @patch("ai_researcher.tools.semantic_scholar.requests.get")
    def test_404_error(self, mock_get):
        import requests

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.side_effect = requests.RequestException(response=mock_resp)

        result = semantic_scholar_citations.invoke({"paper_id": "missing-id"})
        assert "not found" in result.lower()


class TestSemanticScholarReferences:
    """Tests for semantic_scholar_references tool."""

    @patch("ai_researcher.tools.semantic_scholar.requests.get")
    def test_success(self, mock_get, sample_references_json):
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_references_json
        mock_get.return_value = mock_resp

        result = semantic_scholar_references.invoke({"paper_id": "test-id"})
        data = json.loads(result)
        assert data[0]["title"] == "An older paper"

    @patch("ai_researcher.tools.semantic_scholar.requests.get")
    def test_no_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        result = semantic_scholar_references.invoke({"paper_id": "test-id"})
        assert "No references found" in result

    @patch("ai_researcher.tools.semantic_scholar.requests.get")
    def test_generic_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.RequestException("Timeout")

        result = semantic_scholar_references.invoke({"paper_id": "test-id"})
        assert "API Error" in result
