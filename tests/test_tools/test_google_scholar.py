"""Tests for the Google Scholar search tool."""

import pytest
from unittest.mock import patch, MagicMock

from ai_researcher.tools.google_scholar import google_scholar_search


class TestGoogleScholarSearch:
    """Tests for the google_scholar_search tool."""

    @patch("ai_researcher.tools.google_scholar.requests.post")
    def test_successful_search(self, mock_post):
        """Test successful Google Scholar search with results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Attention Is All You Need",
                    "link": "https://arxiv.org/abs/1706.03762",
                    "snippet": "We propose a new architecture based on attention mechanisms.",
                    "publicationInfo": "NeurIPS 2017",
                },
                {
                    "title": "BERT: Pre-training",
                    "link": "https://arxiv.org/abs/1810.04805",
                    "snippet": "We introduce BERT.",
                    "publicationInfo": "NAACL 2019",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = google_scholar_search.invoke({"query": "transformers"})

        assert "Attention Is All You Need" in result
        assert "BERT" in result
        assert "[1]" in result
        assert "[2]" in result
        mock_post.assert_called_once()

    @patch("ai_researcher.tools.google_scholar.requests.post")
    def test_no_results(self, mock_post):
        """Test handling when no results are found."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"organic": []}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = google_scholar_search.invoke({"query": "xyznonexistent"})
        assert "No results found" in result

    @patch("ai_researcher.tools.google_scholar.requests.post")
    def test_api_error(self, mock_post):
        """Test handling of HTTP errors."""
        from requests.exceptions import RequestException
        mock_post.side_effect = RequestException("Connection refused")

        result = google_scholar_search.invoke({"query": "transformers"})
        assert "API Error" in result

    @patch("ai_researcher.tools.google_scholar.get_settings")
    def test_missing_api_key(self, mock_settings):
        """Test handling when SERPER_API_KEY is not set."""
        mock_settings.return_value = MagicMock(serper_api_key=None)

        result = google_scholar_search.invoke({"query": "transformers"})
        assert "API Error" in result or "missing" in result.lower()
