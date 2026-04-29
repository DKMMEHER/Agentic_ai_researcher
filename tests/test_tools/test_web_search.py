"""Tests for the web search tools."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_researcher.tools.web_search import (
    WebSearchError,
    tavily_search,
)


class TestTavilySearch:
    """Tests for Tavily search tool."""

    @patch("ai_researcher.tools.web_search.get_settings")
    @patch("ai_researcher.tools.web_search.TavilyClient", create=True)
    def test_successful_search(self, mock_tavily_class, mock_get_settings):
        """Test a normal successful Tavily search."""
        # Mock settings to provide an API key
        mock_settings = MagicMock()
        mock_settings.tavily_api_key = "fake-tavily-key"
        mock_get_settings.return_value = mock_settings

        # Mock Tavily client
        mock_client = MagicMock()
        mock_tavily_class.return_value = mock_client

        # Mock search response
        mock_response = {
            "results": [
                {"title": "AI Research", "url": "url1", "content": "Detailed analysis"}
            ]
        }
        mock_client.search.return_value = mock_response

        result_str = tavily_search.invoke({"query": "ai trends"})
        results = json.loads(result_str)

        assert len(results) == 1
        assert results[0]["title"] == "AI Research"
        mock_client.search.assert_called_once()

    @patch("ai_researcher.tools.web_search.get_settings")
    def test_missing_api_key(self, mock_get_settings):
        """Test behavior when Tavily API key is missing."""
        mock_settings = MagicMock()
        mock_settings.tavily_api_key = None
        mock_get_settings.return_value = mock_settings

        result = tavily_search.invoke({"query": "test"})
        assert "TAVILY_API_KEY is missing" in result

    @patch("ai_researcher.tools.web_search.get_settings")
    @patch("ai_researcher.tools.web_search.TavilyClient", create=True)
    def test_error_raises_websearcherror(self, mock_tavily_class, mock_get_settings):
        """Test that Tavily client errors raise WebSearchError."""
        mock_settings = MagicMock()
        mock_settings.tavily_api_key = "key"
        mock_get_settings.return_value = mock_settings

        mock_tavily_class.side_effect = Exception("Quota Exceeded")

        with pytest.raises(WebSearchError) as excinfo:
            tavily_search.invoke({"query": "test"})
        assert "Tavily search failed" in str(excinfo.value)
