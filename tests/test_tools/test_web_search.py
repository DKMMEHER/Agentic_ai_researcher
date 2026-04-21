"""Tests for the web search tools."""

import json
import pytest
from unittest.mock import patch, MagicMock
from ai_researcher.tools.web_search import duckduckgo_search, tavily_search, WebSearchError

class TestDuckDuckGoSearch:
    """Tests for DuckDuckGo search tool."""

    @patch("ai_researcher.tools.web_search.DDGS", create=True)
    def test_successful_search(self, mock_ddgs_class):
        """Test a normal successful DuckDuckGo search."""
        # Setup mock context manager
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

        # Mock text search results
        mock_results = [
            {"title": "Result 1", "href": "url1", "body": "Snippet 1"},
            {"title": "Result 2", "href": "url2", "body": "Snippet 2"}
        ]
        mock_ddgs.text.return_value = mock_results

        result_str = duckduckgo_search.invoke({"query": "langgraph"})
        results = json.loads(result_str)

        assert len(results) == 2
        assert results[0]["title"] == "Result 1"
        assert results[1]["body"] == "Snippet 2"

    @patch("ai_researcher.tools.web_search.DDGS", create=True)
    def test_no_results_found(self, mock_ddgs_class):
        """Test behavior when DuckDuckGo returns no results."""
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs
        mock_ddgs.text.return_value = []

        result = duckduckgo_search.invoke({"query": "nonexistent"})
        assert result == "No results found."

    @patch("ai_researcher.tools.web_search.DDGS", create=True)
    def test_error_raises_websearcherror(self, mock_ddgs_class):
        """Test that errors in DDGS raise WebSearchError."""
        mock_ddgs_class.side_effect = Exception("Service Unavailable")

        with pytest.raises(WebSearchError) as excinfo:
            duckduckgo_search.invoke({"query": "test"})
        assert "DuckDuckGo search failed" in str(excinfo.value)


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
