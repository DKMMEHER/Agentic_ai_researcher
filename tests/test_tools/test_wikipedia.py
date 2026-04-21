"""Tests for the Wikipedia search tool."""

import pytest
from unittest.mock import patch, MagicMock
import wikipedia

from ai_researcher.tools.wikipedia_tool import wikipedia_search


class TestWikipediaSearch:
    """Tests for the wikipedia_search tool."""

    @patch("ai_researcher.tools.wikipedia_tool.wikipedia")
    def test_successful_search(self, mock_wikipedia):
        """Test successful Wikipedia search returns a summary."""
        mock_wikipedia.search.return_value = ["Transformer (machine learning)", "Transformer", "Autobot"]
        mock_wikipedia.summary.return_value = (
            "A transformer is a deep learning model architecture. "
            "It was introduced in 2017 by researchers at Google."
        )

        result = wikipedia_search.invoke({"query": "transformer machine learning"})

        assert "WIKIPEDIA SUMMARY" in result
        assert "deep learning" in result
        assert "Transformer" in result
        mock_wikipedia.search.assert_called_once()

    @patch("ai_researcher.tools.wikipedia_tool.wikipedia")
    def test_no_results(self, mock_wikipedia):
        """Test handling when Wikipedia finds nothing."""
        mock_wikipedia.search.return_value = []

        result = wikipedia_search.invoke({"query": "xyznonexistentquery123"})
        assert "No results found" in result

    def test_disambiguation_error(self):
        """Test handling of ambiguous search terms."""
        with patch("ai_researcher.tools.wikipedia_tool.wikipedia.search") as mock_search, \
             patch("ai_researcher.tools.wikipedia_tool.wikipedia.summary") as mock_summary:
            
            mock_search.return_value = ["Mercury"]
            mock_summary.side_effect = wikipedia.exceptions.DisambiguationError(
                "Mercury", ["Mercury (planet)", "Mercury (element)", "Mercury (mythology)"]
            )

            result = wikipedia_search.invoke({"query": "Mercury"})
            assert "ambiguous" in result.lower()

    def test_page_error(self):
        """Test handling when a specific page cannot be fetched."""
        with patch("ai_researcher.tools.wikipedia_tool.wikipedia.search") as mock_search, \
             patch("ai_researcher.tools.wikipedia_tool.wikipedia.summary") as mock_summary:
            
            mock_search.return_value = ["SomePageThatDoesNotExist"]
            mock_summary.side_effect = wikipedia.exceptions.PageError(
                pageid="12345"
            )

            result = wikipedia_search.invoke({"query": "nonexistent page"})
            # In wikipedia_tool.py: return f"Could not fetch the specific page for '{query}'."
            assert "Could not fetch" in result

    @patch("ai_researcher.tools.wikipedia_tool.wikipedia")
    def test_network_error(self, mock_wikipedia):
        """Test handling of network failures."""
        from requests.exceptions import RequestException
        mock_wikipedia.search.side_effect = RequestException("Connection refused")

        result = wikipedia_search.invoke({"query": "transformers"})
        assert "API Error" in result
