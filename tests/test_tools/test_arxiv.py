"""Tests for the arXiv search tool."""

import pytest
from unittest.mock import patch, MagicMock

from ai_researcher.tools.arxiv import (
    _sanitize_query,
    _parse_arxiv_xml,
    _search_arxiv_papers,
)
from ai_researcher.exceptions import ArxivSearchError
from ai_researcher.models.schemas import SearchResult


class TestSanitizeQuery:
    """Tests for query sanitization."""

    def test_basic_topic(self):
        result = _sanitize_query("machine learning")
        assert result == "machine+learning"

    def test_removes_parentheses(self):
        result = _sanitize_query("transformer (attention)")
        assert "(" not in result
        assert ")" not in result

    def test_removes_quotes(self):
        result = _sanitize_query('"deep learning"')
        assert '"' not in result

    def test_strips_whitespace(self):
        result = _sanitize_query("  neural networks  ")
        assert result == "neural+networks"

    def test_empty_query_raises(self):
        with pytest.raises(ArxivSearchError):
            _sanitize_query("")

    def test_only_special_chars_raises(self):
        with pytest.raises(ArxivSearchError):
            _sanitize_query('()"\'')

    def test_lowercases(self):
        result = _sanitize_query("QUANTUM Computing")
        assert result == "quantum+computing"


class TestParseArxivXml:
    """Tests for XML parsing."""

    def test_parses_entries(self, sample_arxiv_xml):
        entries = _parse_arxiv_xml(sample_arxiv_xml)
        assert len(entries) == 2

    def test_extracts_title(self, sample_arxiv_xml):
        entries = _parse_arxiv_xml(sample_arxiv_xml)
        assert entries[0]["title"] == "Attention Is All You Need"

    def test_extracts_authors(self, sample_arxiv_xml):
        entries = _parse_arxiv_xml(sample_arxiv_xml)
        assert "Ashish Vaswani" in entries[0]["authors"]
        assert "Noam Shazeer" in entries[0]["authors"]

    def test_extracts_categories(self, sample_arxiv_xml):
        entries = _parse_arxiv_xml(sample_arxiv_xml)
        assert "cs.CL" in entries[0]["categories"]
        assert "cs.LG" in entries[0]["categories"]

    def test_extracts_pdf_link(self, sample_arxiv_xml):
        entries = _parse_arxiv_xml(sample_arxiv_xml)
        assert entries[0]["pdf"] == "http://arxiv.org/pdf/1706.03762v7"

    def test_empty_feed(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
        entries = _parse_arxiv_xml(xml)
        assert entries == []


class TestSearchArxivPapers:
    """Tests for the arXiv search function."""

    @patch("ai_researcher.tools.arxiv.requests.get")
    def test_successful_search(self, mock_get, sample_arxiv_xml):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_arxiv_xml
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        entries = _search_arxiv_papers("attention mechanisms", max_results=5)
        assert len(entries) == 2
        mock_get.assert_called_once()

    @patch("ai_researcher.tools.arxiv.requests.get")
    def test_failed_request_raises(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("Network error")

        with pytest.raises(ArxivSearchError):
            _search_arxiv_papers("test topic")


class TestSearchResult:
    """Tests for the SearchResult model."""

    def test_from_entries(self, sample_paper_entries):
        result = SearchResult.from_entries(
            query="test", entries=sample_paper_entries
        )
        assert result.total_results == 2
        assert result.query == "test"
        assert len(result.papers) == 2
        assert result.papers[0].title == "Attention Is All You Need"

    def test_model_dump(self, sample_paper_entries):
        result = SearchResult.from_entries(
            query="test", entries=sample_paper_entries
        )
        data = result.model_dump()
        assert isinstance(data, dict)
        assert "papers" in data
        assert "query" in data
