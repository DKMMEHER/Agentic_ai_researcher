"""Tests for the PDF querying tool."""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from ai_researcher.tools.query_pdf import query_pdf, QueryPDFError

class TestQueryPdf:
    """Tests for PDF semantic search tool."""

    @patch("ai_researcher.tools.query_pdf.get_vector_store")
    def test_successful_query(self, mock_get_store):
        """Test a normal successful query returning multiple documents."""
        # Setup mock vector store
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        # Mock similarity search results
        mock_docs = [
            Document(page_content="Model A is 10 layers deep.", metadata={"source": "url1"}),
            Document(page_content="Model A uses Adam optimizer.", metadata={"source": "url1"})
        ]
        mock_store.similarity_search.return_value = mock_docs

        result = query_pdf.invoke({
            "url": "url1",
            "search_query": "how deep is model A?"
        })

        assert "--- Search Result 1 ---" in result
        assert "10 layers deep" in result
        assert "Adam optimizer" in result
        mock_store.similarity_search.assert_called_once_with(
            "how deep is model A?",
            k=4,
            filter={"source": "url1"}
        )

    @patch("ai_researcher.tools.query_pdf.get_vector_store")
    def test_no_results_returned(self, mock_get_store):
        """Test behavior when the vector store returns an empty list."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.similarity_search.return_value = []

        result = query_pdf.invoke({
            "url": "url1",
            "search_query": "random query"
        })

        assert "No results found" in result

    @patch("ai_researcher.tools.query_pdf.get_vector_store")
    def test_generic_failure_raises(self, mock_get_store):
        """Test that errors in the vector store raise QueryPDFError."""
        mock_get_store.side_effect = Exception("DB Connection Lost")

        with pytest.raises(QueryPDFError) as excinfo:
            query_pdf.invoke({"url": "url1", "search_query": "fail"})
        assert "Vector search failed" in str(excinfo.value)
