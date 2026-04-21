"""Tests for the Map-Reduce Summarizer tool."""

import pytest
from unittest.mock import patch, MagicMock

from ai_researcher.tools.summarizer import summarize_long_document


class TestSummarizeLongDocument:
    """Tests for the summarize_long_document tool."""

    @patch("ai_researcher.tools.summarizer.load_summarize_chain")
    @patch("ai_researcher.tools.summarizer.ChatGoogleGenerativeAI")
    @patch("ai_researcher.tools.summarizer.get_vector_store")
    def test_successful_summarization(self, mock_get_store, mock_llm, mock_chain_loader):
        """Test successful map-reduce summarization."""
        from langchain_core.documents import Document

        # Mock the vector store to return some docs
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="This paper discusses transformers.", metadata={"source": "http://example.com/paper.pdf"}),
            Document(page_content="Self-attention is the key mechanism.", metadata={"source": "http://example.com/paper.pdf"}),
        ]
        mock_get_store.return_value = mock_store

        # Mock the chain to return a summary
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"output_text": "This is a summary of the paper."}
        mock_chain_loader.return_value = mock_chain

        result = summarize_long_document.invoke({"url": "http://example.com/paper.pdf"})

        assert "EXECUTIVE SUMMARY" in result
        assert "summary of the paper" in result

    @patch("ai_researcher.tools.summarizer.get_vector_store")
    def test_no_chunks_found(self, mock_get_store):
        """Test handling when no document chunks are found in ChromaDB."""
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []
        mock_get_store.return_value = mock_store

        result = summarize_long_document.invoke({"url": "http://example.com/missing.pdf"})

        assert "Error" in result
        assert "read_pdf" in result

    @patch("ai_researcher.tools.summarizer.get_vector_store")
    def test_exception_handling(self, mock_get_store):
        """Test graceful handling of unexpected errors."""
        mock_get_store.side_effect = Exception("ChromaDB connection failed")

        result = summarize_long_document.invoke({"url": "http://example.com/paper.pdf"})

        assert "Processing Error" in result
