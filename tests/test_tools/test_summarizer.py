"""Tests for the Map-Reduce Summarizer tool."""

from unittest.mock import MagicMock, patch

from ai_researcher.tools.summarizer import summarize_long_document


class TestSummarizeLongDocument:
    """Tests for the summarize_long_document tool."""

    @patch("ai_researcher.tools.summarizer.ChatGroq")
    @patch("ai_researcher.tools.summarizer.ChatGoogleGenerativeAI")
    @patch("ai_researcher.tools.summarizer.get_vector_store")
    def test_successful_summarization(
        self, mock_get_store, mock_gemini_class, mock_groq_class
    ):
        """Test successful map-reduce summarization."""
        from langchain_core.documents import Document
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_groq import ChatGroq

        # Mock the vector store
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="Chunk 1", metadata={"source": "url"}),
            Document(page_content="Chunk 2", metadata={"source": "url"}),
        ]
        mock_get_store.return_value = mock_store

        # Mock LLM instances
        mock_gemini = MagicMock(spec=ChatGoogleGenerativeAI)
        mock_groq = MagicMock(spec=ChatGroq)
        
        # Configure both mocks to return the same sequence
        test_summaries = [
            "Summary 1",
            "Summary 2",
            "This is the final executive summary of the paper.",
        ]
        mock_gemini.invoke.side_effect = test_summaries
        mock_groq.invoke.side_effect = test_summaries
        
        mock_gemini_class.return_value = mock_gemini
        mock_groq_class.return_value = mock_groq

        result = summarize_long_document.invoke({"url": "url"})

        assert "EXECUTIVE SUMMARY" in result
        assert "final executive summary of the paper" in result

    @patch("ai_researcher.tools.summarizer.get_vector_store")
    def test_no_chunks_found(self, mock_get_store):
        """Test handling when no document chunks are found in ChromaDB."""
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []
        mock_get_store.return_value = mock_store

        result = summarize_long_document.invoke(
            {"url": "http://example.com/missing.pdf"}
        )

        assert "Error" in result
        assert "read_pdf" in result

    @patch("ai_researcher.tools.summarizer.get_vector_store")
    def test_exception_handling(self, mock_get_store):
        """Test graceful handling of unexpected errors."""
        mock_get_store.side_effect = Exception("ChromaDB connection failed")

        result = summarize_long_document.invoke({"url": "http://example.com/paper.pdf"})

        assert "Processing Error" in result
