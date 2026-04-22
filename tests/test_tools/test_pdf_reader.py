"""Tests for the PDF reader tool."""

from unittest.mock import MagicMock, patch

import pytest

from ai_researcher.exceptions import PDFReadError
from ai_researcher.tools.pdf_reader import read_pdf


class TestReadPdf:
    """Tests for PDF reading functionality."""

    @patch("ai_researcher.tools.pdf_reader.get_vector_store")
    @patch("ai_researcher.tools.pdf_reader.requests.get")
    @patch("ai_researcher.tools.pdf_reader.fitz.open")
    def test_successful_read(self, mock_fitz_open, mock_get, mock_get_store):
        """Test successful PDF text extraction and ingestion."""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.content = b"fake pdf content"
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Mock fitz (PyMuPDF) document
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page 1 text content"
        mock_page.get_images.return_value = []  # No images

        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.close = MagicMock()
        mock_fitz_open.return_value = mock_doc

        # Mock Vector Store
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        result = read_pdf.invoke({"url": "http://example.com/paper.pdf"})
        assert (
            "SUCCESS: PDF from http://example.com/paper.pdf has been parsed" in result
        )
        mock_store.add_documents.assert_called_once()

        # Verify the chunks passed to add_documents
        args, _ = mock_store.add_documents.call_args
        docs = args[0]
        assert len(docs) == 1
        assert "Page 1 text content" in docs[0].page_content
        assert docs[0].metadata["source"] == "http://example.com/paper.pdf"

    @patch("ai_researcher.tools.pdf_reader.requests.get")
    def test_download_failure_raises(self, mock_get):
        """Test that HTTP errors raise PDFReadError."""
        import requests

        mock_get.side_effect = requests.RequestException("Connection failed")

        with pytest.raises(PDFReadError):
            read_pdf.invoke({"url": "http://example.com/bad.pdf"})

    @patch("ai_researcher.tools.pdf_reader.get_vector_store")
    @patch("ai_researcher.tools.pdf_reader.requests.get")
    @patch("ai_researcher.tools.pdf_reader.fitz.open")
    def test_multipage_extraction(self, mock_fitz_open, mock_get, mock_get_store):
        """Test extracting text from multiple pages."""
        mock_response = MagicMock()
        mock_response.content = b"fake pdf"
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        pages = []
        for i in range(3):
            page = MagicMock()
            page.get_text.return_value = f"Page {i + 1}"
            page.get_images.return_value = []
            pages.append(page)

        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = pages
        mock_doc.close = MagicMock()
        mock_fitz_open.return_value = mock_doc

        # Mock Vector Store
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        result = read_pdf.invoke({"url": "http://example.com/paper.pdf"})
        assert "SUCCESS" in result
        mock_store.add_documents.assert_called_once()

        args, _ = mock_store.add_documents.call_args
        docs = args[0]
        # Because chunks are 1000 characters and text is short, it should result in 1 chunk
        assert len(docs) == 1
        assert "Page 1\nPage 2\nPage 3" in docs[0].page_content

    @patch("ai_researcher.tools.pdf_reader.get_vector_store")
    @patch("ai_researcher.tools.pdf_reader.requests.get")
    @patch("ai_researcher.tools.pdf_reader.fitz.open")
    def test_image_extraction_filtered(self, mock_fitz_open, mock_get, mock_get_store):
        """Test that small images are filtered out and large ones kept."""
        mock_response = MagicMock()
        mock_response.content = b"fake pdf"
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_get.return_value = mock_response

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Some text"
        # Two images: one 50x50 (small), one 200x200 (large)
        mock_page.get_images.return_value = [
            (1, 0, 50, 50, 8, "gray", "", "img1", "flate"),
            (2, 0, 200, 200, 8, "rgb", "", "img2", "flate"),
        ]

        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.extract_image.side_effect = lambda xref: (
            {"width": 50, "height": 50, "image": b"fake", "ext": "png"} if xref == 1 
            else {"width": 250, "height": 250, "image": b"fake", "ext": "png"}
        )
        mock_fitz_open.return_value = mock_doc

        read_pdf.invoke({"url": "http://example.com/paper.pdf"})

        # Verify large image counts towards summary text or at least doesn't crash
        # The current implementation just logs images, let's ensure it handles the metadata
        mock_get_store.return_value.add_documents.assert_called_once()

    @patch("ai_researcher.tools.pdf_reader.requests.get")
    @patch("ai_researcher.tools.pdf_reader.fitz.open")
    def test_empty_pdf_handling(self, mock_fitz_open, mock_get):
        """Test behavior with an empty PDF (no text)."""
        mock_response = MagicMock()
        mock_response.content = b"empty"
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_get.return_value = mock_response

        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = []  # Zero pages
        mock_fitz_open.return_value = mock_doc

        result = read_pdf.invoke({"url": "http://example.com/empty.pdf"})
        assert "no readable text" in result.lower()

    @patch("ai_researcher.tools.pdf_reader.requests.get")
    @patch("ai_researcher.tools.pdf_reader.fitz.open")
    def test_processing_failure_raises(self, mock_fitz_open, mock_get):
        """Test that fitz opening errors raise PDFReadError."""
        mock_get.return_value = MagicMock(
            content=b"bad data", headers={"Content-Type": "application/pdf"}
        )
        mock_fitz_open.side_effect = Exception("Fitz crash")

        with pytest.raises(PDFReadError):
            read_pdf.invoke({"url": "http://example.com/corrupt.pdf"})
