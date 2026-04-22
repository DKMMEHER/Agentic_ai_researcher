"""PDF reader tool.

Downloads and extracts text content into the Chroma Vector Database,
and extracts raster images (figures/tables) into the local output directory.
"""

import hashlib

import fitz  # PyMuPDF
import requests
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_researcher.config import get_settings
from ai_researcher.exceptions import PDFReadError
from ai_researcher.logging import get_logger
from ai_researcher.tools.db import get_vector_store

logger = get_logger(__name__)


@tool
def read_pdf(url: str) -> str:
    """Download a PDF, extract its text into the Vector Database, and extract its figures locally.
    MUST BE RUN BEFORE querying the PDF.

    Args:
        url: The URL of the PDF file to read.
    """
    settings = get_settings()
    logger.info("Downloading and processing PDF from URL: %s", url)

    figures_dir = settings.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, timeout=settings.pdf_request_timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise PDFReadError(message=f"Failed to download PDF: {e}", url=url) from e

    try:
        doc_id = hashlib.md5(url.encode()).hexdigest()

        pdf_bytes = response.content
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        text_parts = []
        extracted_figures = []

        # Parse each page
        for page_num, page in enumerate(doc):
            # Extract Text
            page_text = page.get_text()
            if page_text:
                text_parts.append(page_text)

            # Extract Images/Figures
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)

                # Skip tiny images (< 200px width or height) typically icons or watermarks
                if base_image["width"] < 200 or base_image["height"] < 200:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Save locally
                fig_filename = f"{doc_id}_p{page_num + 1}_i{img_index}.{image_ext}"
                fig_path = figures_dir / fig_filename

                with open(fig_path, "wb") as f:
                    f.write(image_bytes)

                extracted_figures.append(str(fig_path).replace("\\", "/"))

        doc.close()

        # --- Handle Text Embedding ---
        full_text = "\n".join(text_parts).strip()

        if not full_text:
            return f"Error: PDF at {url} appears to contain no readable text."

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        chunks = text_splitter.split_text(full_text)

        docs = [
            Document(page_content=chunk, metadata={"source": url, "doc_id": doc_id})
            for chunk in chunks
        ]
        vector_store = get_vector_store()
        vector_store.add_documents(docs)

        # --- Compile the LLM Report ---
        logger.info(
            "Successfully ingested %d text chunks and %d valid figures from PDF.",
            len(chunks),
            len(extracted_figures),
        )

        report = f"SUCCESS: PDF from {url} has been parsed.\n"
        report += "- Text has been ingested. You MUST use the `query_pdf` tool to ask questions about its contents.\n"

        if extracted_figures:
            report += f"- Found {len(extracted_figures)} figures/tables. They are saved locally at:\n"
            for fig in extracted_figures:
                report += f"  - {fig}\n"
            report += "Note: You can pass these local file paths in your research summary so the Writer agent can optionally embed them using '\\includegraphics{...}'!"
        else:
            report += "- No readable figures/tables larger than 200px were found."

        return report

    except Exception as e:
        raise PDFReadError(
            message=f"Failed to process and embed PDF: {e}",
            url=url,
        ) from e
