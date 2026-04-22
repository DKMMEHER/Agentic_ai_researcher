"""Pydantic data models for structured data exchange.

These models replace raw dict returns with typed, validated objects.
"""

from pydantic import BaseModel, Field


class ArxivPaper(BaseModel):
    """Represents a single arXiv paper with its metadata."""

    title: str = Field(description="Paper title")
    summary: str = Field(description="Paper abstract/summary")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    categories: list[str] = Field(
        default_factory=list, description="arXiv category tags"
    )
    pdf_url: str | None = Field(default=None, description="Direct URL to the PDF")

    class Config:
        json_schema_extra = {  # noqa: RUF012
            "example": {
                "title": "Attention Is All You Need",
                "summary": "The dominant sequence transduction models...",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "categories": ["cs.CL", "cs.LG"],
                "pdf_url": "http://arxiv.org/pdf/1706.03762v7",
            }
        }


class SearchResult(BaseModel):
    """Collection of arXiv search results."""

    query: str = Field(description="The original search query")
    total_results: int = Field(description="Number of papers returned")
    papers: list[ArxivPaper] = Field(
        default_factory=list, description="List of matching papers"
    )

    @classmethod
    def from_entries(cls, query: str, entries: list[dict]) -> "SearchResult":
        """Create a SearchResult from raw arXiv API parsed entries.

        Args:
            query: The original search query.
            entries: List of parsed entry dicts from XML parsing.

        Returns:
            Structured SearchResult instance.
        """
        papers = [
            ArxivPaper(
                title=e.get("title", "").strip(),
                summary=e.get("summary", "").strip(),
                authors=e.get("authors", []),
                categories=e.get("categories", []),
                pdf_url=e.get("pdf"),
            )
            for e in entries
        ]
        return cls(query=query, total_results=len(papers), papers=papers)
