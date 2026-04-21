"""Custom exception hierarchy for the AI Researcher application.

All exceptions inherit from `AIResearcherError` to allow catching
any application-specific error with a single except clause.
"""


class AIResearcherError(Exception):
    """Base exception for all AI Researcher errors."""

    def __init__(self, message: str = "", *args, **kwargs):
        self.message = message
        super().__init__(message, *args, **kwargs)


# --- Configuration Errors ---


class ConfigurationError(AIResearcherError):
    """Raised when application configuration is invalid or missing."""


# --- Tool Errors ---


class ToolError(AIResearcherError):
    """Base exception for tool-related errors."""


class ArxivSearchError(ToolError):
    """Raised when an arXiv API search fails."""

    def __init__(self, message: str = "", query: str = "", *args, **kwargs):
        self.query = query
        super().__init__(message, *args, **kwargs)


class PDFReadError(ToolError):
    """Raised when PDF reading or text extraction fails."""

    def __init__(self, message: str = "", url: str = "", *args, **kwargs):
        self.url = url
        super().__init__(message, *args, **kwargs)


class LatexRenderError(ToolError):
    """Raised when LaTeX rendering to PDF fails."""

    def __init__(
        self, message: str = "", tex_file: str = "", *args, **kwargs
    ):
        self.tex_file = tex_file
        super().__init__(message, *args, **kwargs)


class WebSearchError(ToolError):
    """Raised when a web search (DuckDuckGo or Tavily) fails."""

    def __init__(self, message: str = "", query: str = "", engine: str = "", *args, **kwargs):
        self.query = query
        self.engine = engine
        super().__init__(message, *args, **kwargs)
