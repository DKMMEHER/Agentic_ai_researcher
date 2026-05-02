"""AI Researcher — An agentic AI system for academic paper research and generation."""

__version__ = "0.2.0"
__author__ = "AI Researcher Team"

import os
import warnings

# Must be set BEFORE transformers is imported anywhere in the dependency chain.
# This suppresses the internal "Accessing `__path__`" advisory warnings from HuggingFace.
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", message=".*Accessing `__path__`.*")

# NOTE: build_graph and get_settings are imported lazily (not at module level)
# to avoid triggering heavy PyTorch/HuggingFace imports during container startup.
# This allows FastAPI to bind its port and respond to health checks immediately.

__all__ = ["__version__", "build_graph", "get_settings"]


def __getattr__(name: str):
    """Lazy imports for heavy modules to speed up container startup."""
    if name == "build_graph":
        from ai_researcher.agent.graph import build_graph

        return build_graph
    if name == "get_settings":
        from ai_researcher.config import get_settings

        return get_settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
