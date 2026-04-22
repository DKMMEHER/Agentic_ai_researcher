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

from ai_researcher.agent.graph import build_graph  # noqa: E402
from ai_researcher.config import get_settings  # noqa: E402

__all__ = ["__version__", "build_graph", "get_settings"]
