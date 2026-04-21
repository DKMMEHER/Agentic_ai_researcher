"""Shared fixtures for end-to-end integration tests.

Provides mock LLM responses, ephemeral ChromaDB, and fresh graph builders
so that E2E tests exercise the real LangGraph routing without needing
live API keys or network access.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

# ---------------------------------------------------------------------------
# Helper: build a mock AIMessage that looks like a tool-call response
# ---------------------------------------------------------------------------

def make_tool_call_message(
    tool_name: str,
    tool_args: dict | None = None,
    content: str = "",
) -> AIMessage:
    """Create an AIMessage with a single tool_call attachment.

    This mirrors the shape LangChain returns from model.invoke()
    when the LLM decides to call a tool.
    """
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    return AIMessage(
        content=content,
        tool_calls=[
            {
                "id": call_id,
                "name": tool_name,
                "args": tool_args or {},
            }
        ],
    )


def make_plain_message(content: str) -> AIMessage:
    """Create a plain AIMessage with no tool calls (conversational reply)."""
    return AIMessage(content=content)


# ---------------------------------------------------------------------------
# Fixture: ephemeral in-memory ChromaDB (no disk I/O)
# ---------------------------------------------------------------------------

@pytest.fixture()
def ephemeral_chroma():
    """Patch the singleton vector store with an ephemeral in-memory Chroma.

    Yields the patched Chroma instance so tests can inspect stored docs.
    """
    import chromadb
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    # Create a truly ephemeral client — no files written to disk
    ephemeral_client = chromadb.Client()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    store = Chroma(
        client=ephemeral_client,
        collection_name="test_research_papers",
        embedding_function=embeddings,
    )

    with patch("ai_researcher.tools.db._VECTOR_STORE", store), \
         patch("ai_researcher.tools.db._EMBEDDINGS", embeddings), \
         patch("ai_researcher.tools.db.get_vector_store", return_value=store):
        yield store


# ---------------------------------------------------------------------------
# Fixture: mock LLM factory
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm_factory():
    """Factory that returns a mock ChatGroq whose responses you control.

    Usage in tests::

        mock_model = mock_llm_factory([
            make_tool_call_message("arxiv_search", {"topic": "transformers"}),
            make_plain_message("Here are the results..."),
        ])
    """

    def _factory(responses: list[AIMessage]):
        model = MagicMock()
        model.invoke = MagicMock(side_effect=responses)
        # bind_tools should return itself (the graph calls bind_tools once)
        model.bind_tools = MagicMock(return_value=model)
        return model

    return _factory


# ---------------------------------------------------------------------------
# Fixture: build a fresh graph with a mocked LLM
# ---------------------------------------------------------------------------

@pytest.fixture()
def build_e2e_graph(mock_llm_factory):
    """Build a real LangGraph agent graph with a mocked LLM.

    Returns a factory callable:
        graph, config = build_e2e_graph(responses=[...])
    
    Mocks both ChatGroq and ChatGoogleGenerativeAI so the graph works
    regardless of which MODEL_NAME is configured.
    """

    def _build(responses: list[AIMessage], thread_id: str | None = None):
        if thread_id is None:
            thread_id = f"e2e-{uuid.uuid4().hex[:8]}"

        mock_model = mock_llm_factory(responses)

        # Patch both for safety, but Gemini is now primary
        with patch("ai_researcher.agent.graph.ChatGoogleGenerativeAI", return_value=mock_model), \
             patch("ai_researcher.agent.supervisor.ChatGoogleGenerativeAI", return_value=mock_model), \
             patch("ai_researcher.agent.graph.ChatGroq", return_value=mock_model), \
             patch("ai_researcher.agent.supervisor.ChatGroq", return_value=mock_model):
            from ai_researcher.agent.graph import build_graph

            graph, config = build_graph(thread_id=thread_id)

        return graph, config

    return _build
