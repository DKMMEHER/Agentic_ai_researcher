"""Shared fixtures for end-to-end integration tests.

Provides mock LLM responses, ephemeral ChromaDB, and fresh graph builders
so that E2E tests exercise the real LangGraph routing without needing
live API keys or network access.

Strategy:
    Patch `_create_models` and `_call_supervisor` at the graph module level
    so that the patches remain active for the entire test, including during
    `graph.invoke()`. This completely bypasses LLM instantiation.
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

    with (
        patch("ai_researcher.tools.db._VECTOR_STORE", store),
        patch("ai_researcher.tools.db._EMBEDDINGS", embeddings),
        patch("ai_researcher.tools.db.get_vector_store", return_value=store),
    ):
        yield store


# ---------------------------------------------------------------------------
# Fixture: build a fresh graph with a mocked LLM
# ---------------------------------------------------------------------------


@pytest.fixture()
def build_e2e_graph():
    """Build a real LangGraph agent graph with a mocked LLM.

    Returns a factory callable that yields (graph, config).
    The patches remain active for the lifetime of the test.

    Strategy:
    - Patch `_create_models` → returns mock models (bypasses all LLM imports)
    - Patch `_call_supervisor` → returns a deterministic intent
    - Use MemorySaver for the checkpointer (no file I/O, no async)
    """
    # Keep track of active patches so we can clean them up
    active_patches: list = []

    def _build(
        responses: list[AIMessage],
        thread_id: str | None = None,
        supervisor_intent: str = "research_paper",
    ):
        if thread_id is None:
            thread_id = f"e2e-{uuid.uuid4().hex[:8]}"

        # ── Build mock model ──────────────────────────────────────────
        mock_model = MagicMock()
        mock_model.invoke = MagicMock(side_effect=responses)
        mock_model.bind_tools = MagicMock(return_value=mock_model)

        # ── Build deterministic supervisor node ───────────────────────
        def fake_supervisor(state):
            """Deterministic supervisor that returns the specified intent."""
            update: dict = {"intent": supervisor_intent, "current_agent": "supervisor"}
            if supervisor_intent == "direct_chat":
                update["messages"] = [
                    AIMessage(
                        content=(
                            "Hello! I'm an AI research assistant. I can help you "
                            "search for papers on arXiv, read PDFs, and write "
                            "LaTeX documents. What topic would you like to explore?"
                        )
                    )
                ]
            return update

        # ── Patch _create_models to bypass all LLM instantiation ─────
        def fake_create_models(_settings):
            return mock_model, mock_model

        # Start patches and keep them active
        p1 = patch(
            "ai_researcher.agent.graph._create_models",
            new=fake_create_models,
        )
        p2 = patch(
            "ai_researcher.agent.graph._call_supervisor",
            new=fake_supervisor,
        )
        p1.start()
        p2.start()
        active_patches.extend([p1, p2])

        from langgraph.checkpoint.memory import MemorySaver

        from ai_researcher.agent.graph import build_graph

        graph, config = build_graph(
            thread_id=thread_id,
            checkpointer=MemorySaver(),
        )

        return graph, config

    yield _build

    # Cleanup: stop all patches after the test finishes
    for p in active_patches:
        p.stop()
