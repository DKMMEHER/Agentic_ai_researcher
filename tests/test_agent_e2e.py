"""End-to-end integration tests for the AI Researcher agent.

These tests build the REAL LangGraph agent graph (with a mocked LLM)
and verify that tool routing, ChromaDB ingestion, and multi-step
workflows behave correctly end-to-end.

Marked with @pytest.mark.e2e so they can be run separately:
    pytest -m e2e
    pytest -m "not e2e"   # skip E2E, run unit tests only

Usage:
    uv run pytest tests/test_agent_e2e.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest_e2e import make_plain_message, make_tool_call_message

pytestmark = pytest.mark.e2e
pytest_plugins = ["tests.conftest_e2e"]


# =========================================================================
# Test 1: arXiv search routing
# =========================================================================


class TestArxivSearchRouting:
    """Verify that the agent routes an arXiv question to the arxiv_search tool."""

    @patch("ai_researcher.tools.arxiv.requests.get")
    def test_agent_calls_arxiv_search(
        self, mock_http_get, build_e2e_graph, sample_arxiv_xml
    ):
        """'Search arxiv for transformers' → agent must invoke arxiv_search."""
        # --- Arrange: LLM script ---
        responses = [
            # Step 1: LLM decides to call arxiv_search
            make_tool_call_message(
                "arxiv_search", {"topic": "transformers"}
            ),
            # Step 2: After tool result, LLM gives final answer
            make_plain_message(
                "I found 2 papers on transformers. Here are the results..."
            ),
            # Step 3: Writer acts
            make_plain_message("LaTeX output"),
        ]
        graph, config = build_e2e_graph(responses=responses)

        # Mock the arXiv HTTP call so no real network request is made
        mock_response = MagicMock()
        mock_response.text = sample_arxiv_xml
        mock_response.raise_for_status = MagicMock()
        mock_http_get.return_value = mock_response

        # --- Act ---
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Search arxiv for transformers"}]},
            config,
        )

        # --- Assert ---
        messages = result["messages"]
        # There should be at least: user → AI(tool_call) → ToolMessage → AI(final)
        assert len(messages) >= 4

        # The second message (AI) must have a tool_call for arxiv_search
        ai_tool_msg = messages[1]
        assert ai_tool_msg.tool_calls, "LLM should have generated a tool call"
        assert ai_tool_msg.tool_calls[0]["name"] == "arxiv_search"

        # The arXiv HTTP endpoint was actually hit (via the tool node)
        mock_http_get.assert_called_once()


# =========================================================================
# Test 2: PDF ingestion to ChromaDB
# =========================================================================


class TestPdfIngestion:
    """Verify that read_pdf ingests chunks into ChromaDB."""

    @patch("ai_researcher.tools.pdf_reader.requests.get")
    def test_pdf_ingested_into_chromadb(
        self, mock_http_get, build_e2e_graph, ephemeral_chroma
    ):
        """'Read this PDF' → read_pdf called → ChromaDB receives chunks."""
        # Minimal valid PDF bytes (PyPDF2 can parse this)
        # We'll mock PyPDF2 to avoid needing a real PDF binary
        fake_url = "http://arxiv.org/pdf/1706.03762v7"

        responses = [
            make_tool_call_message("read_pdf", {"url": fake_url}),
            make_plain_message("The PDF has been ingested successfully."),
            make_plain_message("LaTeX output"),
        ]
        graph, config = build_e2e_graph(responses=responses)

        # Mock HTTP download to return bytes that PyPDF2 can process
        # We mock at the PyPDF2 level for reliability
        mock_response = MagicMock()
        mock_response.content = b"%PDF-fake"
        mock_response.raise_for_status = MagicMock()
        mock_http_get.return_value = mock_response

        fake_pages_text = [
            "This paper proposes the Transformer architecture.",
            "We use self-attention mechanisms for sequence transduction.",
            "Experiments show the model achieves state-of-the-art results.",
        ]

        mock_reader = MagicMock()
        mock_pages = []
        for text in fake_pages_text:
            page = MagicMock()
            page.extract_text.return_value = text
            mock_pages.append(page)
        mock_reader.pages = mock_pages

        with patch("ai_researcher.tools.pdf_reader.PyPDF2.PdfReader", return_value=mock_reader):
            result = graph.invoke(
                {"messages": [{"role": "user", "content": f"Read this PDF: {fake_url}"}]},
                config,
            )

        # Verify chunks landed in ChromaDB
        stored = ephemeral_chroma.similarity_search(
            "transformer architecture", k=10, filter={"source": fake_url}
        )
        assert len(stored) > 0, "ChromaDB should contain chunks from the ingested PDF"

        # Verify the tool was actually called in the message stream
        messages = result["messages"]
        tool_call_names = [
            tc["name"]
            for m in messages
            if getattr(m, "tool_calls", None)
            for tc in m.tool_calls
        ]
        assert "read_pdf" in tool_call_names


# =========================================================================
# Test 3: query_pdf returns results
# =========================================================================


class TestQueryPdf:
    """Verify query_pdf retrieves results from ChromaDB after ingestion."""

    def test_query_pdf_returns_results(self, build_e2e_graph, ephemeral_chroma):
        """After ingestion, 'What is the methodology?' → query_pdf returns content."""
        fake_url = "http://arxiv.org/pdf/1706.03762v7"

        # Pre-populate ChromaDB with some chunks (simulating prior read_pdf)
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content="The methodology uses multi-head self-attention layers.",
                metadata={"source": fake_url, "doc_id": "test123"},
            ),
            Document(
                page_content="Training uses the Adam optimizer with warmup schedule.",
                metadata={"source": fake_url, "doc_id": "test123"},
            ),
        ]
        ephemeral_chroma.add_documents(docs)

        responses = [
            make_tool_call_message(
                "query_pdf",
                {"url": fake_url, "search_query": "What is the methodology?"},
            ),
            make_plain_message(
                "The methodology uses multi-head self-attention layers..."
            ),
            make_plain_message("LaTeX output"),
        ]
        graph, config = build_e2e_graph(responses=responses)

        result = graph.invoke(
            {"messages": [{"role": "user", "content": "What is the methodology in this paper?"}]},
            config,
        )

        messages = result["messages"]

        # Find the ToolMessage (result from query_pdf)
        from langchain_core.messages import ToolMessage

        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_results) >= 1, "query_pdf should return a ToolMessage"

        # The tool result should contain the word 'methodology' or 'attention'
        tool_content = tool_results[0].content
        assert any(
            keyword in tool_content.lower()
            for keyword in ["methodology", "attention", "self-attention"]
        ), f"query_pdf result should contain relevant content, got: {tool_content[:200]}"


# =========================================================================
# Test 4: Web search routing
# =========================================================================


class TestWebSearchRouting:
    """Verify agent routes web search queries to duckduckgo_search."""

    @patch("ai_researcher.tools.web_search.DDGS")
    def test_agent_calls_duckduckgo(self, mock_ddgs_class, build_e2e_graph):
        """'Search the web for...' → duckduckgo_search is called."""
        responses = [
            make_tool_call_message(
                "duckduckgo_search",
                {"query": "latest breakthroughs in quantum computing 2025"},
            ),
            make_plain_message("Here is what I found about quantum computing..."),
            make_plain_message("LaTeX output"),
        ]
        graph, config = build_e2e_graph(responses=responses)

        # Mock DuckDuckGo search
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text.return_value = [
            {
                "title": "Quantum Computing Breakthrough",
                "href": "https://example.com/quantum",
                "body": "Researchers have achieved a major breakthrough...",
            }
        ]
        mock_ddgs_class.return_value = mock_ddgs_instance

        result = graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "Search the web for latest breakthroughs in quantum computing"}
                ]
            },
            config,
        )

        messages = result["messages"]
        tool_call_names = [
            tc["name"]
            for m in messages
            if getattr(m, "tool_calls", None)
            for tc in m.tool_calls
        ]
        assert "duckduckgo_search" in tool_call_names
        mock_ddgs_instance.text.assert_called_once()


# =========================================================================
# Test 5: Multi-step tool chain (search → read)
# =========================================================================


class TestMultiStepChain:
    """Verify the agent can execute a multi-step tool chain."""

    @patch("ai_researcher.tools.pdf_reader.requests.get")
    @patch("ai_researcher.tools.arxiv.requests.get")
    def test_search_then_read_pdf(
        self,
        mock_arxiv_http,
        mock_pdf_http,
        build_e2e_graph,
        sample_arxiv_xml,
        ephemeral_chroma,
    ):
        """Search → agent finds PDF → reads PDF → 2-tool chain verified."""
        fake_url = "http://arxiv.org/pdf/1706.03762v7"

        responses = [
            # Step 1: LLM calls arxiv_search
            make_tool_call_message(
                "arxiv_search", {"topic": "attention mechanisms"}
            ),
            # Step 2: After seeing arXiv results, LLM calls read_pdf
            make_tool_call_message("read_pdf", {"url": fake_url}),
            # Step 3: Final answer
            make_plain_message(
                "I've searched arXiv and ingested the paper on attention mechanisms."
            ),
            # Step 4: Writer
            make_plain_message("LaTeX output"),
        ]
        graph, config = build_e2e_graph(responses=responses)

        # Mock arXiv HTTP
        mock_arxiv_resp = MagicMock()
        mock_arxiv_resp.text = sample_arxiv_xml
        mock_arxiv_resp.raise_for_status = MagicMock()
        mock_arxiv_http.return_value = mock_arxiv_resp

        # Mock PDF HTTP + PyPDF2
        mock_pdf_resp = MagicMock()
        mock_pdf_resp.content = b"%PDF-fake"
        mock_pdf_resp.raise_for_status = MagicMock()
        mock_pdf_http.return_value = mock_pdf_resp

        mock_reader = MagicMock()
        page = MagicMock()
        page.extract_text.return_value = "Attention is all you need content."
        mock_reader.pages = [page]

        with patch(
            "ai_researcher.tools.pdf_reader.PyPDF2.PdfReader",
            return_value=mock_reader,
        ):
            result = graph.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Search for attention mechanisms on arxiv and then read the first paper",
                        }
                    ]
                },
                config,
            )

        messages = result["messages"]
        tool_call_names = [
            tc["name"]
            for m in messages
            if getattr(m, "tool_calls", None)
            for tc in m.tool_calls
        ]

        # Both tools should have been called in sequence
        assert "arxiv_search" in tool_call_names, "arxiv_search should be in the chain"
        assert "read_pdf" in tool_call_names, "read_pdf should be in the chain"

        # Verify order: arxiv_search before read_pdf
        arxiv_idx = tool_call_names.index("arxiv_search")
        pdf_idx = tool_call_names.index("read_pdf")
        assert arxiv_idx < pdf_idx, "arxiv_search should come before read_pdf"


# =========================================================================
# Test 6: Conversational (no tool call)
# =========================================================================


class TestConversationalNoTool:
    """Verify the agent can respond without calling any tool."""

    def test_greeting_does_not_call_tools(self, build_e2e_graph):
        """'Hello, who are you?' → agent responds directly, no tool calls."""
        responses = [
            make_plain_message(
                "Hello! I'm an AI research assistant. I can help you search "
                "for papers on arXiv, read PDFs, and write LaTeX documents. "
                "What topic would you like to explore?"
            ),
            make_plain_message("LaTeX output"),
        ]
        graph, config = build_e2e_graph(responses=responses)

        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Hello, who are you?"}]},
            config,
        )

        messages = result["messages"]

        # Should be exactly 3 messages: user + AI Researcher response + AI Writer response
        assert len(messages) == 3

        # The AI message should have NO tool calls
        ai_msg = messages[-1]
        assert not getattr(
            ai_msg, "tool_calls", None
        ), "Greeting should not trigger any tool calls"

        # Should contain a sensible response
        assert len(ai_msg.content) > 10, "AI should give a substantive response"
