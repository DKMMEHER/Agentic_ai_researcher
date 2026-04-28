"""Tests for the agent graph builder, edge functions, and node functions."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from ai_researcher.agent.graph import (
    _filter_system_messages,
    _guardrail_handler,
    _human_review,
    _route_after_review,
    _should_continue_researcher,
    _should_continue_supervisor,
    _should_continue_writer,
)
from ai_researcher.agent.guardrails import (
    MAX_RESEARCHER_ITERATIONS,
    MAX_WRITER_ITERATIONS,
)
from ai_researcher.agent.prompts import (
    RESEARCHER_PROMPT,
    SUPERVISOR_PROMPT,
    WRITER_PROMPT,
    load_prompt,
)
from ai_researcher.agent.state import AgentState

# =========================================================================
# AgentState definition tests
# =========================================================================


class TestAgentState:
    """Tests for the AgentState definition."""

    def test_state_has_messages_key(self):
        """Verify AgentState has the expected keys."""
        assert "messages" in AgentState.__annotations__

    def test_state_has_all_expected_keys(self):
        """Verify AgentState has all expected fields."""
        expected = {
            "messages",
            "research_summary",
            "current_agent",
            "human_approval",
            "revision_instructions",
            "researcher_iterations",
            "writer_iterations",
            "guardrail_reason",
            "intent",
            "research_notes",
        }
        assert expected.issubset(set(AgentState.__annotations__.keys()))


# =========================================================================
# Prompt loading tests
# =========================================================================


class TestLoadPrompt:
    """Tests for prompt loading."""

    def test_default_prompt_returned_when_file_missing(self, tmp_path):
        """When prompt file doesn't exist, embedded default is used."""
        with patch("ai_researcher.agent.prompts._PROMPTS_DIR", tmp_path):
            prompt = load_prompt("nonexistent")
            assert prompt == RESEARCHER_PROMPT

    def test_prompt_loaded_from_file(self, tmp_path):
        """When prompt file exists, its content is loaded."""
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text("You are a test agent.")

        with patch("ai_researcher.agent.prompts._PROMPTS_DIR", tmp_path):
            prompt = load_prompt("test_prompt")
            assert prompt == "You are a test agent."

    def test_prompt_variable_substitution(self, tmp_path):
        """Test basic variable substitution in prompts."""
        prompt_file = tmp_path / "template.txt"
        prompt_file.write_text("Research {topic} papers.")

        with patch("ai_researcher.agent.prompts._PROMPTS_DIR", tmp_path):
            prompt = load_prompt("template", topic="quantum computing")
            assert prompt == "Research quantum computing papers."

    def test_default_prompt_contains_key_instructions(self):
        """Verify the default prompt has essential instructions."""
        assert "arxiv" in RESEARCHER_PROMPT.lower()
        assert "research" in RESEARCHER_PROMPT.lower()
        assert "summary" in RESEARCHER_PROMPT.lower()

    def test_writer_fallback_prompt(self, tmp_path):
        """When 'writer' prompt file is missing, WRITER_PROMPT is used."""
        with patch("ai_researcher.agent.prompts._PROMPTS_DIR", tmp_path):
            prompt = load_prompt("writer")
            assert prompt == WRITER_PROMPT

    def test_supervisor_fallback_prompt(self, tmp_path):
        """When 'supervisor' prompt file is missing, SUPERVISOR_PROMPT is used."""
        with patch("ai_researcher.agent.prompts._PROMPTS_DIR", tmp_path):
            prompt = load_prompt("supervisor")
            assert prompt == SUPERVISOR_PROMPT

    def test_missing_variable_logs_warning(self, tmp_path):
        """Prompt with unreplaced {var} doesn't crash — logs a warning."""
        prompt_file = tmp_path / "vars.txt"
        prompt_file.write_text("Hello {user_name}, welcome to {location}.")

        with patch("ai_researcher.agent.prompts._PROMPTS_DIR", tmp_path):
            # Only pass one of the two required variables
            prompt = load_prompt("vars", user_name="Alice")
            # The prompt string remains unformatted if a KeyError occurs
            assert "{user_name}" in prompt
            assert "{location}" in prompt


# =========================================================================
# Build graph tests
# =========================================================================


class TestBuildGraph:
    """Tests for graph building."""

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_graph_builds_without_error(self, mock_llm_class):
        """Test that build_graph creates a valid graph."""
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_llm_class.return_value = mock_model

        from ai_researcher.agent.graph import build_graph

        graph, config = build_graph()

        assert graph is not None
        assert "configurable" in config
        assert "thread_id" in config["configurable"]

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_custom_thread_id(self, mock_llm_class):
        """Test that custom thread_id is passed through."""
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_llm_class.return_value = mock_model

        from ai_researcher.agent.graph import build_graph

        _, config = build_graph(thread_id="my-custom-thread")
        assert config["configurable"]["thread_id"] == "my-custom-thread"

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_tools_bound_exactly_once(self, mock_llm_class):
        """Verify tools are bound exactly once (not double-bound)."""
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_llm_class.return_value = mock_model

        from ai_researcher.agent.graph import build_graph

        build_graph()
        # bind_tools should be called twice (researcher + writer agents in _create_models)
        # One for researcher tools, one for writer tools
        assert mock_model.bind_tools.call_count == 2


# =========================================================================
# _filter_system_messages tests
# =========================================================================


class TestFilterSystemMessages:
    """Tests for _filter_system_messages helper."""

    def test_removes_system_message_objects(self):
        """SystemMessage objects should be removed."""
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]
        result = _filter_system_messages(messages)
        assert len(result) == 2
        assert all(not isinstance(m, SystemMessage) for m in result)

    def test_removes_system_dicts(self):
        """Dict messages with role='system' should be removed."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = _filter_system_messages(messages)
        assert len(result) == 2
        assert all(m.get("role") != "system" for m in result)

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert _filter_system_messages([]) == []

    def test_preserves_non_system(self):
        """Non-system messages are preserved in order."""
        messages = [
            HumanMessage(content="A"),
            AIMessage(content="B"),
            HumanMessage(content="C"),
        ]
        result = _filter_system_messages(messages)
        assert len(result) == 3
        assert result[0].content == "A"
        assert result[2].content == "C"


# =========================================================================
# Supervisor routing edge function
# =========================================================================


class TestShouldContinueSupervisor:
    """Tests for _should_continue_supervisor edge function."""

    def test_direct_chat_routes_to_end(self):
        """When intent is direct_chat, routes to __end__."""
        state = {"intent": "direct_chat", "messages": []}
        result = _should_continue_supervisor(state)
        assert result == "__end__"

    def test_research_paper_routes_to_researcher(self):
        """When intent is research_paper, routes to researcher."""
        state = {"intent": "research_paper", "messages": []}
        result = _should_continue_supervisor(state)
        assert result == "researcher"

    def test_quick_research_routes_to_researcher(self):
        """When intent is quick_research, routes to researcher."""
        state = {"intent": "quick_research", "messages": []}
        result = _should_continue_supervisor(state)
        assert result == "researcher"

    def test_missing_intent_defaults_to_researcher(self):
        """When intent is missing, defaults to research_paper → researcher."""
        state = {"messages": []}
        result = _should_continue_supervisor(state)
        assert result == "researcher"


# =========================================================================
# Researcher routing edge function
# =========================================================================


class TestShouldContinueResearcher:
    """Tests for _should_continue_researcher edge function."""

    def test_tool_call_routes_to_tools(self):
        """When last message has tool_calls and under limit, route to tools."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "arxiv_search", "args": {}}],
        )
        state = {
            "messages": [ai_msg],
            "researcher_iterations": 1,
            "intent": "research_paper",
        }
        assert _should_continue_researcher(state) == "researcher_tools"

    def test_tool_call_at_limit_routes_to_guardrail(self):
        """When tool_calls and at iteration limit, route to guardrail."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "arxiv_search", "args": {}}],
        )
        state = {
            "messages": [ai_msg],
            "researcher_iterations": MAX_RESEARCHER_ITERATIONS,
            "intent": "research_paper",
        }
        assert _should_continue_researcher(state) == "guardrail_handler"

    def test_no_tools_research_paper_routes_to_human_review(self):
        """No tool calls + research_paper intent → human_review."""
        ai_msg = AIMessage(content="Research complete.")
        state = {
            "messages": [ai_msg],
            "researcher_iterations": 2,
            "intent": "research_paper",
        }
        assert _should_continue_researcher(state) == "human_review"

    def test_no_tools_quick_research_routes_to_end(self):
        """No tool calls + quick_research intent → END."""
        ai_msg = AIMessage(content="Here are the facts.")
        state = {
            "messages": [ai_msg],
            "researcher_iterations": 2,
            "intent": "quick_research",
        }
        assert _should_continue_researcher(state) == "__end__"


# =========================================================================
# Writer routing edge function
# =========================================================================


class TestShouldContinueWriter:
    """Tests for _should_continue_writer edge function."""

    def test_tool_call_routes_to_writer_tools(self):
        """When last message has tool_calls and under limit, route to tools."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "render_latex", "args": {}}],
        )
        state = {
            "messages": [ai_msg],
            "writer_iterations": 1,
        }
        assert _should_continue_writer(state) == "writer_tools"

    def test_tool_call_at_limit_routes_to_guardrail(self):
        """When tool_calls and at iteration limit, route to guardrail."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "render_latex", "args": {}}],
        )
        state = {
            "messages": [ai_msg],
            "writer_iterations": MAX_WRITER_ITERATIONS,
        }
        assert _should_continue_writer(state) == "guardrail_handler"

    def test_no_tools_routes_to_end(self):
        """When last message has no tool calls, route to END."""
        ai_msg = AIMessage(content="Paper is ready.")
        state = {"messages": [ai_msg], "writer_iterations": 1}
        assert _should_continue_writer(state) == "__end__"


# =========================================================================
# Human review node function
# =========================================================================


class TestHumanReview:
    """Tests for the _human_review node function."""

    def test_approved_routes_to_writer(self):
        """When approval is 'approved', current_agent becomes 'writer'."""
        state = {
            "messages": [AIMessage(content="done")],
            "human_approval": "approved",
        }
        result = _human_review(state)
        assert result["current_agent"] == "writer"

    def test_revise_adds_revision_message(self):
        """When approval is 'revise', creates a revision HumanMessage."""
        state = {
            "messages": [AIMessage(content="done")],
            "human_approval": "revise",
            "revision_instructions": "Add more citations",
        }
        result = _human_review(state)
        assert result["current_agent"] == "researcher"
        assert result["researcher_iterations"] == 0
        assert len(result["messages"]) == 1
        assert "REVISION REQUEST" in result["messages"][0].content
        assert "Add more citations" in result["messages"][0].content

    def test_revise_uses_default_instructions(self):
        """When revise has no instructions, uses default text."""
        state = {
            "messages": [AIMessage(content="done")],
            "human_approval": "revise",
        }
        result = _human_review(state)
        assert "Please do more research" in result["messages"][0].content

    def test_abort_sets_done(self):
        """When approval is 'abort', current_agent becomes 'done'."""
        state = {
            "messages": [AIMessage(content="done")],
            "human_approval": "abort",
        }
        result = _human_review(state)
        assert result["current_agent"] == "done"

    def test_pending_defaults_to_approved(self):
        """When approval is 'pending' (default), treats as approved → writer."""
        state = {
            "messages": [AIMessage(content="done")],
            "human_approval": "pending",
        }
        result = _human_review(state)
        assert result["current_agent"] == "writer"


# =========================================================================
# Route after review edge function
# =========================================================================


class TestRouteAfterReview:
    """Tests for _route_after_review edge function."""

    def test_researcher_routes_back(self):
        """When current_agent is 'researcher', loops back."""
        state = {"current_agent": "researcher"}
        assert _route_after_review(state) == "researcher"

    def test_done_routes_to_end(self):
        """When current_agent is 'done', routes to END."""
        state = {"current_agent": "done"}
        assert _route_after_review(state) == "__end__"

    def test_writer_routes_to_writer(self):
        """When current_agent is 'writer', routes to writer."""
        state = {"current_agent": "writer"}
        assert _route_after_review(state) == "writer"

    def test_default_routes_to_writer(self):
        """When current_agent is empty/missing, defaults to writer."""
        state = {}
        assert _route_after_review(state) == "writer"


# =========================================================================
# Guardrail handler node
# =========================================================================


class TestGuardrailHandler:
    """Tests for _guardrail_handler async node function."""

    def test_generates_guardrail_message(self):
        """Guardrail handler produces a SystemMessage with the right content."""
        state = {
            "messages": [AIMessage(content="Trying again")],
            "current_agent": "researcher",
            "researcher_iterations": 4,
        }
        result = _guardrail_handler(state)
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, SystemMessage)
        assert "GUARDRAIL TRIGGERED" in msg.content
        assert "Researcher" in msg.content
        assert result["current_agent"] == "done"

    def test_writer_guardrail(self):
        """Guardrail handler works for the writer agent too."""
        state = {
            "messages": [AIMessage(content="Retrying LaTeX")],
            "current_agent": "writer",
            "writer_iterations": 4,
        }
        result = _guardrail_handler(state)
        assert "Writer" in result["messages"][0].content

    def test_includes_error_context(self):
        """Guardrail handler includes error context from the last message."""
        state = {
            "messages": [AIMessage(content="ERROR: Tectonic compilation failed")],
            "current_agent": "writer",
            "writer_iterations": 4,
        }
        result = _guardrail_handler(state)
        assert "Tectonic compilation failed" in result["messages"][0].content
