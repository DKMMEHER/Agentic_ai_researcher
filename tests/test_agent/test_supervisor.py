"""Tests for the Supervisor agent node (intent classification)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ai_researcher.agent.supervisor import SupervisorOutput, _call_supervisor


@pytest.fixture
def make_state():
    """Factory for creating a minimal AgentState-like dict."""

    def _factory(user_msg: str = "Hello"):
        return {
            "messages": [HumanMessage(content=user_msg)],
            "intent": "",
            "current_agent": "",
            "research_notes": [],
            "researcher_iterations": 0,
            "writer_iterations": 0,
        }

    return _factory


class TestSupervisorOutput:
    """Tests for the SupervisorOutput Pydantic model."""

    def test_valid_research_paper_intent(self):
        output = SupervisorOutput(intent="research_paper")
        assert output.intent == "research_paper"
        assert output.chat_response == ""

    def test_valid_direct_chat_intent(self):
        output = SupervisorOutput(intent="direct_chat", chat_response="Hi there!")
        assert output.intent == "direct_chat"
        assert output.chat_response == "Hi there!"

    def test_invalid_intent_raises(self):
        with pytest.raises(ValueError):
            SupervisorOutput(intent="invalid_intent")


class TestCallSupervisor:
    """Tests for the _call_supervisor async node function."""

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    @patch(
        "ai_researcher.agent.supervisor.load_prompt",
        return_value="You are a supervisor.",
    )
    def test_classifies_research_paper(
        self, mock_prompt, mock_llm_class, make_state
    ):
        """When the LLM returns research_paper intent, the state is updated correctly."""
        mock_model = MagicMock()
        mock_llm_class.return_value = mock_model

        prediction = SupervisorOutput(intent="research_paper")
        raw_msg = AIMessage(content="")
        mock_structured = MagicMock(return_value={"parsed": prediction, "raw": raw_msg})
        mock_model.with_structured_output.return_value = MagicMock(
            invoke=mock_structured
        )

        state = make_state("Write a paper on transformers")
        result = _call_supervisor(state)

        assert result["intent"] == "research_paper"
        assert result["current_agent"] == "supervisor"
        assert "messages" not in result  # No direct chat message for research

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    @patch(
        "ai_researcher.agent.supervisor.load_prompt",
        return_value="You are a supervisor.",
    )
    def test_classifies_direct_chat(
        self, mock_prompt, mock_llm_class, make_state
    ):
        """When the LLM returns direct_chat, the state includes an AIMessage response."""
        mock_model = MagicMock()
        mock_llm_class.return_value = mock_model

        prediction = SupervisorOutput(
            intent="direct_chat", chat_response="Hello! How can I help?"
        )
        raw_msg = AIMessage(content="")
        mock_structured = MagicMock(return_value={"parsed": prediction, "raw": raw_msg})
        mock_model.with_structured_output.return_value = MagicMock(
            invoke=mock_structured
        )

        state = make_state("Hello")
        result = _call_supervisor(state)

        assert result["intent"] == "direct_chat"
        assert "messages" in result
        assert result["messages"][0].content == "Hello! How can I help?"

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    @patch(
        "ai_researcher.agent.supervisor.load_prompt",
        return_value="You are a supervisor.",
    )
    def test_defaults_on_llm_failure(
        self, mock_prompt, mock_llm_class, make_state
    ):
        """When the LLM raises an exception, defaults to research_paper."""
        mock_model = MagicMock()
        mock_llm_class.return_value = mock_model

        mock_structured = MagicMock(side_effect=Exception("API timeout"))
        mock_model.with_structured_output.return_value = MagicMock(
            invoke=mock_structured
        )

        state = make_state("Some query")
        result = _call_supervisor(state)

        assert result["intent"] == "research_paper"

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    @patch(
        "ai_researcher.agent.supervisor.load_prompt",
        return_value="You are a supervisor.",
    )
    def test_defaults_when_parsed_is_none(
        self, mock_prompt, mock_llm_class, make_state
    ):
        """When structured output returns None for parsed, defaults to research_paper."""
        mock_model = MagicMock()
        mock_llm_class.return_value = mock_model

        raw_msg = AIMessage(content="garbage")
        mock_structured = MagicMock(return_value={"parsed": None, "raw": raw_msg})
        mock_model.with_structured_output.return_value = MagicMock(
            invoke=mock_structured
        )

        state = make_state("Some query")
        result = _call_supervisor(state)

        assert result["intent"] == "research_paper"
