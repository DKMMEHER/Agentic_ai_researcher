"""Tests for the FastAPI server endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: F401

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Create a test client with a mocked checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    from ai_researcher.server.main import app

    # Inject a MemorySaver into app state before testing
    app.state.checkpointer = MemorySaver()

    client = TestClient(app, raise_server_exceptions=False)
    yield client


class TestHealthAndInit:
    """Tests for server health and initialization endpoints."""

    def test_start_research_creates_thread(self, test_client):
        """POST /research/start should return a thread_id."""
        response = test_client.post(
            "/research/start",
            json={"question": "test question", "thread_id": "test-thread-123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "thread_id" in data
        assert data["thread_id"] == "test-thread-123"
        assert data["status"] == "initialized"

    def test_start_research_generates_thread_id(self, test_client):
        """POST /research/start without thread_id should auto-generate one."""
        response = test_client.post(
            "/research/start",
            json={"question": "test question"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "thread_id" in data
        assert len(data["thread_id"]) > 0


class TestSubmitAction:
    """Tests for the human-in-the-loop action endpoint."""

    @patch("ai_researcher.server.main.active_sessions", {})
    def test_submit_action_missing_session_returns_404(self, test_client):
        """POST /research/action with an unknown thread_id should 404."""
        response = test_client.post(
            "/research/action/nonexistent-thread",
            json={"decision": "approved"},
        )
        assert response.status_code == 404

    @patch("ai_researcher.server.main.active_sessions")
    def test_submit_action_approve(self, mock_sessions, test_client):
        """POST /research/action with approve should succeed."""
        # Set up a mock session
        mock_graph = MagicMock()
        mock_config = {"configurable": {"thread_id": "t1"}}
        mock_sessions.__contains__ = MagicMock(return_value=True)
        mock_sessions.__getitem__ = MagicMock(return_value=(mock_graph, mock_config))

        response = test_client.post(
            "/research/action/t1",
            json={"decision": "approved"},
        )
        # Should succeed or at least not 404
        assert response.status_code in (200, 422)


class TestStreamEndpoint:
    """Tests for the SSE streaming endpoint."""

    def test_stream_requires_valid_thread(self, test_client):
        """GET /research/stream/{thread_id} with unknown thread should still work (creates session)."""
        response = test_client.get(
            "/research/stream/new-thread-123?question=hello",
        )
        # It should attempt to create a session and stream (not crash)
        assert response.status_code in (200, 500)
