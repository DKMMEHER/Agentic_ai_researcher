"""Tests for the ResearchClient UI backend client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_researcher.ui.client import ResearchClient


@pytest.mark.asyncio
async def test_start_research():
    """Test start_research sends POST and returns thread_id."""
    client = ResearchClient("http://testserver")

    mock_response = MagicMock()
    mock_response.json.return_value = {"thread_id": "test-thread-123"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        thread_id = await client.start_research("test question")

        assert thread_id == "test-thread-123"
        args, kwargs = mock_post.call_args
        assert args[0] == "http://testserver/research/start"
        assert kwargs["json"]["question"] == "test question"


@pytest.mark.asyncio
async def test_submit_action():
    """Test submit_action sends POST with correct payload."""
    client = ResearchClient("http://testserver")

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "ok"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        await client.submit_action("thread-1", "revise", "make it better")

        args, kwargs = mock_post.call_args
        assert args[0] == "http://testserver/research/action"
        assert kwargs["json"]["action"] == "revise"
        assert kwargs["json"]["instructions"] == "make it better"


@pytest.mark.asyncio
async def test_stream_research_parsing():
    """Test stream_research correctly parses SSE events."""
    client = ResearchClient("http://testserver")

    # Mock SSE stream data
    mock_lines = [
        b"event: status",
        b'data: {"agent": "researcher"}',
        b"",
        b"event: token",
        b'data: {"content": "Hello"}',
        b"",
        b"event: done",
        b'data: {"status": "complete"}',
        b"",
    ]

    async def mock_aiter_lines():
        for line in mock_lines:
            yield line.decode("utf-8")

    mock_response = AsyncMock()
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.raise_for_status = MagicMock()

    # Mock the context manager for stream()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_response

    with patch("httpx.AsyncClient.stream", return_value=mock_ctx):
        events = []
        async for event in client.stream_research("thread-1"):
            events.append(event)

        assert len(events) == 3
        assert events[0]["event"] == "status"
        assert events[0]["data"]["agent"] == "researcher"
        assert events[1]["event"] == "token"
        assert events[1]["data"]["content"] == "Hello"
        assert events[2]["event"] == "done"
