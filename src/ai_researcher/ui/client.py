"""Client for communicating with the FastAPI research backend."""

import json
from collections.abc import AsyncGenerator

import httpx


class ResearchClient:
    """Handles communication with the FastAPI backend."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def start_research(self, question: str, thread_id: str | None = None) -> str:
        """Initialize a research thread and return the thread_id."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/research/start",
                json={"question": question, "thread_id": thread_id},
            )
            response.raise_for_status()
            return response.json()["thread_id"]

    async def stream_research(
        self, thread_id: str, question: str | None = None
    ) -> AsyncGenerator:
        """Stream events from the backend using SSE."""
        import urllib.parse

        url = f"{self.base_url}/research/stream/{thread_id}"
        if question:
            url += f"?question={urllib.parse.quote(question)}"

        async with httpx.AsyncClient(timeout=None) as client:  # noqa: SIM117
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                # SSE lines are like "event: token\ndata: {...}\n\n"
                current_event = None
                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        current_event = line[len("event: ") :].strip()
                    elif line.startswith("data: ") and current_event:
                        raw_data = line[len("data: ") :].strip()
                        try:
                            data = json.loads(raw_data)
                            yield {"event": current_event, "data": data}
                        except json.JSONDecodeError:
                            print(
                                f"[CLIENT ERROR] Received invalid JSON in 'data:' line: {raw_data}"
                            )
                            # Fallback if valid but non-JSON data is sent
                            yield {"event": current_event, "data": {"raw": raw_data}}
                        current_event = None

    async def submit_action(
        self, thread_id: str, action: str, instructions: str | None = None
    ):
        """Submit a HITL action (approve/revise/abort)."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/research/action",
                json={
                    "thread_id": thread_id,
                    "action": action,
                    "instructions": instructions,
                },
            )
            response.raise_for_status()
            return response.json()
