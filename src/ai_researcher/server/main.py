"""FastAPI backend for the AI Researcher agent."""

import json
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    SystemMessage,
)
from sse_starlette.sse import EventSourceResponse

from ai_researcher.agent.graph import build_graph
from ai_researcher.config import get_settings
from ai_researcher.logging import setup_logging
from ai_researcher.models.api_schemas import ActionRequest, ResearchRequest

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: initialize the SQLite checkpointer on startup and close on shutdown.

    AsyncSqliteSaver.from_conn_string() returns an async context manager that MUST
    be entered before use. Using 'async with' here ensures the underlying aiosqlite
    connection is properly opened, kept alive for all requests, and cleanly closed
    on server shutdown.
    """
    settings = get_settings()

    if settings.checkpoint_backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            logger.info(
                "[Lifespan] Initializing AsyncSqliteSaver checkpointer (db=%s)",
                settings.checkpoint_db_url,
            )
            async with AsyncSqliteSaver.from_conn_string(
                settings.checkpoint_db_url
            ) as checkpointer:
                app.state.checkpointer = checkpointer
                logger.info("[Lifespan] SQLite checkpointer ready ✅")
                yield
        except ImportError:
            logger.warning(
                "[Lifespan] langgraph-checkpoint-sqlite not found, falling back to MemorySaver"
            )
            from langgraph.checkpoint.memory import MemorySaver

            app.state.checkpointer = MemorySaver()
            yield

    else:
        # "memory" or unknown backend
        from langgraph.checkpoint.memory import MemorySaver

        app.state.checkpointer = MemorySaver()
        logger.info(
            "[Lifespan] Using MemorySaver checkpointer (in-memory, not persistent)"
        )
        yield

    logger.info("[Lifespan] Checkpointer shut down.")


app = FastAPI(title="AI Researcher API", lifespan=lifespan)

# Setup CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your UI URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run and load balancers."""
    return {"status": "ok"}


# Global store for active graph instances (shared compiled graph per thread)
active_sessions = {}


def get_session(request: Request, thread_id: str):
    """Retrieve or create a graph session for a thread.

    Uses the app-level checkpointer (properly initialized in lifespan)
    so all sessions share one persistent SQLite connection.
    """
    if thread_id not in active_sessions:
        checkpointer = request.app.state.checkpointer
        graph, config = build_graph(thread_id=thread_id, checkpointer=checkpointer)
        active_sessions[thread_id] = (graph, config)
    return active_sessions[thread_id]


@app.post("/research/start")
async def start_research(request: Request, body: ResearchRequest):
    """Initializes a research thread."""
    thread_id = body.thread_id or os.urandom(8).hex()
    # Ensure graph is ready
    get_session(request, thread_id)
    return {"thread_id": thread_id, "status": "initialized"}


@app.get("/research/stream/{thread_id}")
async def stream_research(
    thread_id: str, request: Request, question: str | None = None
):
    """Streams research events for a specific thread."""
    # Auto-initialize session if it doesn't exist
    graph, config = get_session(request, thread_id)

    # If a question is provided, it's a new input to the graph.
    # NOTE: We only send the user message here. Each graph node (supervisor,
    # researcher, writer) prepends its own system prompt before calling the LLM.
    # Injecting a system prompt here would pollute the checkpointed message history
    # and confuse the supervisor's intent classification.
    input_data = None
    if question:
        input_data = {
            "messages": [
                {"role": "user", "content": question},
            ]
        }

    async def event_generator() -> AsyncGenerator:
        astream = graph.astream(input_data, config, stream_mode=["messages", "values"])
        try:
            async for mode, payload in astream:
                if mode == "messages":
                    msg, _metadata = payload
                    if isinstance(msg, AIMessageChunk):  # noqa: SIM102
                        if msg.content:
                            # Skip raw structured output from the supervisor node.
                            # The supervisor uses with_structured_output(), which emits
                            # JSON like {"intent": "...", "chat_response": "..."} as chunks.
                            # The clean response is emitted separately via the "values" stream.
                            node = _metadata.get("langgraph_node", "")
                            if node == "supervisor":
                                continue

                            yield {
                                "event": "token",
                                "data": json.dumps(
                                    {
                                        "content": str(msg.content),
                                        "id": getattr(msg, "id", None),
                                    }
                                ),
                            }
                elif mode == "values":
                    v_state = payload
                    agent = v_state.get("current_agent", "supervisor")

                    logger.info(f"[STREAM] Agent transition: {agent}")

                    # Send status update
                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {
                                "agent": agent,
                            }
                        ),
                    }

                    # Check for guardrail/system messages in the latest message
                    messages = v_state.get("messages", [])
                    if messages:
                        last_msg = messages[-1]

                        # Emit supervisor's direct_chat response (full AIMessage, not chunked)
                        if (
                            isinstance(last_msg, AIMessage)
                            and last_msg.content
                            and agent == "supervisor"
                            and v_state.get("intent") == "direct_chat"
                        ):
                            logger.info(
                                "[STREAM] Emitting supervisor direct_chat response"
                            )
                            yield {
                                "event": "token",
                                "data": json.dumps(
                                    {
                                        "content": str(last_msg.content),
                                        "id": getattr(last_msg, "id", None),
                                    }
                                ),
                            }

                        # Emit guardrail SystemMessage content so the UI can display it
                        if isinstance(last_msg, SystemMessage) and last_msg.content:  # noqa: SIM102
                            if "GUARDRAIL" in str(last_msg.content):
                                logger.warning("[STREAM] Guardrail message detected")
                                yield {
                                    "event": "guardrail",
                                    "data": json.dumps(
                                        {"content": str(last_msg.content)}
                                    ),
                                }

                        # Extract Token Telemetry if present in AIMessage
                        if isinstance(last_msg, AIMessage) and getattr(
                            last_msg, "usage_metadata", None
                        ):
                            usage = last_msg.usage_metadata
                            try:
                                # Handle both dict-like and object-like access
                                if isinstance(usage, dict):
                                    in_t = usage.get("input_tokens", 0)
                                    out_t = usage.get("output_tokens", 0)
                                else:
                                    in_t = getattr(usage, "input_tokens", 0)
                                    out_t = getattr(usage, "output_tokens", 0)

                                yield {
                                    "event": "telemetry",
                                    "data": json.dumps(
                                        {
                                            "input_tokens": in_t,
                                            "output_tokens": out_t,
                                        }
                                    ),
                                }
                            except Exception as te:
                                logger.error(
                                    f"[STREAM] Telemetry extraction failed but skipping crash: {te}"
                                )

                        # Check for tool calls in the latest message
                        if getattr(last_msg, "tool_calls", None):
                            t_names = [tc["name"] for tc in last_msg.tool_calls]
                            logger.info(f"[STREAM] Tool calls: {t_names}")
                            yield {
                                "event": "status",
                                "data": json.dumps({"tool_calls": t_names}),
                            }

                # Check if we hit an interrupt (must use async method with AsyncSqliteSaver)
                state_snapshot = await graph.aget_state(config)
                if state_snapshot.next and "human_review" in state_snapshot.next:
                    logger.info("[STREAM] Hit human_review interrupt")
                    yield {
                        "event": "status",
                        "data": json.dumps({"interrupt": "human_review"}),
                    }

            yield {"event": "done", "data": json.dumps({"status": "complete"})}

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield {"event": "error", "data": json.dumps({"raw": str(e)})}

        finally:
            # Always cleanly close the upstream astream when the SSE connection ends
            # (client disconnect, GeneratorExit, or normal completion).
            # This prevents the 'GeneratorExit' traceback from LangGraph internals.
            await astream.aclose()

    return EventSourceResponse(event_generator())


@app.post("/research/action")
async def handle_action(request: Request, body: ActionRequest):
    """Handles HITL actions (approve/revise/abort)."""
    if body.thread_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    graph, config = active_sessions[body.thread_id]

    update = {"human_approval": body.action}
    if body.action == "revise" and body.instructions:
        update["revision_instructions"] = body.instructions  # type: ignore

    # Must use async method with AsyncSqliteSaver
    await graph.aupdate_state(config, update)  # type: ignore
    return {"status": "action_recorded", "next_step": "ready_to_stream"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ai_researcher.server.main:app", host="0.0.0.0", port=8000, reload=True)
