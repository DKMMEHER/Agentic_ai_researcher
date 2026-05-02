"""Streamlit-based frontend for the AI Researcher agent.

Run with:
    streamlit run src/ai_researcher/ui/streamlit_app.py
    or
    ai-researcher --mode ui
"""

import asyncio
import os
import re
import uuid
from pathlib import Path

import streamlit as st
from langsmith import Client
from streamlit_feedback import streamlit_feedback  # type: ignore

# Must be the first Streamlit command
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _initialize_agent():
    """Initialize the API client and config in session state."""
    # Try to load session ID from URL query params (survives refresh)
    if "session_id" not in st.session_state:
        if "thread" in st.query_params:
            st.session_state.session_id = st.query_params["thread"]
        else:
            new_id = str(uuid.uuid4())
            st.session_state.session_id = new_id
            st.query_params["thread"] = new_id

    if "client" not in st.session_state:
        from ai_researcher.logging import setup_logging
        from ai_researcher.ui.client import ResearchClient

        setup_logging()

        # Initialize the API client pointing to the backend
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        st.session_state.client = ResearchClient(base_url=backend_url)
        st.session_state.langsmith_client = Client()


def _scan_for_pdfs(message_content) -> None:
    """Helper to detect and register PDF files found in message content."""
    if not message_content:
        return

    content_str = str(message_content)

    # Direct cleanup check (handles if the string IS the path, potentially with quotes/newlines)
    clean_path = content_str.strip("'\" \n\r\t")
    if clean_path.endswith(".pdf") and Path(clean_path).exists():  # noqa: SIM102
        if clean_path not in st.session_state.pdf_paths:
            st.session_state.pdf_paths.append(clean_path)

    # Regex check to extract paths embedded within regular text/responses
    # Matches Windows (C:\...) and Unix (/...) style absolute paths without spaces
    matches = re.findall(
        r"([A-Za-z]:\\[^\s*<>\|]+?\.pdf|/[^\s*<>\|]+?\.pdf)", content_str
    )
    for match in matches:
        clean_match = match.strip("'\" \n\r\t.,")
        if Path(clean_match).exists() and clean_match not in st.session_state.pdf_paths:
            st.session_state.pdf_paths.append(clean_match)


def _submit_feedback(feedback, run_id):
    """Callback to send feedback to LangSmith."""
    if feedback and run_id:
        # LangGraph message IDs use format "run--{uuid}-{index}"
        # LangSmith expects a plain UUID, so extract it
        run_id_str = str(run_id)
        if run_id_str.startswith("run--"):
            # Extract UUID from "run--xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx-N"
            parts = run_id_str[5:]  # remove "run--"
            # UUID is 36 chars (8-4-4-4-12), take just that part
            run_id_str = parts[:36]

        try:
            score = 1.0 if feedback["score"] == "👍" else 0.0
            st.session_state.langsmith_client.create_feedback(
                run_id=run_id_str,
                key="user_satisfaction",
                score=score,
                comment=feedback.get("text", ""),
            )
            st.toast("Feedback submitted! Thank you.", icon="✅")
        except Exception as e:
            st.toast(f"Could not submit feedback: {e}", icon="⚠️")


def _initialize_session():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_paths" not in st.session_state:
        st.session_state.pdf_paths = []
    if "awaiting_approval" not in st.session_state:
        st.session_state.awaiting_approval = False
    if "show_revision_input" not in st.session_state:
        st.session_state.show_revision_input = False
    if "total_input_tokens" not in st.session_state:
        st.session_state.total_input_tokens = 0
    if "total_output_tokens" not in st.session_state:
        st.session_state.total_output_tokens = 0
    if "tool_counts" not in st.session_state:
        st.session_state.tool_counts = {}


def _render_sidebar():
    """Render the sidebar with app info and controls."""
    with st.sidebar:
        st.title("🔬 AI Researcher")
        st.markdown("---")
        st.markdown(
            """
            - 🌐 **Web Knowledge**: Tavily Search & Wikipedia
            - 📚 **Academic Search**: arXiv, PubMed, Google Scholar
            - 🕸️ **Citation Graph**: Semantic Scholar (Snowballing)
            - 📖 **PDF Engine**: Figure Extraction & Vector Retrieval
            - 🗜️ **Map-Reduce**: Long Document Summarization
            - 🎥 **Media Processing**: YouTube Transcripts
            - 🧠 **Memory**: Persistent Agentic Scratchpad
            - 📝 **Typesetting**: Autonomous LaTeX Compilation
            """
        )
        st.markdown("---")

        st.markdown("### 📊 Telemetry & Cost")
        with st.expander("Session Metrics", expanded=True):
            in_t = st.session_state.total_input_tokens
            out_t = st.session_state.total_output_tokens
            # Assume Gemini 2.5 Flash pricing: $0.075 / 1M Input, $0.30 / 1M Output
            cost_in = (in_t / 1_000_000) * 0.075
            cost_out = (out_t / 1_000_000) * 0.30
            total_cost_usd = cost_in + cost_out
            total_cost_inr = (
                total_cost_usd * 83.50
            )  # Conversion rate: 1 USD = 83.50 INR

            st.metric(label="Input Tokens", value=f"{in_t:,}")
            st.metric(label="Output Tokens", value=f"{out_t:,}")
            st.metric(label="Est. Cost (Gemini)", value=f"₹{total_cost_inr:.4f}")

            st.markdown("**🔧 Tools Invoked:**")
            if not st.session_state.tool_counts:
                st.caption("No tools invoked yet.")
            else:
                for tool_name, count in st.session_state.tool_counts.items():
                    st.text(f"  {tool_name}: {count}x")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Screen", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.pdf_paths = []
                st.session_state.awaiting_approval = False
                st.session_state.show_revision_input = False
                st.rerun()

        with col2:
            if st.button("🔥 Wipe Memory", use_container_width=True, type="primary"):
                st.session_state.chat_history = []
                st.session_state.pdf_paths = []
                st.session_state.awaiting_approval = False
                st.session_state.show_revision_input = False
                st.session_state.pop("resume_decision", None)
                st.session_state.pop("revision_instructions", None)
                st.session_state.total_input_tokens = 0
                st.session_state.total_output_tokens = 0
                st.session_state.tool_counts = {}
                # Rotate the thread ID to completely reset agent memory on the backend
                new_id = str(uuid.uuid4())
                st.session_state.session_id = new_id
                st.query_params["thread"] = new_id
                st.rerun()

        # Show generated PDFs
        if st.session_state.pdf_paths:
            st.markdown("### 📄 Generated Papers")
            for pdf_path in st.session_state.pdf_paths:
                path = Path(pdf_path)
                if path.exists():
                    with open(path, "rb") as f:
                        st.download_button(
                            label=f"📥 {path.name}",
                            data=f.read(),
                            file_name=path.name,
                            mime="application/pdf",
                            use_container_width=True,
                        )


def _render_chat_history():
    """Render all messages in the chat history."""
    for idx, msg in enumerate(st.session_state.chat_history):
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
        elif role == "system":
            if "GUARDRAIL" in content or "ERROR" in content:
                st.error(content)
            else:
                st.success(content)
        elif role == "tool":
            with st.chat_message("assistant", avatar="🔧"):
                st.caption(f"Tool: {msg.get('tool_name', 'unknown')}")
                st.code(content[:500] + ("..." if len(content) > 500 else ""))

        # Display feedback buttons for AI messages that have a known LangSmith Run ID
        if role == "assistant" and msg.get("run_id"):
            streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Why did you give this rating?",
                key=f"fb_{msg['run_id']}_{idx}",
                on_submit=_submit_feedback,
                kwargs={"run_id": msg["run_id"]},
            )


def _render_approval_ui():
    """Render the human-in-the-loop approval interface."""
    st.markdown("---")
    st.markdown("### 📋 Research Complete — Your Review Required")
    st.info(
        "The Researcher has finished gathering information. "
        "Review the summary above, then choose how to proceed."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(
            "✅ Approve & Write Paper",
            key="hitl_approve",
            use_container_width=True,
        ):
            st.session_state.resume_decision = "approved"
            st.rerun()
    with col2:
        if st.button(
            "🔄 Request Revision",
            key="hitl_revise",
            use_container_width=True,
        ):
            st.session_state.show_revision_input = True
            st.rerun()
    with col3:
        if st.button(
            "❌ Abort",
            key="hitl_abort",
            use_container_width=True,
        ):
            st.session_state.resume_decision = "abort"
            st.rerun()

    if st.session_state.get("show_revision_input"):
        revision_text = st.text_area(
            "What should the researcher focus on?",
            placeholder="e.g., Find more recent papers on transformer architectures...",
            key="revision_text_input",
        )
        if st.button(
            "📨 Submit Revision",
            key="hitl_submit_revision",
            use_container_width=True,
        ):
            st.session_state.resume_decision = "revise"
            st.session_state.revision_instructions = (
                revision_text or "Please do more research."
            )
            st.session_state.show_revision_input = False
            st.rerun()


def _resume_graph():
    """Resume the agent graph after human review via the FastAPI backend."""
    decision = st.session_state.pop("resume_decision", None)
    revision_instructions = st.session_state.pop("revision_instructions", "")

    if not decision:
        return

    client = st.session_state.client
    thread_id = st.session_state.session_id

    # Record current PDF count to detect if a new one is generated
    initial_pdf_count = len(st.session_state.pdf_paths)

    # --- Handle abort or submit action to backend ---
    try:
        asyncio.run(client.submit_action(thread_id, decision, revision_instructions))
    except Exception as e:
        st.error(f"❌ Failed to submit action to backend: {e}")
        return

    if decision == "abort":
        st.session_state.awaiting_approval = False
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": "🛑 Research aborted by user.",
                "run_id": None,
            }
        )
        st.rerun()
        return

    spinner_text = (
        "Initializing Writer Agent..."
        if decision == "approved"
        else "Initializing Researcher Agent..."
    )
    with st.chat_message("assistant", avatar="🤖"):
        status_container = st.status(spinner_text, expanded=True)
        full_response = ""
        run_id = None
        response_placeholder = st.empty()

        try:

            async def _run_stream():
                nonlocal full_response, run_id
                async for event in client.stream_research(thread_id):
                    event_type = event["event"]
                    data = event["data"]

                    if event_type == "token":
                        content = data.get("content", "")
                        if content:
                            if not run_id:
                                run_id = data.get("id")
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                    elif event_type == "status":
                        agent_name = data.get("agent", "researcher")
                        status_label = f"🟢 {agent_name.capitalize()}: Working..."
                        status_container.update(label=status_label, state="running")

                        if "tool_calls" in data:
                            for tc in data["tool_calls"]:
                                status_container.markdown(
                                    f"**🔧 Executing tool:** `{tc}`"
                                )

                        if data.get("interrupt") == "human_review":
                            st.session_state.awaiting_approval = True
                    elif event_type == "telemetry":
                        st.session_state.total_input_tokens += data.get(
                            "input_tokens", 0
                        )
                        st.session_state.total_output_tokens += data.get(
                            "output_tokens", 0
                        )
                    elif event_type == "tool_calls":
                        for tc_name in data.get("tools", []):
                            st.session_state.tool_counts[tc_name] = (
                                st.session_state.tool_counts.get(tc_name, 0) + 1
                            )
                    elif event_type == "error":
                        status_container.error(f"Backend Error: {data}")

            asyncio.run(_run_stream())
            response_placeholder.markdown(full_response)
            _scan_for_pdfs(full_response)
            status_container.update(
                label="✅ Task Complete", state="complete", expanded=False
            )

        except Exception as e:
            st.error(f"❌ An error occurred during streaming: {e}")
            full_response = f"I encountered an error: {e}"

            # FALLBACK: If no streaming chunks were received, pull the last message from the final state
            if not full_response:
                full_response = (
                    "I encountered an issue connecting to the backend state."
                )
                response_placeholder.markdown(full_response)

            # Final safety scan on the entire accumulated text
            _scan_for_pdfs(full_response)
            status_container.update(
                label="✅ Task Complete", state="complete", expanded=False
            )

        except Exception as e:  # noqa: B025
            st.error(f"❌ An error occurred: {e}")
            full_response = f"I encountered an error: {e}"

    if full_response:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_response, "run_id": run_id}
        )

    # If a new PDF was generated, notify the user explicitly
    if len(st.session_state.pdf_paths) > initial_pdf_count:
        new_pdf = st.session_state.pdf_paths[-1]
        msg = f"✨ **PDF Document Successfully Generated!**\n\nIt is available for download in the sidebar or at: `{new_pdf}`"
        st.session_state.chat_history.append({"role": "system", "content": msg})
        st.toast("PDF Generated Successfully!", icon="📄")

    # The backend handles interrupts and will send an interrupt status event
    # We now trust the stream event handler to set `st.session_state.awaiting_approval` appropriately.
    # Fallback message polling is removed since the UI cannot access the persistence graph directly.

    st.rerun()


def _process_user_input(user_input: str):
    """Process user input through the FastAPI backend and display results."""
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    client = st.session_state.client
    thread_id = st.session_state.session_id

    # Stream the agent response from the backend
    with st.chat_message("assistant", avatar="🤖"):
        status_container = st.status("Initializing Supervisor Agent...", expanded=True)
        full_response = ""
        run_id = None
        response_placeholder = st.empty()

        # Record current PDF count to detect if a new one is generated
        initial_pdf_count = len(st.session_state.pdf_paths)

        try:

            async def _run_stream():
                nonlocal full_response, run_id
                async for event in client.stream_research(
                    thread_id, question=user_input
                ):
                    event_type = event["event"]
                    data = event["data"]

                    if event_type == "token":
                        content = data.get("content", "")
                        if content:
                            if not run_id:
                                run_id = data.get("id")
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                    elif event_type == "status":
                        agent_name = data.get("agent", "researcher")
                        status_label = f"🟢 {agent_name.capitalize()}: Working..."
                        status_container.update(label=status_label, state="running")

                        if "tool_calls" in data:
                            for tc in data["tool_calls"]:
                                status_container.markdown(
                                    f"**🔧 Executing tool:** `{tc}`"
                                )

                        if data.get("interrupt") == "human_review":
                            st.session_state.awaiting_approval = True
                    elif event_type == "telemetry":
                        st.session_state.total_input_tokens += data.get(
                            "input_tokens", 0
                        )
                        st.session_state.total_output_tokens += data.get(
                            "output_tokens", 0
                        )
                    elif event_type == "tool_calls":
                        for tc_name in data.get("tools", []):
                            st.session_state.tool_counts[tc_name] = (
                                st.session_state.tool_counts.get(tc_name, 0) + 1
                            )
                    elif event_type == "error":
                        status_container.error(f"Backend Error: {data}")

                response_placeholder.markdown(full_response)

            asyncio.run(_run_stream())
            _scan_for_pdfs(full_response)
            status_container.update(
                label="✅ Task Complete", state="complete", expanded=False
            )

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            full_response = f"I encountered an error: {e}"

    # Save assistant response to history
    if full_response:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_response, "run_id": run_id}
        )

    # If a new PDF was generated, notify the user explicitly
    if len(st.session_state.pdf_paths) > initial_pdf_count:
        new_pdf = st.session_state.pdf_paths[-1]
        msg = f"✨ **PDF Document Successfully Generated!**\n\nIt is available for download in the sidebar or at: `{new_pdf}`"
        st.session_state.chat_history.append({"role": "system", "content": msg})
        st.toast("PDF Generated Successfully!", icon="📄")

    # Force a rerun to show the feedback/approval buttons
    st.rerun()


def main():
    """Main entry point for the Streamlit app."""
    _initialize_agent()
    _initialize_session()
    _render_sidebar()
    _render_chat_history()

    # Handle pending resume from human review button click
    if st.session_state.get("resume_decision"):
        _resume_graph()
        return

    # Show approval UI if awaiting human review
    if st.session_state.get("awaiting_approval"):
        _render_approval_ui()
        return

    # Normal chat input
    user_input = st.chat_input("What research topic would you like to explore?")
    if user_input:
        _process_user_input(user_input)


if __name__ == "__main__":
    main()
