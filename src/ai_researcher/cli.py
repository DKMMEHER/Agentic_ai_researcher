"""Command-line interface for the AI Researcher application.

Provides two modes:
    - cli:  Interactive terminal chat with the research agent
    - ui:   Launch the Streamlit web interface
"""

import argparse
import subprocess
import sys
from pathlib import Path

from ai_researcher.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _run_cli():
    """Run the interactive CLI chat loop."""
    from ai_researcher.agent.graph import build_graph
    from ai_researcher.agent.prompts import load_prompt

    graph, config = build_graph()
    system_prompt = load_prompt()

    print("\n🔬 AI Research Agent — Interactive Mode")
    print("=" * 50)
    print("Type your message and press Enter. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! 👋")
            break

        messages = [
            {"role": "user", "content": user_input},
        ]
        input_data = {"messages": messages}

        try:
            for s in graph.stream(input_data, config, stream_mode="values"):
                message = s["messages"][-1]
                if hasattr(message, "content") and message.content:
                    content = (
                        message.content
                        if isinstance(message.content, str)
                        else str(message.content)
                    )
                    # Only print AI responses, not tool messages
                    if hasattr(message, "tool_calls") or not hasattr(
                        message, "tool_call_id"
                    ):
                        message.pretty_print()

            # --- Human-in-the-Loop: check for review interrupt ---
            state_snapshot = graph.get_state(config)
            while state_snapshot.next and "human_review" in state_snapshot.next:
                print("\n" + "=" * 50)
                print("📋 RESEARCH COMPLETE — Your Review Required")
                print("=" * 50)
                print("  [a] Approve & write paper")
                print("  [r] Revise (provide new instructions)")
                print("  [x] Abort")

                try:
                    choice = input("\nYour decision: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = "x"

                if choice in ("a", "approve", "y", "yes"):
                    graph.update_state(config, {"human_approval": "approved"})
                elif choice in ("r", "revise"):
                    try:
                        instructions = input("Revision instructions: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        instructions = ""
                    graph.update_state(config, {
                        "human_approval": "revise",
                        "revision_instructions": instructions or "Please do more research.",
                    })
                elif choice in ("x", "abort", "n", "no"):
                    graph.update_state(config, {"human_approval": "abort"})
                    for _ in graph.stream(None, config, stream_mode="values"):
                        pass
                    print("\n🛑 Research aborted.")
                    break
                else:
                    print("Invalid choice. Please try again.")
                    continue

                # Resume the graph and stream output
                for s in graph.stream(None, config, stream_mode="values"):
                    message = s["messages"][-1]
                    if hasattr(message, "content") and message.content:
                        if hasattr(message, "tool_calls") or not hasattr(
                            message, "tool_call_id"
                        ):
                            message.pretty_print()

                # Check for another interrupt (revision cycle)
                state_snapshot = graph.get_state(config)

        except Exception as e:
            logger.exception("Error during agent execution")
            print(f"\n❌ Error: {e}\n")


def _run_server(reload: bool = False):
    """Launch the FastAPI backend server."""
    import uvicorn
    print("\n🚀 Launching FastAPI Backend...")
    print(f"   Host: 0.0.0.0, Port: 8000, Reload: {reload}\n")
    
    # Run the server using uvicorn
    uvicorn.run(
        "ai_researcher.server.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=reload
    )


def _run_ui():
    """Launch the Streamlit web interface."""
    app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    print("\n🚀 Launching Streamlit UI...")
    print(f"   App: {app_path}\n")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="ai-researcher",
        description="🔬 AI Research Agent — Search, analyze, and write research papers",
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "ui", "server"],
        default="cli",
        help="Run mode: 'cli' (terminal), 'ui' (web app), or 'server' (backend API)",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Conversation thread ID for state persistence",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload server on code changes (only for --mode server)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override the logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)
    
    if args.mode == "server":
        _run_server(reload=args.reload)
    elif args.mode == "ui":
        _run_ui()
    else:
        logger.info("Starting AI Researcher in '%s' mode", args.mode)
        _run_cli()


if __name__ == "__main__":
    main()
