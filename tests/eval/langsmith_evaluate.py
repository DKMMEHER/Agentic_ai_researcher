"""LangSmith-native Tool Selection Evaluation.

Runs the agent against the LangSmith dataset and uploads results
as a tracked experiment with side-by-side comparison support.

Usage:
    python -m tests.eval.langsmith_evaluate
    python -m tests.eval.langsmith_evaluate --experiment-name "v2-decision-tree-prompt"
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Push env vars before importing anything
from ai_researcher.config import get_settings  # noqa: E402

get_settings()

from langsmith import Client, evaluate  # noqa: E402

DATASET_NAME = "Agentic_ai_researcher"


def create_agent():
    """Create a fresh agent instance for evaluation."""
    from ai_researcher.agent.graph import build_graph
    from ai_researcher.agent.prompts import load_prompt
    from ai_researcher.logging import setup_logging

    setup_logging()
    graph, config = build_graph(thread_id="langsmith-eval")
    system_prompt = load_prompt()
    return graph, config, system_prompt


def agent_target(inputs: dict) -> dict:
    """Target function that LangSmith calls for each test case.

    Takes a question, runs it through the agent, and returns
    the first tool the agent tries to call.
    """
    graph, config, system_prompt = _AGENT_CACHE

    question = inputs["question"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    input_data = {"messages": messages}

    actual_tool = None
    try:
        for event in graph.stream(input_data, config, stream_mode="values"):
            last_msg = event["messages"][-1]
            if getattr(last_msg, "tool_calls", None):
                actual_tool = last_msg.tool_calls[0]["name"]
                break
    except Exception as e:
        actual_tool = f"ERROR: {e!s}"

    # Rate limit protection for Groq free tier
    time.sleep(8)

    return {"actual_tool": actual_tool or "NO_TOOL_CALLED"}


def tool_selection_evaluator(run, example) -> dict:
    """Custom evaluator that checks if the agent selected the correct tool.

    This function is called by LangSmith for each test case after
    the agent has produced its output. It compares actual vs expected.
    """
    expected = example.outputs["expected_tool"]
    actual = run.outputs.get("actual_tool", "NO_TOOL_CALLED")

    is_correct = actual == expected
    category = example.inputs.get("category", "unknown")

    return {
        "key": "tool_selection_correct",
        "score": 1.0 if is_correct else 0.0,
        "comment": f"Category: {category} | Expected: {expected} | Got: {actual}",
    }


def main():
    global _AGENT_CACHE

    parser = argparse.ArgumentParser(
        description="Run LangSmith tool selection evaluation"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (e.g. 'v2-decision-tree-prompt'). Auto-generated if not provided.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🧪 LangSmith Tool Selection Evaluation")
    print("=" * 60)

    # Initialize agent once and cache it
    print("  Initializing agent...")
    _AGENT_CACHE = create_agent()
    print("  ✅ Agent ready!\n")

    # Verify dataset exists
    client = Client()
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if not datasets:
        print(f"  ❌ Dataset '{DATASET_NAME}' not found on LangSmith!")
        print("  Run this first: python -m tests.eval.langsmith_upload_dataset")
        return

    print(f"  📦 Dataset: {DATASET_NAME}")
    print(f"  🔬 Experiment: {args.experiment_name or '(auto-generated)'}")
    print("  ⏳ Running evaluation (this takes a few minutes)...\n")

    # Run the evaluation
    evaluate(
        agent_target,
        data=DATASET_NAME,
        evaluators=[tool_selection_evaluator],
        experiment_prefix=args.experiment_name or "tool-selection",
        max_concurrency=1,  # Sequential to avoid Groq rate limits
    )

    print("\n" + "=" * 60)
    print("  ✅ Evaluation complete!")
    print("  🔗 View results at: https://smith.langchain.com/datasets")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
