"""LangSmith-native Trajectory Evaluation.

Runs multi-step tasks against the LangSmith dataset and scores
the agent's tool chain efficiency as a tracked experiment.

Usage:
    python -m tests.eval.langsmith_evaluate_trajectory
    python -m tests.eval.langsmith_evaluate_trajectory --experiment-name "v2-improved-routing"
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_researcher.config import get_settings  # noqa: E402

get_settings()

from langsmith import Client, evaluate  # noqa: E402

DATASET_NAME = "Agentic_ai_researcher_Trajectory Evaluation"


def create_agent():
    """Create a fresh agent instance."""
    from ai_researcher.agent.graph import build_graph
    from ai_researcher.agent.prompts import load_prompt
    from ai_researcher.logging import setup_logging

    setup_logging()
    graph, config = build_graph(thread_id="langsmith-trajectory-eval")
    system_prompt = load_prompt()
    return graph, config, system_prompt


def agent_target(inputs: dict) -> dict:
    """Target function: runs the agent and captures the full tool trajectory."""
    graph, config, system_prompt = _AGENT_CACHE

    task = inputs["task"]
    max_steps = inputs.get("max_steps", 10)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    input_data = {"messages": messages}

    trajectory = []
    errors = []
    step_count = 0

    try:
        for event in graph.stream(input_data, config, stream_mode="values"):
            last_msg = event["messages"][-1]

            if getattr(last_msg, "tool_calls", None):
                for tc in last_msg.tool_calls:
                    trajectory.append(tc["name"])
                    step_count += 1

            if hasattr(last_msg, "tool_call_id") and hasattr(last_msg, "content"):
                content = str(last_msg.content)
                if "error" in content.lower() or "failed" in content.lower():
                    errors.append(content[:100])

            if step_count >= max_steps:
                errors.append(f"Hit max step limit ({max_steps})")
                break

    except Exception as e:
        errors.append(str(e)[:200])

    # Rate limit protection
    time.sleep(10)

    return {
        "trajectory": trajectory,
        "actual_steps": step_count,
        "error_count": len(errors),
        "errors": errors,
    }


# --- EVALUATORS ---


def step_efficiency_evaluator(run, example) -> dict:
    """Scores how close the agent's step count is to optimal."""
    optimal_steps = example.outputs["optimal_steps"]
    actual_steps = run.outputs.get("actual_steps", 0)

    score = 0.0 if actual_steps == 0 else min(optimal_steps / actual_steps, 1.0)

    return {
        "key": "step_efficiency",
        "score": round(score, 3),
        "comment": f"Optimal: {optimal_steps} steps | Actual: {actual_steps} steps",
    }


def tool_overlap_evaluator(run, example) -> dict:
    """Scores whether the agent used the correct tools."""
    optimal = set(example.outputs["optimal_trajectory"])
    actual = set(run.outputs.get("trajectory", []))

    score = 1.0 if not optimal else len(optimal & actual) / len(optimal)

    return {
        "key": "tool_overlap",
        "score": round(score, 3),
        "comment": f"Expected tools: {optimal} | Used: {actual}",
    }


def error_penalty_evaluator(run, example) -> dict:
    """Penalizes errors encountered during execution."""
    error_count = run.outputs.get("error_count", 0)
    score = max(1.0 - (error_count * 0.25), 0.0)

    return {
        "key": "error_free",
        "score": round(score, 3),
        "comment": f"{error_count} error(s) encountered",
    }


def wasted_calls_evaluator(run, example) -> dict:
    """Penalizes tool calls not in the optimal trajectory."""
    optimal_set = set(example.outputs["optimal_trajectory"])
    actual_trajectory = run.outputs.get("trajectory", [])
    wasted = [t for t in actual_trajectory if t not in optimal_set]
    score = max(1.0 - (len(wasted) * 0.2), 0.0)

    return {
        "key": "no_wasted_calls",
        "score": round(score, 3),
        "comment": f"{len(wasted)} wasted call(s): {wasted}"
        if wasted
        else "No wasted calls",
    }


def main():
    global _AGENT_CACHE

    parser = argparse.ArgumentParser(description="Run LangSmith trajectory evaluation")
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🔄 LangSmith Trajectory Evaluation")
    print("=" * 60)

    print("  Initializing agent...")
    _AGENT_CACHE = create_agent()
    print("  ✅ Agent ready!\n")

    client = Client()
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if not datasets:
        print(f"  ❌ Dataset '{DATASET_NAME}' not found!")
        print("  Run first: python -m tests.eval.langsmith_upload_trajectory")
        return

    print(f"  📦 Dataset: {DATASET_NAME}")
    print(f"  🔬 Experiment: {args.experiment_name or '(auto-generated)'}")
    print("  ⏳ Running (this takes several minutes with rate limiting)...\n")

    evaluate(
        agent_target,
        data=DATASET_NAME,
        evaluators=[
            step_efficiency_evaluator,
            tool_overlap_evaluator,
            error_penalty_evaluator,
            wasted_calls_evaluator,
        ],
        experiment_prefix=args.experiment_name or "trajectory",
        max_concurrency=1,
    )

    print("\n" + "=" * 60)
    print("  ✅ Trajectory evaluation complete!")
    print("  🔗 View results at: https://smith.langchain.com/datasets")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
