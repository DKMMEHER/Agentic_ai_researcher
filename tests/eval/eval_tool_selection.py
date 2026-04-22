"""Tool Selection Accuracy Evaluation Script.

Feeds predefined test questions to the AI agent and measures
whether it selects the correct tool for each query.

Usage:
    python -m tests.eval.eval_tool_selection
    python -m tests.eval.eval_tool_selection --max-cases 5
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure the project src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_test_cases(
    path: Path | None = None, max_cases: int | None = None
) -> list[dict]:
    """Load test cases from JSON file."""
    if path is None:
        path = Path(__file__).parent / "test_cases.json"
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    if max_cases:
        cases = cases[:max_cases]
    return cases


async def get_first_tool_call(
    graph, config, question: str
) -> tuple[str | None, str | None]:
    """Run the agent on a single question and capture the FIRST tool it tries to call.

    Now awareness of the Supervisor pattern:
    1. The first event will likely be the Supervisor deciding intent.
    2. We continue streaming until we hit a node that calls a tool (usually Researcher).
    """
    import asyncio

    input_data = {"messages": [{"role": "user", "content": question}]}
    classified_intent = None
    tool_found = None

    try:
        # Use astream and manage termination for LangSmith cleanliness
        streamer = graph.astream(input_data, config, stream_mode="updates")
        try:
            async for event in streamer:
                # Check for supervisor result to log the intent
                if "supervisor" in event:
                    classified_intent = event["supervisor"].get("intent")
                    print(f"         [Supervisor] Intent: {classified_intent}")

                # Check any node's output for tool calls (usually 'researcher')
                for _node_name, output in event.items():
                    if isinstance(output, dict) and "messages" in output:
                        last_msg = output["messages"][-1]
                        if getattr(last_msg, "tool_calls", None):
                            tool_found = last_msg.tool_calls[0]["name"]
                            break

                if tool_found:
                    break
        finally:
            # Explicitly close to prevent CancelledError on early return
            await streamer.aclose()

    except (asyncio.CancelledError, GeneratorExit):
        # This is expected when we stop early and call aclose()
        pass
    except Exception as e:
        print(f"    [!] Agent error: {e}")
        return None, classified_intent

    return tool_found, classified_intent


async def run_evaluation(test_cases: list[dict]) -> dict:
    """Run the full evaluation suite and return results."""
    from ai_researcher.agent.graph import build_graph
    from ai_researcher.logging import setup_logging

    setup_logging()

    graph, config = build_graph(thread_id="eval-tool-selection")

    results = []
    correct = 0
    total = len(test_cases)

    print("\n" + "=" * 70)
    print("  [EVAL] TOOL SELECTION ACCURACY EVALUATION")
    print("=" * 70)
    print(f"  Running {total} test cases...\n")

    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        expected = case["expected_tool"]
        category = case["category"]

        print(f"  [{i}/{total}] {question[:60]}...")

        start = time.time()
        actual_tool, intent = await get_first_tool_call(graph, config, question)
        elapsed = round(time.time() - start, 2)

        is_correct = actual_tool == expected
        if is_correct:
            correct += 1
            status = "[OK] PASS"
        else:
            status = "[FAIL] FAIL"

        print(f"         Expected: {expected}")
        print(f"         Got:      {actual_tool or 'NO TOOL CALLED'}")
        print(f"         Result:   {status}  ({elapsed}s)\n")

        results.append(
            {
                "id": case["id"],
                "question": question,
                "category": category,
                "expected_tool": expected,
                "actual_tool": actual_tool,
                "intent": intent,
                "correct": is_correct,
                "elapsed_seconds": elapsed,
            }
        )

        # Pause to avoid Groq free-tier rate limits (6,000 tokens/min)
        time.sleep(8)

    accuracy = round((correct / total) * 100, 1) if total > 0 else 0

    # Category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy_percent": accuracy,
        "category_breakdown": {
            cat: {
                "accuracy": round((v["correct"] / v["total"]) * 100, 1),
                "correct": v["correct"],
                "total": v["total"],
            }
            for cat, v in categories.items()
        },
        "results": results,
    }

    return summary


def print_summary(summary: dict):
    """Print a formatted evaluation summary."""
    print("\n" + "=" * 70)
    print("  [STATS] EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n  Overall Accuracy: {summary['accuracy_percent']}%")
    print(f"  Correct: {summary['correct']} / {summary['total_cases']}")
    print(f"  Incorrect: {summary['incorrect']} / {summary['total_cases']}")

    print("\n  Category Breakdown:")
    print("  " + "-" * 50)
    for cat, data in summary["category_breakdown"].items():
        bar = "█" * int(data["accuracy"] / 10)
        print(
            f"    {cat:<20} {data['accuracy']:>5}%  {bar}  ({data['correct']}/{data['total']})"
        )

    # Show failures
    failures = [r for r in summary["results"] if not r["correct"]]
    if failures:
        print(f"\n  [FAIL] Failed Cases ({len(failures)}):")
        print("  " + "-" * 50)
        for f in failures:
            print(f"    Q: {f['question'][:55]}...")
            print(f"       Expected: {f['expected_tool']}  |  Got: {f['actual_tool']}")
    else:
        print("\n  [SUCCESS] Perfect Score! All tools selected correctly!")

    print("\n" + "=" * 70)


def save_report(summary: dict):
    """Save the evaluation report to a JSON file."""
    reports_dir = PROJECT_ROOT / "tests" / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"tool_selection_{timestamp}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  📁 Report saved to: {report_path}\n")
    return report_path


def main():
    import asyncio

    parser = argparse.ArgumentParser(description="Evaluate tool selection accuracy")
    parser.add_argument(
        "--max-cases", type=int, default=None, help="Limit number of test cases to run"
    )
    parser.add_argument(
        "--test-file", type=str, default=None, help="Path to custom test_cases.json"
    )
    args = parser.parse_args()

    test_path = Path(args.test_file) if args.test_file else None
    test_cases = load_test_cases(path=test_path, max_cases=args.max_cases)

    summary = asyncio.run(run_evaluation(test_cases))
    print_summary(summary)
    save_report(summary)


if __name__ == "__main__":
    main()
