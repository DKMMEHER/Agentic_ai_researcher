"""Trajectory Evaluation — Agent Loop Efficiency Scorer.

Runs multi-step tasks through the agent and measures:
  - Total tool calls vs optimal trajectory length
  - Error count and recovery behavior
  - Efficiency score per task

Usage:
    python -m tests.eval.eval_trajectory
    python -m tests.eval.eval_trajectory --max-cases 3
"""

import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_trajectory_cases(path=None, max_cases=None):
    """Load trajectory test cases from JSON."""
    if path is None:
        path = Path(__file__).parent / "trajectory_cases.json"
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if max_cases:
        cases = cases[:max_cases]
    return cases


def run_agent_trajectory(graph, config, system_prompt, task, max_steps):
    """Run the agent on a task and capture the full tool call trajectory.
    
    Returns a dict with:
      - trajectory: ordered list of tool names called
      - errors: list of any errors encountered
      - steps: total number of tool calls
    """
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

            # Track tool calls
            if getattr(last_msg, "tool_calls", None):
                for tc in last_msg.tool_calls:
                    trajectory.append(tc["name"])
                    step_count += 1

            # Track tool errors
            if hasattr(last_msg, "tool_call_id") and hasattr(last_msg, "content"):
                content = str(last_msg.content)
                if "error" in content.lower() or "failed" in content.lower():
                    errors.append({
                        "tool": getattr(last_msg, "name", "unknown"),
                        "error": content[:200],
                    })

            # Safety: stop if agent loops too many times
            if step_count >= max_steps:
                errors.append({"tool": "SYSTEM", "error": f"Hit max step limit ({max_steps})"})
                break

    except Exception as e:
        errors.append({"tool": "SYSTEM", "error": str(e)[:200]})

    return {
        "trajectory": trajectory,
        "errors": errors,
        "steps": step_count,
    }


def score_trajectory(actual, optimal_trajectory, optimal_steps):
    """Score a trajectory against the optimal path.
    
    Returns:
      - efficiency_score: 0.0 to 1.0 (1.0 = perfect match)
      - details: breakdown of the scoring
    """
    actual_steps = actual["steps"]
    actual_trajectory = actual["trajectory"]
    error_count = len(actual["errors"])

    # 1. Step Efficiency: optimal_steps / actual_steps (capped at 1.0)
    if actual_steps == 0:
        step_score = 0.0
    else:
        step_score = min(optimal_steps / actual_steps, 1.0)

    # 2. Tool Accuracy: how many of the optimal tools appeared in the trajectory?
    optimal_set = set(optimal_trajectory)
    actual_set = set(actual_trajectory)
    if optimal_set:
        tool_overlap = len(optimal_set & actual_set) / len(optimal_set)
    else:
        tool_overlap = 1.0

    # 3. Error Penalty: -0.15 per error
    error_penalty = min(error_count * 0.15, 0.5)

    # 4. Wasted Calls: tools called that aren't in the optimal set
    wasted_calls = len([t for t in actual_trajectory if t not in optimal_set])
    waste_penalty = min(wasted_calls * 0.1, 0.3)

    # Final composite score
    efficiency_score = max(
        (step_score * 0.4) + (tool_overlap * 0.4) - error_penalty - waste_penalty,
        0.0,
    )

    return {
        "efficiency_score": round(efficiency_score, 3),
        "step_score": round(step_score, 3),
        "tool_overlap": round(tool_overlap, 3),
        "error_penalty": round(error_penalty, 3),
        "waste_penalty": round(waste_penalty, 3),
        "actual_steps": actual_steps,
        "optimal_steps": optimal_steps,
        "wasted_calls": wasted_calls,
        "error_count": error_count,
    }


def run_evaluation(cases):
    """Run the full trajectory evaluation."""
    from ai_researcher.agent.graph import build_graph
    from ai_researcher.agent.prompts import load_prompt
    from ai_researcher.logging import setup_logging

    setup_logging()
    graph, config = build_graph(thread_id="eval-trajectory")
    system_prompt = load_prompt()

    results = []

    print("\n" + "=" * 70)
    print("  🔄 TRAJECTORY EVALUATION — AGENT LOOP EFFICIENCY")
    print("=" * 70)
    print(f"  Running {len(cases)} trajectory tests...\n")

    for i, case in enumerate(cases, 1):
        task = case["task"]
        optimal = case["optimal_trajectory"]
        optimal_steps = case["optimal_steps"]
        max_steps = case["max_steps"]

        print(f"  [{i}/{len(cases)}] {task[:60]}...")
        print(f"         Optimal: {' → '.join(optimal)} ({optimal_steps} steps)")

        start = time.time()
        actual = run_agent_trajectory(graph, config, system_prompt, task, max_steps)
        elapsed = round(time.time() - start, 2)

        scores = score_trajectory(actual, optimal, optimal_steps)

        traj_str = " → ".join(actual["trajectory"]) if actual["trajectory"] else "NO TOOLS CALLED"
        print(f"         Actual:  {traj_str} ({actual['steps']} steps)")
        print(f"         Score:   {scores['efficiency_score']:.1%}  |  Errors: {scores['error_count']}  ({elapsed}s)")

        if actual["errors"]:
            for err in actual["errors"]:
                print(f"         ⚠️  [{err['tool']}] {err['error'][:80]}")
        print()

        results.append({
            "id": case["id"],
            "task": task,
            "category": case["category"],
            "optimal_trajectory": optimal,
            "actual_trajectory": actual["trajectory"],
            "errors": actual["errors"],
            "scores": scores,
            "elapsed_seconds": elapsed,
        })

        # Rate limit protection
        time.sleep(10)

    return results


def print_summary(results):
    """Print formatted summary."""
    total = len(results)
    avg_efficiency = sum(r["scores"]["efficiency_score"] for r in results) / total if total else 0
    total_errors = sum(r["scores"]["error_count"] for r in results)
    total_wasted = sum(r["scores"]["wasted_calls"] for r in results)

    print("=" * 70)
    print("  📊 TRAJECTORY EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n  Average Efficiency: {avg_efficiency:.1%}")
    print(f"  Total Errors: {total_errors}")
    print(f"  Total Wasted Tool Calls: {total_wasted}")

    print("\n  Per-Task Breakdown:")
    print("  " + "-" * 60)
    for r in results:
        s = r["scores"]
        bar = "█" * int(s["efficiency_score"] * 10)
        status = "✅" if s["efficiency_score"] >= 0.7 else "⚠️" if s["efficiency_score"] >= 0.4 else "❌"
        print(f"    {status} [{s['actual_steps']}/{s['optimal_steps']} steps]  {s['efficiency_score']:>5.1%}  {bar}")
        print(f"       {r['task'][:55]}...")

    # Error Recovery Analysis
    error_cases = [r for r in results if r["errors"]]
    if error_cases:
        print(f"\n  🔧 Error Recovery Analysis ({len(error_cases)} tasks had errors):")
        print("  " + "-" * 60)
        for r in error_cases:
            for err in r["errors"]:
                recovered = r["scores"]["efficiency_score"] > 0.3
                status = "🔄 Recovered" if recovered else "💥 Failed"
                print(f"    {status} | [{err['tool']}] {err['error'][:60]}")
    else:
        print("\n  🎉 No errors encountered! Perfect error-free execution.")

    print("\n" + "=" * 70)


def save_report(results):
    """Save report to JSON."""
    reports_dir = PROJECT_ROOT / "tests" / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"trajectory_{timestamp}.json"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(results),
        "avg_efficiency": round(
            sum(r["scores"]["efficiency_score"] for r in results) / len(results), 3
        ) if results else 0,
        "results": results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  📁 Report saved to: {report_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent trajectory efficiency")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit test cases")
    args = parser.parse_args()

    cases = load_trajectory_cases(max_cases=args.max_cases)
    results = run_evaluation(cases)
    print_summary(results)
    save_report(results)


if __name__ == "__main__":
    main()
