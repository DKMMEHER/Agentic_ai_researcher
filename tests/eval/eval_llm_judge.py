"""LLM-as-a-Judge — Automated Output Quality Grading.

Runs a research task through the agent, then feeds the output
to a SECOND LLM call that grades it on a structured rubric.

Uses Gemini or Groq as the judge model.

Usage:
    python -m tests.eval.eval_llm_judge
    python -m tests.eval.eval_llm_judge --judge-provider gemini
    python -m tests.eval.eval_llm_judge --judge-provider groq --judge-model llama-3.1-8b-instant
    python -m tests.eval.eval_llm_judge --max-cases 2
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ----- JUDGE RUBRIC PROMPT -----

JUDGE_SYSTEM_PROMPT = """You are a strict academic paper reviewer and grading assistant.
You will receive a RESEARCH TASK and the AI AGENT'S RESPONSE.
Grade the response on the following 5 rubric dimensions.

GRADING RUBRIC (score each 1-5):

1. ACCURACY (1-5):
   - Are the stated facts correct and verifiable?
   - Are paper titles, authors, and dates accurate?
   - 1 = Major factual errors, 5 = All facts verifiable

2. COMPLETENESS (1-5):
   - Does the response cover all aspects of the topic?
   - Are key concepts, methods, and findings mentioned?
   - 1 = Very shallow, 5 = Thorough and comprehensive

3. COHERENCE (1-5):
   - Is the writing logically structured and easy to follow?
   - Are transitions smooth? Is there a clear flow?
   - 1 = Disorganized, 5 = Excellent structure

4. CITATION_QUALITY (1-5):
   - Are sources properly referenced?
   - Are paper titles, URLs, or DOIs included?
   - 1 = No citations, 5 = Well-cited with links

5. DEPTH (1-5):
   - Does the response show deep understanding?
   - Are technical details, equations, or methodologies explained?
   - 1 = Surface-level, 5 = Expert-level depth

IMPORTANT: You MUST respond with ONLY a valid JSON object in this exact format:
{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "coherence": <1-5>,
  "citation_quality": <1-5>,
  "depth": <1-5>,
  "justification": "<Brief 2-3 sentence explanation>"
}

Do NOT include any text before or after the JSON. Do NOT wrap in markdown code fences."""


def load_judge_cases(path=None, max_cases=None):
    """Load judge test cases."""
    if path is None:
        path = Path(__file__).parent / "judge_test_cases.json"
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if max_cases:
        cases = cases[:max_cases]
    return cases


def get_agent_response(graph, config, system_prompt, task):
    """Run the agent on a task and collect the final response text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    input_data = {"messages": messages}

    final_response = ""
    try:
        for event in graph.stream(input_data, config, stream_mode="values"):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
                if not getattr(last_msg, "tool_call_id", None):  # Skip tool messages
                    final_response = last_msg.content
    except Exception as e:
        final_response = f"ERROR: Agent failed with: {e}"

    return final_response


def judge_response(judge_llm, task, response):
    """Send the agent's output to the judge LLM for grading."""
    judge_input = f"""RESEARCH TASK:
{task}

AI AGENT'S RESPONSE:
{response[:4000]}"""  # Limit to avoid token overflow

    try:
        result = judge_llm.invoke([
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_input},
        ])

        raw = result.content.strip()

        # Clean markdown fences if present
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        scores = json.loads(raw)
        return scores

    except json.JSONDecodeError:
        return {
            "accuracy": 0, "completeness": 0, "coherence": 0,
            "citation_quality": 0, "depth": 0,
            "justification": f"Judge returned invalid JSON: {raw[:200]}"
        }
    except Exception as e:
        return {
            "accuracy": 0, "completeness": 0, "coherence": 0,
            "citation_quality": 0, "depth": 0,
            "justification": f"Judge error: {e!s}"
        }


def compute_average(scores):
    """Compute average score across all rubric dimensions."""
    dims = ["accuracy", "completeness", "coherence", "citation_quality", "depth"]
    values = [scores.get(d, 0) for d in dims]
    return round(sum(values) / len(values), 2)


def create_judge_llm(provider, model_name):
    """Create the judge LLM based on provider choice."""
    settings = get_settings()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=settings.google_api_key,
        )
    else:  # groq
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.groq_api_key,
            model_name=model_name,
            temperature=0,
        )


def run_evaluation(cases, judge_provider="gemini", judge_model=None):
    """Run the full LLM-as-a-Judge evaluation pipeline."""
    from ai_researcher.agent.graph import build_graph
    from ai_researcher.agent.prompts import load_prompt
    from ai_researcher.config import get_settings
    from ai_researcher.logging import setup_logging

    setup_logging()
    settings = get_settings()

    # Pick default model per provider
    if judge_model is None:
        judge_model = "gemini-2.0-flash" if judge_provider == "gemini" else "llama-3.3-70b-versatile"

    # Agent (generates the response)
    graph, config = build_graph(thread_id="eval-judge")
    agent_prompt = load_prompt()

    # Judge (grades the response) — separate LLM instance
    judge_llm = create_judge_llm(judge_provider, judge_model)

    results = []

    print("\n" + "=" * 70)
    print("  ⚖️  LLM-AS-A-JUDGE — AUTOMATED OUTPUT GRADING")
    print("=" * 70)
    print(f"  Judge: {judge_provider} / {judge_model}")
    print(f"  Test Cases: {len(cases)}\n")

    for i, case in enumerate(cases, 1):
        task = case["task"]
        category = case["category"]

        print(f"  [{i}/{len(cases)}] {task[:55]}...")
        print(f"         Category: {category}")

        # Step 1: Get agent response
        print(f"         🤖 Generating response...")
        start = time.time()
        response = get_agent_response(graph, config, agent_prompt, task)
        gen_time = round(time.time() - start, 2)
        print(f"         ✅ Response generated ({gen_time}s, {len(response)} chars)")

        # Pause for rate limits
        time.sleep(5)

        # Step 2: Judge the response
        print(f"         ⚖️  Judging...")
        scores = judge_response(judge_llm, task, response)
        avg = compute_average(scores)

        status = "✅" if avg >= case.get("min_expected_score", 3.0) else "❌"
        print(f"         {status} Score: {avg}/5.0")
        print(f"            A:{scores.get('accuracy',0)} C:{scores.get('completeness',0)} "
              f"H:{scores.get('coherence',0)} Q:{scores.get('citation_quality',0)} D:{scores.get('depth',0)}")
        if scores.get("justification"):
            print(f"            💬 {scores['justification'][:80]}...")
        print()

        results.append({
            "id": case["id"],
            "task": task,
            "category": category,
            "response_length": len(response),
            "response_preview": response[:300],
            "scores": scores,
            "average_score": avg,
            "passed": avg >= case.get("min_expected_score", 3.0),
            "gen_time_seconds": gen_time,
        })

        # Rate limit protection
        time.sleep(8)

    return results


def print_summary(results):
    """Print formatted summary."""
    total = len(results)
    avg_total = sum(r["average_score"] for r in results) / total if total else 0
    passed = sum(1 for r in results if r["passed"])

    print("=" * 70)
    print("  📊 LLM-AS-A-JUDGE RESULTS")
    print("=" * 70)
    print(f"\n  Overall Average: {avg_total:.1f}/5.0")
    print(f"  Pass Rate: {passed}/{total} ({passed/total*100:.0f}%)" if total else "")

    # Per-dimension averages
    dims = ["accuracy", "completeness", "coherence", "citation_quality", "depth"]
    print("\n  Per-Dimension Averages:")
    print("  " + "-" * 50)
    for d in dims:
        avg_d = sum(r["scores"].get(d, 0) for r in results) / total if total else 0
        bar = "█" * int(avg_d)
        print(f"    {d:<20} {avg_d:.1f}/5  {bar}")

    print("\n  Per-Task Scores:")
    print("  " + "-" * 50)
    for r in results:
        status = "✅" if r["passed"] else "❌"
        print(f"    {status} {r['average_score']:.1f}/5  | {r['task'][:50]}...")

    print("\n" + "=" * 70)


def save_report(results):
    """Save report."""
    reports_dir = PROJECT_ROOT / "tests" / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"judge_{timestamp}.json"

    total = len(results)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": total,
        "average_score": round(sum(r["average_score"] for r in results) / total, 2) if total else 0,
        "pass_rate": round(sum(1 for r in results if r["passed"]) / total, 3) if total else 0,
        "results": results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  📁 Report saved to: {report_path}\n")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge evaluation")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit test cases")
    parser.add_argument("--judge-provider", type=str, default="gemini",
                        choices=["gemini", "groq"],
                        help="Which LLM provider to use as judge (default: gemini)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Model name (default: gemini-2.0-flash or llama-3.3-70b-versatile)")
    args = parser.parse_args()

    cases = load_judge_cases(max_cases=args.max_cases)
    results = run_evaluation(cases, judge_provider=args.judge_provider, judge_model=args.judge_model)
    print_summary(results)
    save_report(results)


if __name__ == "__main__":
    main()
