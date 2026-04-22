"""LangSmith-native LLM-as-a-Judge Evaluation.

Runs research tasks through the agent, then grades the output
using a second Groq LLM call. Results are tracked as experiments.

Usage:
    python -m tests.eval.langsmith_evaluate_judge
    python -m tests.eval.langsmith_evaluate_judge --judge-provider gemini
    python -m tests.eval.langsmith_evaluate_judge --experiment-name "v2-improved-prompt"
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_researcher.config import get_settings  # noqa: E402

get_settings()

from langsmith import Client, evaluate  # noqa: E402

DATASET_NAME = "Agentic_ai_researcher_LLM Judge"

JUDGE_SYSTEM_PROMPT = """You are a strict academic paper reviewer.
Grade the AI's response on these 5 dimensions (1-5 each):

1. ACCURACY: Are facts correct and verifiable?
2. COMPLETENESS: Does it cover all aspects?
3. COHERENCE: Is the writing logically structured?
4. CITATION_QUALITY: Are sources properly referenced?
5. DEPTH: Does it show deep technical understanding?

Respond with ONLY valid JSON:
{"accuracy": <1-5>, "completeness": <1-5>, "coherence": <1-5>, "citation_quality": <1-5>, "depth": <1-5>, "justification": "<brief explanation>"}

No markdown fences. No extra text."""

_AGENT_CACHE = None
_JUDGE_LLM = None


def create_agent_and_judge(judge_provider="gemini", judge_model=None):
    """Initialize both the agent and the judge LLM."""
    from ai_researcher.agent.graph import build_graph
    from ai_researcher.agent.prompts import load_prompt
    from ai_researcher.config import get_settings
    from ai_researcher.logging import setup_logging

    setup_logging()
    settings = get_settings()

    if judge_model is None:
        judge_model = (
            "gemini-2.0-flash"
            if judge_provider == "gemini"
            else "llama-3.3-70b-versatile"
        )

    graph, config = build_graph(thread_id="langsmith-judge-eval")
    system_prompt = load_prompt()

    if judge_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        judge_llm = ChatGoogleGenerativeAI(
            model=judge_model,
            temperature=0,
            google_api_key=settings.google_api_key,
        )
    else:
        from langchain_groq import ChatGroq

        judge_llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=judge_model,
            temperature=0,
        )

    return graph, config, system_prompt, judge_llm


def agent_target(inputs: dict) -> dict:
    """Run the agent, then judge the output."""
    graph, config, system_prompt, judge_llm = _AGENT_CACHE

    task = inputs["task"]

    # Step 1: Get agent response
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    final_response = ""
    try:
        for event in graph.stream({"messages": messages}, config, stream_mode="values"):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and isinstance(last_msg.content, str):  # noqa: SIM102
                if not getattr(last_msg, "tool_call_id", None):
                    final_response = last_msg.content
    except Exception as e:
        final_response = f"ERROR: {e}"

    time.sleep(5)

    # Step 2: Judge the response
    judge_input = (
        f"RESEARCH TASK:\n{task}\n\nAI AGENT'S RESPONSE:\n{final_response[:4000]}"
    )

    try:
        result = judge_llm.invoke(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_input},
            ]
        )
        raw = result.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        scores = json.loads(raw)
    except Exception:
        scores = {
            "accuracy": 0,
            "completeness": 0,
            "coherence": 0,
            "citation_quality": 0,
            "depth": 0,
        }

    dims = ["accuracy", "completeness", "coherence", "citation_quality", "depth"]
    avg = round(sum(scores.get(d, 0) for d in dims) / len(dims), 2)

    time.sleep(8)

    return {
        "response_preview": final_response[:500],
        "accuracy": scores.get("accuracy", 0),
        "completeness": scores.get("completeness", 0),
        "coherence": scores.get("coherence", 0),
        "citation_quality": scores.get("citation_quality", 0),
        "depth": scores.get("depth", 0),
        "average_score": avg,
        "justification": scores.get("justification", ""),
    }


# --- EVALUATORS ---


def accuracy_evaluator(run, example) -> dict:
    return {"key": "accuracy", "score": run.outputs.get("accuracy", 0) / 5.0}


def completeness_evaluator(run, example) -> dict:
    return {"key": "completeness", "score": run.outputs.get("completeness", 0) / 5.0}


def coherence_evaluator(run, example) -> dict:
    return {"key": "coherence", "score": run.outputs.get("coherence", 0) / 5.0}


def citation_quality_evaluator(run, example) -> dict:
    return {
        "key": "citation_quality",
        "score": run.outputs.get("citation_quality", 0) / 5.0,
    }


def depth_evaluator(run, example) -> dict:
    return {"key": "depth", "score": run.outputs.get("depth", 0) / 5.0}


def overall_evaluator(run, example) -> dict:
    avg = run.outputs.get("average_score", 0)
    min_score = example.outputs.get("min_expected_score", 3.0)
    return {
        "key": "overall_pass",
        "score": 1.0 if avg >= min_score else 0.0,
        "comment": f"Score: {avg}/5.0 (min: {min_score})",
    }


def main():
    global _AGENT_CACHE

    parser = argparse.ArgumentParser(
        description="Run LangSmith LLM-as-a-Judge evaluation"
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--judge-provider", type=str, default="gemini", choices=["gemini", "groq"]
    )
    parser.add_argument("--judge-model", type=str, default=None)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ⚖️  LangSmith LLM-as-a-Judge Evaluation")
    print("=" * 60)

    print(f"  Initializing agent and judge ({args.judge_provider})...")
    graph, config, system_prompt, judge_llm = create_agent_and_judge(
        judge_provider=args.judge_provider, judge_model=args.judge_model
    )
    _AGENT_CACHE = (graph, config, system_prompt, judge_llm)
    print("  ✅ Ready!\n")

    client = Client()
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if not datasets:
        print(f"  ❌ Dataset '{DATASET_NAME}' not found!")
        print("  Run first: python -m tests.eval.langsmith_upload_judge")
        return

    exp_name = args.experiment_name or "llm-judge"
    print(f"  📦 Dataset: {DATASET_NAME}")
    print(f"  🔬 Experiment: {exp_name}")
    print("  ⏳ Running (agent + judge for each task)...\n")

    evaluate(
        agent_target,
        data=DATASET_NAME,
        evaluators=[
            accuracy_evaluator,
            completeness_evaluator,
            coherence_evaluator,
            citation_quality_evaluator,
            depth_evaluator,
            overall_evaluator,
        ],
        experiment_prefix=exp_name,
        max_concurrency=1,
    )

    print("\n" + "=" * 60)
    print("  ✅ LLM-as-a-Judge evaluation complete!")
    print("  🔗 View results at: https://smith.langchain.com/datasets")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
