"""Upload judge test cases to LangSmith as a Dataset.

Usage:
    python -m tests.eval.langsmith_upload_judge
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_researcher.config import get_settings
get_settings()

from langsmith import Client

DATASET_NAME = "Agentic_ai_researcher_LLM Judge"
DATASET_DESCRIPTION = "5 research tasks for LLM-as-a-Judge automated output grading."


def main():
    client = Client()

    test_path = Path(__file__).parent / "judge_test_cases.json"
    with open(test_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        print(f"⚠️  Dataset '{DATASET_NAME}' already exists. Deleting and re-creating...")
        client.delete_dataset(dataset_id=existing[0].id)

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
    )
    print(f"✅ Created dataset: '{DATASET_NAME}' (ID: {dataset.id})")

    for case in cases:
        client.create_example(
            dataset_id=dataset.id,
            inputs={
                "task": case["task"],
                "category": case["category"],
            },
            outputs={
                "min_expected_score": case["min_expected_score"],
            },
        )

    print(f"✅ Uploaded {len(cases)} judge test cases to LangSmith!")
    print(f"🔗 View at: https://smith.langchain.com/datasets")


if __name__ == "__main__":
    main()
