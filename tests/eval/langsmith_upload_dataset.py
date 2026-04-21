"""Upload test cases to LangSmith as a Dataset.

This script creates a LangSmith dataset from test_cases.json so that
evaluations run against the cloud and results are tracked as experiments.

Usage:
    python -m tests.eval.langsmith_upload_dataset
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Push env vars before importing langsmith
from ai_researcher.config import get_settings
get_settings()

from langsmith import Client


DATASET_NAME = "Agentic_ai_researcher"
DATASET_DESCRIPTION = "25 curated test questions to evaluate whether the AI agent selects the correct tool for each research query."


def main():
    client = Client()

    # Load test cases
    test_cases_path = Path(__file__).parent / "test_cases.json"
    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Check if dataset already exists
    existing_datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing_datasets:
        print(f"⚠️  Dataset '{DATASET_NAME}' already exists. Deleting and re-creating...")
        client.delete_dataset(dataset_id=existing_datasets[0].id)

    # Create new dataset
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
    )
    print(f"✅ Created dataset: '{DATASET_NAME}' (ID: {dataset.id})")

    # Upload each test case as an example
    for case in test_cases:
        client.create_example(
            dataset_id=dataset.id,
            inputs={
                "question": case["question"],
                "category": case["category"],
            },
            outputs={
                "expected_tool": case["expected_tool"],
            },
        )

    print(f"✅ Uploaded {len(test_cases)} test cases to LangSmith!")
    print(f"🔗 View at: https://smith.langchain.com/datasets")


if __name__ == "__main__":
    main()
