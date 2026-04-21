"""Upload trajectory test cases to LangSmith as a Dataset.

Usage:
    python -m tests.eval.langsmith_upload_trajectory
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_researcher.config import get_settings
get_settings()

from langsmith import Client


DATASET_NAME = "Agentic_ai_researcher_Trajectory Evaluation"
DATASET_DESCRIPTION = "8 multi-step tasks to evaluate agent loop efficiency and tool chain accuracy."


def main():
    client = Client()

    test_cases_path = Path(__file__).parent / "trajectory_cases.json"
    with open(test_cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Delete if exists
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
                "max_steps": case["max_steps"],
            },
            outputs={
                "optimal_trajectory": case["optimal_trajectory"],
                "optimal_steps": case["optimal_steps"],
            },
        )

    print(f"✅ Uploaded {len(cases)} trajectory cases to LangSmith!")
    print(f"🔗 View at: https://smith.langchain.com/datasets")


if __name__ == "__main__":
    main()
