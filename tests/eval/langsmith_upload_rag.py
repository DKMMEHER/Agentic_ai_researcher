"""Upload RAG test cases to LangSmith as a Dataset.

Usage:
    python -m tests.eval.langsmith_upload_rag
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_researcher.config import get_settings  # noqa: E402

get_settings()

from langsmith import Client  # noqa: E402

DATASET_NAME = "Agentic_ai_researcher_RAG Evaluation"
DATASET_DESCRIPTION = (
    "12 questions across 3 landmark papers to evaluate ChromaDB retrieval quality."
)


def main():
    client = Client()

    test_cases_path = Path(__file__).parent / "rag_test_cases.json"
    with open(test_cases_path, encoding="utf-8") as f:
        papers = json.load(f)

    # Delete if exists
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        print(
            f"⚠️  Dataset '{DATASET_NAME}' already exists. Deleting and re-creating..."
        )
        client.delete_dataset(dataset_id=existing[0].id)

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
    )
    print(f"✅ Created dataset: '{DATASET_NAME}' (ID: {dataset.id})")

    count = 0
    for paper in papers:
        for q in paper["questions"]:
            client.create_example(
                dataset_id=dataset.id,
                inputs={
                    "pdf_url": paper["pdf_url"],
                    "paper_title": paper["paper_title"],
                    "question": q["question"],
                },
                outputs={
                    "ground_truth": q["ground_truth"],
                },
            )
            count += 1

    print(f"✅ Uploaded {count} RAG test cases to LangSmith!")
    print("🔗 View at: https://smith.langchain.com/datasets")


if __name__ == "__main__":
    main()
