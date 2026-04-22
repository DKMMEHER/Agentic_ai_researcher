"""LangSmith-native RAG Evaluation.

Ingests PDFs into ChromaDB, queries them, and scores retrieval quality
as a tracked experiment on LangSmith with side-by-side comparison.

Usage:
    python -m tests.eval.langsmith_evaluate_rag
    python -m tests.eval.langsmith_evaluate_rag --experiment-name "chunk-1000-overlap-200"
    python -m tests.eval.langsmith_evaluate_rag --chunk-size 500 --chunk-overlap 100 --experiment-name "chunk-500-overlap-100"
"""

import argparse
import hashlib
import io
import sys
from pathlib import Path

import PyPDF2
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_researcher.config import get_settings  # noqa: E402

get_settings()

from langsmith import Client, evaluate  # noqa: E402

DATASET_NAME = "Agentic_ai_researcher_RAG Evaluation"

# Configurable chunking params (set by CLI args)
_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200

# Cache for ingested PDFs to avoid re-downloading
_INGESTED_PDFS = set()
_VECTOR_STORE = None


def get_store():
    """Lazy-load the vector store."""
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        from ai_researcher.tools.db import get_vector_store

        _VECTOR_STORE = get_vector_store()
    return _VECTOR_STORE


def ensure_pdf_ingested(url):
    """Download and ingest a PDF if not already done in this session."""
    if url in _INGESTED_PDFS:
        return

    print(f"    📥 Ingesting: {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    pdf_file = io.BytesIO(response.content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    text_parts = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    full_text = "\n".join(text_parts).strip()
    doc_id = hashlib.md5(url.encode()).hexdigest()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_text(full_text)

    docs = [
        Document(page_content=chunk, metadata={"source": url, "doc_id": doc_id})
        for chunk in chunks
    ]

    store = get_store()
    store.add_documents(docs)
    _INGESTED_PDFS.add(url)
    print(
        f"    ✅ Ingested {len(chunks)} chunks (size={_CHUNK_SIZE}, overlap={_CHUNK_OVERLAP})"
    )


def rag_target(inputs: dict) -> dict:
    """Target function: ingest PDF if needed, query ChromaDB, return results."""
    url = inputs["pdf_url"]
    question = inputs["question"]

    # Ensure PDF is ingested
    ensure_pdf_ingested(url)

    # Query ChromaDB
    store = get_store()
    results = store.similarity_search_with_relevance_scores(
        question, k=4, filter={"source": url}
    )

    contexts = [doc.page_content for doc, _score in results]
    scores = [round(score, 4) for _doc, score in results]

    return {
        "contexts": contexts,
        "relevance_scores": scores,
        "num_results": len(results),
    }


# --- EVALUATORS ---


def context_precision_evaluator(run, example) -> dict:
    """Measures what fraction of retrieved chunks are relevant to the ground truth."""
    ground_truth = example.outputs["ground_truth"]
    contexts = run.outputs.get("contexts", [])

    if not contexts:
        return {
            "key": "context_precision",
            "score": 0.0,
            "comment": "No chunks retrieved",
        }

    gt_keywords = set(ground_truth.lower().split())
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "were",
        "are",
        "with",
        "and",
        "or",
        "of",
        "in",
        "to",
        "for",
        "on",
        "at",
        "by",
    }
    gt_keywords = gt_keywords - stop_words

    relevant = 0
    for ctx in contexts:
        ctx_lower = ctx.lower()
        hits = sum(1 for kw in gt_keywords if kw in ctx_lower)
        if hits / max(len(gt_keywords), 1) > 0.3:
            relevant += 1

    score = relevant / len(contexts)
    return {
        "key": "context_precision",
        "score": round(score, 3),
        "comment": f"{relevant}/{len(contexts)} chunks relevant",
    }


def context_recall_evaluator(run, example) -> dict:
    """Measures what fraction of ground truth keywords appear in retrieved chunks."""
    ground_truth = example.outputs["ground_truth"]
    contexts = run.outputs.get("contexts", [])

    gt_keywords = set(ground_truth.lower().split())
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "were",
        "are",
        "with",
        "and",
        "or",
        "of",
        "in",
        "to",
        "for",
        "on",
        "at",
        "by",
    }
    gt_keywords = gt_keywords - stop_words

    all_text = " ".join(contexts).lower()
    found = sum(1 for kw in gt_keywords if kw in all_text)
    score = found / max(len(gt_keywords), 1)

    return {
        "key": "context_recall",
        "score": round(score, 3),
        "comment": f"{found}/{len(gt_keywords)} keywords found",
    }


def hit_evaluator(run, example) -> dict:
    """Binary: did ANY chunk contain the core answer?"""
    ground_truth = example.outputs["ground_truth"]
    contexts = run.outputs.get("contexts", [])

    gt_keywords = set(ground_truth.lower().split())
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "were",
        "are",
        "with",
        "and",
        "or",
        "of",
        "in",
        "to",
        "for",
        "on",
        "at",
        "by",
    }
    gt_keywords = gt_keywords - stop_words

    all_text = " ".join(contexts).lower()
    found = sum(1 for kw in gt_keywords if kw in all_text)
    recall = found / max(len(gt_keywords), 1)

    hit = recall > 0.5
    return {
        "key": "hit",
        "score": 1.0 if hit else 0.0,
        "comment": f"Recall={recall:.0%} → {'HIT' if hit else 'MISS'}",
    }


def main():
    global _CHUNK_SIZE, _CHUNK_OVERLAP

    parser = argparse.ArgumentParser(description="Run LangSmith RAG evaluation")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    _CHUNK_SIZE = args.chunk_size
    _CHUNK_OVERLAP = args.chunk_overlap

    default_name = f"rag-cs{_CHUNK_SIZE}-co{_CHUNK_OVERLAP}"

    print("\n" + "=" * 60)
    print("  🧠 LangSmith RAG Evaluation")
    print("=" * 60)

    # Initialize logging
    from ai_researcher.logging import setup_logging

    setup_logging()

    client = Client()
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if not datasets:
        print(f"  ❌ Dataset '{DATASET_NAME}' not found!")
        print("  Run first: python -m tests.eval.langsmith_upload_rag")
        return

    exp_name = args.experiment_name or default_name
    print(f"  📦 Dataset: {DATASET_NAME}")
    print(f"  🔬 Experiment: {exp_name}")
    print(f"  📐 Chunk Size: {_CHUNK_SIZE} | Overlap: {_CHUNK_OVERLAP}")
    print("  ⏳ Running evaluation...\n")

    evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[
            context_precision_evaluator,
            context_recall_evaluator,
            hit_evaluator,
        ],
        experiment_prefix=exp_name,
        max_concurrency=1,
    )

    print("\n" + "=" * 60)
    print("  ✅ RAG evaluation complete!")
    print("  🔗 View results at: https://smith.langchain.com/datasets")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
