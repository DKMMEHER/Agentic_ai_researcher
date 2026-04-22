"""RAG Evaluation — ChromaDB Retrieval Quality Scorer.

Measures how well the Vector Database retrieves relevant context
when queried about ingested PDFs. Uses both custom metrics and
RAGAS framework for comprehensive evaluation.

Usage:
    python -m tests.eval.eval_rag
    python -m tests.eval.eval_rag --max-papers 1
    python -m tests.eval.eval_rag --chunk-size 500 --chunk-overlap 100
"""

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import PyPDF2
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_rag_cases(path=None, max_papers=None):
    """Load RAG test cases."""
    if path is None:
        path = Path(__file__).parent / "rag_test_cases.json"
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    if max_papers:
        cases = cases[:max_papers]
    return cases


def ingest_pdf(url, vector_store, chunk_size=1000, chunk_overlap=200):
    """Download and ingest a PDF into ChromaDB with configurable chunking."""
    print(f"    📥 Downloading PDF: {url}")
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(full_text)

    docs = [
        Document(page_content=chunk, metadata={"source": url, "doc_id": doc_id})
        for chunk in chunks
    ]

    vector_store.add_documents(docs)
    print(
        f"    ✅ Ingested {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})"
    )
    return len(chunks)


def query_and_evaluate(vector_store, url, question, ground_truth, k=4):
    """Query ChromaDB and evaluate retrieval quality against ground truth."""
    results = vector_store.similarity_search_with_relevance_scores(
        question, k=k, filter={"source": url}
    )

    if not results:
        return {
            "question": question,
            "ground_truth": ground_truth,
            "contexts": [],
            "scores": [],
            "context_precision": 0.0,
            "context_recall": 0.0,
            "hit": False,
        }

    contexts = [doc.page_content for doc, _score in results]
    scores = [round(score, 4) for _doc, score in results]

    # --- Custom Metric 1: Context Precision ---
    # How many retrieved chunks actually contain ground truth keywords?
    gt_keywords = set(ground_truth.lower().split())
    # Remove common words
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

    relevant_count = 0
    for ctx in contexts:
        ctx_lower = ctx.lower()
        keyword_hits = sum(1 for kw in gt_keywords if kw in ctx_lower)
        # Consider relevant if >30% of ground truth keywords found
        if keyword_hits / max(len(gt_keywords), 1) > 0.3:
            relevant_count += 1

    context_precision = relevant_count / len(contexts) if contexts else 0.0

    # --- Custom Metric 2: Context Recall ---
    # What fraction of ground truth keywords appear across ALL retrieved chunks?
    all_context_text = " ".join(contexts).lower()
    found_keywords = sum(1 for kw in gt_keywords if kw in all_context_text)
    context_recall = found_keywords / max(len(gt_keywords), 1)

    # --- Custom Metric 3: Hit ---
    # Does ANY retrieved chunk contain the core answer?
    # Check if the most important fact from ground truth appears
    hit = context_recall > 0.5

    return {
        "question": question,
        "ground_truth": ground_truth,
        "contexts": contexts,
        "scores": scores,
        "context_precision": round(context_precision, 3),
        "context_recall": round(context_recall, 3),
        "hit": hit,
    }


def run_ragas_evaluation(eval_results, use_ragas=False):
    """Optionally run full RAGAS metrics (requires LLM tokens).

    If use_ragas=False, returns only custom metrics.
    If use_ragas=True, uses RAGAS framework for faithfulness & relevancy.
    """
    if not use_ragas:
        return None

    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        # Build RAGAS dataset
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for r in eval_results:
            data["question"].append(r["question"])
            data["answer"].append(
                r["contexts"][0] if r["contexts"] else "No context found"
            )
            data["contexts"].append(r["contexts"])
            data["ground_truth"].append(r["ground_truth"])

        dataset = Dataset.from_dict(data)

        result = ragas_evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        )

        return result.to_pandas().to_dict()

    except Exception as e:
        print(f"    ⚠️  RAGAS evaluation failed: {e}")
        print("    ℹ️  RAGAS requires an OpenAI API key for LLM-based metrics.")  # noqa: RUF001
        print("    ℹ️  Custom metrics are still available below.")  # noqa: RUF001
        return None


def run_evaluation(cases, chunk_size=1000, chunk_overlap=200, use_ragas=False):
    """Run the full RAG evaluation."""
    from ai_researcher.logging import setup_logging
    from ai_researcher.tools.db import get_vector_store

    setup_logging()
    vector_store = get_vector_store()

    all_results = []

    print("\n" + "=" * 70)
    print("  🧠 RAG EVALUATION — CHROMADB RETRIEVAL QUALITY")
    print("=" * 70)
    print(f"  Chunk Size: {chunk_size} | Overlap: {chunk_overlap}")
    print(f"  Papers to test: {len(cases)}\n")

    for paper in cases:
        title = paper["paper_title"]
        url = paper["pdf_url"]
        questions = paper["questions"]

        print(f"  📄 {title}")

        # Ingest the PDF
        try:
            ingest_pdf(url, vector_store, chunk_size, chunk_overlap)
        except Exception as e:
            print(f"    ❌ Failed to ingest: {e}")
            continue

        time.sleep(2)

        # Evaluate each question
        paper_results = []
        for q in questions:
            result = query_and_evaluate(
                vector_store, url, q["question"], q["ground_truth"]
            )
            paper_results.append(result)

            hit_icon = "✅" if result["hit"] else "❌"
            print(
                f"    {hit_icon} P:{result['context_precision']:.0%}  R:{result['context_recall']:.0%}  | {q['question'][:50]}..."
            )

        all_results.extend(paper_results)
        print()

    # Optionally run RAGAS
    ragas_results = run_ragas_evaluation(all_results, use_ragas=use_ragas)

    return all_results, ragas_results


def print_summary(results, ragas_results=None):
    """Print formatted summary."""
    total = len(results)
    avg_precision = sum(r["context_precision"] for r in results) / total if total else 0
    avg_recall = sum(r["context_recall"] for r in results) / total if total else 0
    hit_rate = sum(1 for r in results if r["hit"]) / total if total else 0

    print("=" * 70)
    print("  📊 RAG EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n  Total Questions: {total}")
    print(f"  Avg Context Precision: {avg_precision:.1%}")
    print(f"  Avg Context Recall:    {avg_recall:.1%}")
    print(f"  Hit Rate:              {hit_rate:.1%}")

    print("\n  Per-Question Breakdown:")
    print("  " + "-" * 60)
    for r in results:
        hit_icon = "✅" if r["hit"] else "❌"
        print(
            f"    {hit_icon} P:{r['context_precision']:.0%}  R:{r['context_recall']:.0%}  | {r['question'][:50]}..."
        )

    # Show worst performers
    worst = sorted(results, key=lambda x: x["context_recall"])[:3]
    if worst and worst[0]["context_recall"] < 0.5:
        print("\n  ⚠️ Lowest Recall Questions:")
        print("  " + "-" * 60)
        for r in worst:
            if r["context_recall"] < 0.5:
                print(f"    R:{r['context_recall']:.0%}  | {r['question']}")
                print(f"           GT: {r['ground_truth'][:60]}...")

    if ragas_results:
        print("\n  📈 RAGAS Metrics (LLM-based):")
        print(f"    {json.dumps(ragas_results, indent=2)[:500]}")

    print("\n" + "=" * 70)


def save_report(results, ragas_results, chunk_size, chunk_overlap):
    """Save report."""
    reports_dir = PROJECT_ROOT / "tests" / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"rag_{timestamp}.json"

    total = len(results)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_questions": total,
        "avg_context_precision": round(
            sum(r["context_precision"] for r in results) / total, 3
        )
        if total
        else 0,
        "avg_context_recall": round(
            sum(r["context_recall"] for r in results) / total, 3
        )
        if total
        else 0,
        "hit_rate": round(sum(1 for r in results if r["hit"]) / total, 3)
        if total
        else 0,
        "ragas_results": ragas_results,
        "results": [{k: v for k, v in r.items() if k != "contexts"} for r in results],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  📁 Report saved to: {report_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument(
        "--max-papers", type=int, default=None, help="Limit papers to test"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Chunk size for splitting"
    )
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument(
        "--use-ragas",
        action="store_true",
        help="Enable RAGAS LLM-based metrics (needs OpenAI key)",
    )
    args = parser.parse_args()

    cases = load_rag_cases(max_papers=args.max_papers)
    results, ragas_results = run_evaluation(
        cases,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_ragas=args.use_ragas,
    )
    print_summary(results, ragas_results)
    save_report(results, ragas_results, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
