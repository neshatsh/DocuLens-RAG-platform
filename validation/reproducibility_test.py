"""
Reproducibility test: runs each of 15-20 benchmark queries 5x through the real
pipeline and measures stability of retrieved chunk sets and generated answers.

Tests whether the system produces consistent results across repeated identical
queries (relevant for deployed systems where users may retry queries).

Saves results to validation/results/reproducibility.json.

Run:
    python validation/benchmark_dataset.py   # build benchmark first
    python validation/reproducibility_test.py
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_client import LLMClient
from src.utils.config import get_settings
from src.utils.logger import logger

settings = get_settings()

BENCHMARK_PATH = Path("validation/data/benchmark.json")
REPRO_PATH = Path("validation/results/reproducibility.json")
N_QUERIES = 15
N_RUNS = 5
CUAD_COLLECTION = "cuad_v1"
HIGH_VARIANCE_THRESHOLD_JACCARD = 0.8   # flag if mean chunk Jaccard < this
HIGH_VARIANCE_THRESHOLD_OVERLAP = 0.6   # flag if mean answer word-overlap < this


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union)


def _word_overlap(text_a: str, text_b: str) -> float:
    """Overlap coefficient: |A∩B| / min(|A|, |B|)"""
    wa = set(re.findall(r"[a-z0-9]+", text_a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", text_b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def run_once(query: str, document_filter: list[str] | None,
             retriever: Retriever, reranker: Reranker,
             prompt_builder: PromptBuilder, llm: LLMClient) -> dict:
    raw = retriever.retrieve(query, document_filter=document_filter)
    reranked = reranker.rerank(query, raw, top_k=5)
    system_prompt, user_msg = prompt_builder.build_rag_prompt(query, reranked)
    answer = llm.complete(system_prompt, user_msg)
    chunk_ids = [c["id"] for c in reranked]
    return {"chunk_ids": chunk_ids, "answer": answer}


def test_query(item: dict, retriever: Retriever, reranker: Reranker,
               prompt_builder: PromptBuilder, llm: LLMClient) -> dict:
    doc_filter = [item["document_id"]] if item.get("document_id") else None
    runs = []

    for r in range(N_RUNS):
        logger.info(f"    Run {r+1}/{N_RUNS}")
        try:
            result = run_once(item["question"], doc_filter, retriever, reranker, prompt_builder, llm)
            runs.append(result)
        except Exception as exc:
            logger.warning(f"    Run {r+1} failed: {exc}")
            runs.append({"chunk_ids": [], "answer": "", "error": str(exc)})
        time.sleep(0.5)  # polite rate-limiting

    valid_runs = [r for r in runs if not r.get("error") and r["chunk_ids"]]
    if len(valid_runs) < 2:
        return {
            "id": item["id"],
            "question": item["question"][:120],
            "category": item["category"],
            "n_valid_runs": len(valid_runs),
            "avg_chunk_jaccard": None,
            "min_chunk_jaccard": None,
            "avg_answer_overlap": None,
            "min_answer_overlap": None,
            "high_variance": True,
            "error": "insufficient valid runs",
        }

    # Chunk set stability: pairwise Jaccard of chunk ID sets
    chunk_sets = [set(r["chunk_ids"]) for r in valid_runs]
    jaccards = []
    for i in range(len(chunk_sets)):
        for j in range(i + 1, len(chunk_sets)):
            jaccards.append(_jaccard(chunk_sets[i], chunk_sets[j]))
    avg_jaccard = sum(jaccards) / len(jaccards)
    min_jaccard = min(jaccards)

    # Answer text stability: pairwise word-overlap
    answers = [r["answer"] for r in valid_runs if r.get("answer")]
    overlaps = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            overlaps.append(_word_overlap(answers[i], answers[j]))
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    min_overlap = min(overlaps) if overlaps else 0.0

    high_variance = avg_jaccard < HIGH_VARIANCE_THRESHOLD_JACCARD or avg_overlap < HIGH_VARIANCE_THRESHOLD_OVERLAP

    return {
        "id": item["id"],
        "question": item["question"][:120],
        "category": item["category"],
        "n_valid_runs": len(valid_runs),
        "avg_chunk_jaccard": round(avg_jaccard, 4),
        "min_chunk_jaccard": round(min_jaccard, 4),
        "avg_answer_overlap": round(avg_overlap, 4),
        "min_answer_overlap": round(min_overlap, 4),
        "high_variance": high_variance,
    }


def main() -> None:
    if not BENCHMARK_PATH.exists():
        print(f"Benchmark not found. Run benchmark_dataset.py first.")
        sys.exit(1)

    data = json.loads(BENCHMARK_PATH.read_text())
    items = data["items"][:N_QUERIES]
    logger.info(f"Running {N_RUNS}x reproducibility test on {len(items)} queries")

    vector_store = VectorStore(collection_name=CUAD_COLLECTION)
    retriever = Retriever(top_k=20, vector_store=vector_store)
    reranker = Reranker(top_k=5)
    prompt_builder = PromptBuilder()
    llm = LLMClient()

    results = []
    for i, item in enumerate(items):
        logger.info(f"[{i+1}/{len(items)}] Query: {item['id']}")
        result = test_query(item, retriever, reranker, prompt_builder, llm)
        results.append(result)

    # Aggregate
    valid = [r for r in results if r.get("avg_chunk_jaccard") is not None]
    high_var = [r for r in results if r.get("high_variance")]

    avg_chunk_j = sum(r["avg_chunk_jaccard"] for r in valid) / len(valid) if valid else None
    avg_ans_ov = sum(r["avg_answer_overlap"] for r in valid) / len(valid) if valid else None

    if high_var:
        print(f"\n[FLAGGED] {len(high_var)} high-variance queries:")
        for r in high_var:
            print(f"  [{r['id']}] chunk_jaccard={r.get('avg_chunk_jaccard')} "
                  f"answer_overlap={r.get('avg_answer_overlap')}")
            print(f"    Q: {r['question'][:100]}")
    else:
        print(f"\n[OK] All queries within variance thresholds.")

    output = {
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "n_queries": len(items),
            "n_runs_per_query": N_RUNS,
            "chunk_jaccard_threshold": HIGH_VARIANCE_THRESHOLD_JACCARD,
            "answer_overlap_threshold": HIGH_VARIANCE_THRESHOLD_OVERLAP,
        },
        "aggregate": {
            "avg_chunk_jaccard": round(avg_chunk_j, 4) if avg_chunk_j is not None else None,
            "avg_answer_overlap": round(avg_ans_ov, 4) if avg_ans_ov is not None else None,
            "high_variance_queries": len(high_var),
            "total_queries": len(results),
        },
        "items": results,
    }

    REPRO_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPRO_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"Results saved to {REPRO_PATH}")

    print("\n=== Reproducibility Results ===")
    print(f"  Queries tested:          {len(results)} ({N_RUNS} runs each)")
    print(f"  Avg chunk Jaccard:       {avg_chunk_j:.4f}" if avg_chunk_j else "  Avg chunk Jaccard: N/A")
    print(f"  Avg answer word-overlap: {avg_ans_ov:.4f}" if avg_ans_ov else "  Avg answer word-overlap: N/A")
    print(f"  High-variance queries:   {len(high_var)}/{len(results)}")


if __name__ == "__main__":
    main()
