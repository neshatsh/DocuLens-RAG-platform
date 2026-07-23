"""
Evaluate retrieval quality on CUAD-QA pairs and log results to MLflow.

Requires contracts to be ingested first:
    python scripts/download_cuad_v1.py
    python scripts/ingest_cuad.py --max 200 --clear

Run:
    python scripts/evaluate.py --max-samples 100 --top-k 5
    python scripts/evaluate.py --max-samples 100 --top-k 5 --per-document
"""
from __future__ import annotations
import argparse
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from datasets import load_dataset
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore
from src.utils.config import get_settings
from src.utils.logger import logger

settings = get_settings()

DEFAULT_COLLECTION = "cuad_v1"


def _word_normalize(text: str) -> str:
    """
    Reduce text to lowercase word tokens only (letters + digits, space-separated).

    Makes substring matching robust to pipeline-induced text differences:
    - The WordPiece tokenizer lowercases and spaces out abbreviations
      ('L.L.C.' -> 'l. l. c.' in stored chunks)
    - TextCleaner collapses repeated chars ('[*****]' -> '[***]')

    Both gold answer and chunk text are normalized the same way so those
    artefacts cancel out.
    """
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def load_cuad_qa_pairs(
    max_samples: int = 200,
    ingested_names: set[str] | None = None,
) -> list[dict]:
    """
    Load CUAD-QA SQuAD-format pairs.

    If ingested_names is provided, only pairs whose title matches a contract
    that was actually ingested are returned — prevents evaluating against
    contracts whose text was never indexed.
    """
    logger.info("Loading CUAD-QA pairs from theatticusproject/cuad-qa ...")
    dataset = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)

    skipped_not_ingested = 0
    pairs = []
    for item in dataset:
        question = item.get("question", "")
        answers = item.get("answers", {})
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        context = item.get("context", "")
        title = item.get("title", "unknown")

        if not question or not answer_texts or not context:
            continue

        if ingested_names is not None and title not in ingested_names:
            skipped_not_ingested += 1
            continue

        pairs.append({
            "question": question,
            "answer": answer_texts[0] if answer_texts else "",
            "document_name": title,
        })

        if len(pairs) >= max_samples:
            break

    if skipped_not_ingested:
        logger.info(
            f"Skipped {skipped_not_ingested} QA pairs for contracts not in the ingested corpus"
        )
    logger.info(f"Loaded {len(pairs)} QA pairs")
    return pairs


def evaluate_retrieval(
    pairs: list[dict],
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
    per_document: bool = False,
) -> dict:
    """
    Evaluate retrieval quality on QA pairs.

    A hit is recorded when a word-normalised prefix (first 30 tokens) of the
    gold answer appears in any of the top-k retrieved chunks.

    per_document=True  — restricts each query to the specific contract the QA
      pair belongs to. Measures clause-finding accuracy within a known document
      (aligned with how CUAD-QA is designed). Yields Hit Rate@5 ~0.71.

    per_document=False — cross-corpus mode: retrieves across all ingested
      contracts. Harder because every contract has similar clause types.
      Yields Hit Rate@5 ~0.13.
    """
    vector_store = VectorStore(collection_name=collection_name)
    retriever = Retriever(top_k=20, vector_store=vector_store)
    reranker = Reranker(top_k=top_k)

    doc_id_cache: dict[str, str | None] = {}

    hits = 0
    mrr_total = 0.0
    ndcg_total = 0.0
    total = 0

    for i, pair in enumerate(pairs):
        question = pair["question"]
        gold_answer = pair["answer"]
        doc_name = pair.get("document_name", "")

        gold_words = _word_normalize(gold_answer)
        gold_prefix_words = " ".join(gold_words.split()[:30])

        document_filter = None
        if per_document and doc_name:
            if doc_name not in doc_id_cache:
                doc_id_cache[doc_name] = vector_store.get_document_id_by_name(doc_name)
            doc_id = doc_id_cache[doc_name]
            if doc_id:
                document_filter = [doc_id]

        try:
            raw_chunks = retriever.retrieve(question, document_filter=document_filter)
            reranked = reranker.rerank(question, raw_chunks, top_k=top_k)
        except Exception as exc:
            logger.warning(f"Retrieval failed for pair {i}: {exc}")
            continue

        hit_rank = None
        for rank, chunk in enumerate(reranked, start=1):
            chunk_words = _word_normalize(chunk["text"])
            if gold_prefix_words and gold_prefix_words in chunk_words:
                hit_rank = rank
                break

        if hit_rank:
            hits += 1
            mrr_total += 1.0 / hit_rank
            ndcg_total += 1.0 / math.log2(hit_rank + 1)

        total += 1
        if (i + 1) % 20 == 0:
            logger.info(
                f"  Evaluated {i+1}/{len(pairs)} pairs | "
                f"hit_rate={hits/total:.3f} | mrr={mrr_total/total:.3f} | ndcg={ndcg_total/total:.3f}"
            )

    metrics = {
        f"hit_rate_at_{top_k}": hits / total if total > 0 else 0.0,
        "mrr": mrr_total / total if total > 0 else 0.0,
        f"ndcg_at_{top_k}": ndcg_total / total if total > 0 else 0.0,
        "total_evaluated": total,
        "top_k": top_k,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocuLens retrieval on CUAD-QA")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--collection", type=str, default=DEFAULT_COLLECTION,
        help="ChromaDB collection to evaluate against (must match ingest --collection)",
    )
    parser.add_argument(
        "--per-document", action="store_true",
        help=(
            "Restrict each query to the contract it belongs to. "
            "Measures clause-finding within a known document "
            "(per-document QA mode, aligned with CUAD-QA design). "
            "Without this flag, retrieval is cross-corpus."
        ),
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    run_name = "retrieval_eval_cuad_v1_per_doc" if args.per_document else "retrieval_eval_cuad_v1"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "retrieval_top_k": settings.retrieval_top_k,
            "reranker_top_k": args.top_k,
            "max_samples": args.max_samples,
            "collection": args.collection,
            "corpus": "CUAD_v1_full_contracts",
            "eval_mode": "per_document" if args.per_document else "cross_corpus",
        })

        store = VectorStore(collection_name=args.collection)
        ingested_names = set(store.list_document_names())
        logger.info(
            f"Found {len(ingested_names)} ingested contracts in '{args.collection}' "
            f"({store.count()} total chunks)"
        )

        if not ingested_names:
            logger.error(
                "No contracts found in collection. Ingest first:\n"
                "  python scripts/download_cuad_v1.py\n"
                "  python scripts/ingest_cuad.py --max 200 --clear"
            )
            sys.exit(1)

        mlflow.log_param("ingested_contracts", len(ingested_names))
        mlflow.log_param("total_chunks", store.count())

        pairs = load_cuad_qa_pairs(
            max_samples=args.max_samples,
            ingested_names=ingested_names,
        )

        if not pairs:
            logger.error(
                "No QA pairs matched ingested contracts. "
                "Check that ingestion used CUAD v1 full contracts, "
                "not the HuggingFace excerpt version."
            )
            sys.exit(1)

        mode_label = "per-document" if args.per_document else "cross-corpus"
        logger.info(f"Evaluation mode: {mode_label}")
        metrics = evaluate_retrieval(
            pairs,
            top_k=args.top_k,
            collection_name=args.collection,
            per_document=args.per_document,
        )

        mlflow.log_metrics(metrics)
        logger.info(f"Results: {metrics}")
        print("\n=== Evaluation Results ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        hr = metrics.get(f"hit_rate_at_{args.top_k}", 0.0)
        mrr = metrics.get("mrr", 0.0)
        if hr < 0.1 and mrr < 0.05:
            print(
                "\n[WARNING] Hit Rate and MRR are still very low. "
                "Possible causes:\n"
                "  - Corpus not fully ingested (try --max 500)\n"
                "  - Wrong collection (check --collection matches ingest)\n"
                "  - QA titles don't match ingested document names\n"
                f"  - Ingested names sample: {list(ingested_names)[:3]}"
            )


if __name__ == "__main__":
    main()
