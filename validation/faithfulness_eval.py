"""
RAGAS faithfulness evaluation against the DocuLens benchmark dataset.

Calls the real pipeline (Retriever → Reranker → LLM) for each benchmark item,
then scores the answers with ragas metrics: Faithfulness, ContextPrecision,
ContextRecall, ResponseRelevancy.

Results are logged to MLflow (same experiment as scripts/evaluate.py) and saved
to validation/results/faithfulness.json.

Run:
    python validation/benchmark_dataset.py   # build benchmark first
    python validation/faithfulness_eval.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from ragas import evaluate as ragas_evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_client import LLMClient
from src.utils.config import get_settings
from src.utils.logger import logger

settings = get_settings()

BENCHMARK_PATH = Path("validation/data/benchmark.json")
ANSWERS_PATH = Path("validation/results/answers.json")
FAITHFULNESS_PATH = Path("validation/results/faithfulness.json")
FAITHFULNESS_THRESHOLD = 0.7
CUAD_COLLECTION = "cuad_v1"


def generate_answers(items: list[dict]) -> list[dict]:
    """
    Run the DocuLens pipeline (retrieve → rerank → generate) for each benchmark
    item, restricted to the specific contract (per-document mode).

    Saves full answers including chunk texts for ragas.
    """
    vector_store = VectorStore(collection_name=CUAD_COLLECTION)
    retriever = Retriever(top_k=20, vector_store=vector_store)
    reranker = Reranker(top_k=5)
    prompt_builder = PromptBuilder()
    llm = LLMClient()

    results = []
    for i, item in enumerate(items):
        logger.info(f"[{i+1}/{len(items)}] Generating answer for: {item['id']}")
        doc_filter = [item["document_id"]] if item.get("document_id") else None

        try:
            t0 = time.perf_counter()
            raw = retriever.retrieve(item["question"], document_filter=doc_filter)
            reranked = reranker.rerank(item["question"], raw, top_k=5)
            system_prompt, user_msg = prompt_builder.build_rag_prompt(item["question"], reranked)
            answer_text = llm.complete(system_prompt, user_msg)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            results.append({
                **item,
                "answer": answer_text,
                "retrieved_contexts": [c["text"] for c in reranked],
                "sources": [
                    {
                        "document_name": c.get("metadata", {}).get("document_name", ""),
                        "page_number": c.get("metadata", {}).get("page_number", "?"),
                        "rerank_score": round(c.get("rerank_score", 0), 4),
                    }
                    for c in reranked
                ],
                "elapsed_ms": round(elapsed_ms, 1),
                "error": None,
            })
        except Exception as exc:
            logger.warning(f"  Failed: {exc}")
            results.append({**item, "answer": "", "retrieved_contexts": [], "sources": [], "error": str(exc)})

    return results


def run_ragas(answered: list[dict]) -> tuple[dict, list[dict]]:
    """Score answers with ragas metrics. Returns (aggregate_metrics, per_item_scores)."""
    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key, temperature=0)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
    )

    metrics = [Faithfulness(), ContextPrecision(), ContextRecall(), ResponseRelevancy()]

    # Build ragas dataset — skip items with errors or no answer
    valid = [a for a in answered if a.get("answer") and not a.get("error")]
    samples = [
        SingleTurnSample(
            user_input=a["question"],
            response=a["answer"],
            retrieved_contexts=a["retrieved_contexts"] or ["[no context retrieved]"],
            reference=a["gold_answer"],
            reference_contexts=[a["reference_context"]],
        )
        for a in valid
    ]

    logger.info(f"Running ragas on {len(samples)} samples ...")
    dataset = EvaluationDataset(samples=samples)
    result = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        show_progress=True,
        raise_exceptions=False,
        allow_nest_asyncio=True,
    )

    scores_df = result.to_pandas()
    metric_names = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]

    aggregate = {}
    for col in metric_names:
        if col in scores_df.columns:
            aggregate[col] = round(float(scores_df[col].mean(skipna=True)), 4)

    per_item = []
    for idx, a in enumerate(valid):
        row = scores_df.iloc[idx] if idx < len(scores_df) else {}
        item_scores = {col: round(float(row[col]), 4) if col in row and row[col] == row[col] else None
                       for col in metric_names}
        per_item.append({
            "id": a["id"],
            "category": a["category"],
            "contract_name": a["contract_name"],
            "question": a["question"][:120],
            "gold_answer": a["gold_answer"][:120],
            "answer": a["answer"][:300],
            **item_scores,
        })

    return aggregate, per_item


def main() -> None:
    if not BENCHMARK_PATH.exists():
        print(f"Benchmark not found at {BENCHMARK_PATH}. Run benchmark_dataset.py first.")
        sys.exit(1)

    data = json.loads(BENCHMARK_PATH.read_text())
    items = data["items"]
    logger.info(f"Loaded {len(items)} benchmark items")

    # Step 1: generate answers (or load cached)
    ANSWERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ANSWERS_PATH.exists():
        logger.info(f"Loading cached answers from {ANSWERS_PATH}")
        answered = json.loads(ANSWERS_PATH.read_text())
    else:
        answered = generate_answers(items)
        ANSWERS_PATH.write_text(json.dumps(answered, indent=2), encoding="utf-8")
        logger.info(f"Answers saved to {ANSWERS_PATH}")

    # Step 2: ragas scoring
    aggregate, per_item = run_ragas(answered)
    logger.info(f"Aggregate metrics: {aggregate}")

    # Step 3: flag low faithfulness
    low_faith = [p for p in per_item if p.get("faithfulness") is not None
                 and p["faithfulness"] < FAITHFULNESS_THRESHOLD]
    if low_faith:
        print(f"\n[FLAGGED] {len(low_faith)} answers below faithfulness threshold ({FAITHFULNESS_THRESHOLD}):")
        for p in low_faith:
            print(f"  [{p['id']}] faithfulness={p['faithfulness']}")
            print(f"    Q: {p['question'][:100]}")
            print(f"    A: {p['answer'][:150]}")

    # Step 4: save results
    output = {
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "faithfulness_threshold": FAITHFULNESS_THRESHOLD,
            "collection": CUAD_COLLECTION,
            "n_evaluated": len(per_item),
            "n_errors": sum(1 for a in answered if a.get("error")),
        },
        "aggregate": aggregate,
        "low_faithfulness_count": len(low_faith),
        "items": per_item,
    }
    FAITHFULNESS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"Results saved to {FAITHFULNESS_PATH}")

    # Step 5: MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name="validation_faithfulness"):
        mlflow.log_params({"collection": CUAD_COLLECTION, "n_items": len(items)})
        mlflow.log_metrics({k: v for k, v in aggregate.items() if v is not None})
        mlflow.log_metric("low_faithfulness_count", len(low_faith))

    print("\n=== Faithfulness Evaluation Results ===")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")
    print(f"  items_below_threshold ({FAITHFULNESS_THRESHOLD}): {len(low_faith)}/{len(per_item)}")


if __name__ == "__main__":
    main()
