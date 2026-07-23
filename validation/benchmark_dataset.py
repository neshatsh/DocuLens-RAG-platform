"""
Build a hand-labeled gold benchmark (~40 Q&A pairs) from CUAD-QA.

Pulls representative items from the already-ingested cuad_v1 collection
across 6 clause categories, saves as validation/data/benchmark.json.

Run:
    python validation/benchmark_dataset.py
"""
from __future__ import annotations
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from src.retrieval.vector_store import VectorStore
from src.utils.logger import logger

BENCHMARK_PATH = Path("validation/data/benchmark.json")
CUAD_COLLECTION = "cuad_v1"

# Categories chosen to cover the conceptual review risks:
# - Cap On Liability / Uncapped Liability: numeric precision risk
# - Anti-Assignment: negation handling risk
# - Governing Law / Audit Rights: straightforward retrieval baseline
# - License Grant: defined-term scope risk
TARGET_CATEGORIES = [
    "Governing Law",
    "Cap On Liability",
    "Anti-Assignment",
    "Audit Rights",
    "Expiration Date",
    "License Grant",
]
ITEMS_PER_CATEGORY = 7  # ~42 total


def _make_item_id(category: str, idx: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
    return f"{slug}_{idx:03d}"


def build_benchmark(collection: str = CUAD_COLLECTION) -> list[dict]:
    store = VectorStore(collection_name=collection)
    ingested_names = set(store.list_document_names())

    # Pre-cache name → document_id
    name_to_id: dict[str, str] = {}
    for name in ingested_names:
        doc_id = store.get_document_id_by_name(name)
        if doc_id:
            name_to_id[name] = doc_id

    logger.info("Loading CUAD-QA dataset ...")
    ds = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)

    per_cat: dict[str, list] = {cat: [] for cat in TARGET_CATEGORIES}

    for item in ds:
        title = item.get("title", "")
        if title not in ingested_names:
            continue

        q = item.get("question", "")
        m = re.search(r'related to "([^"]+)"', q)
        if not m:
            continue
        category = m.group(1)
        if category not in per_cat:
            continue
        if len(per_cat[category]) >= ITEMS_PER_CATEGORY:
            continue

        answers = item.get("answers", {}).get("text", [])
        if not answers or not answers[0].strip():
            continue

        context = item.get("context", "")
        doc_id = name_to_id.get(title, "")
        if not doc_id:
            continue

        idx = len(per_cat[category])
        per_cat[category].append({
            "id": _make_item_id(category, idx),
            "question": q,
            "gold_answer": answers[0],
            # Truncate very long contexts; ragas uses this as reference_context
            "reference_context": context[:3000],
            "contract_name": title,
            "document_id": doc_id,
            "category": category,
        })

    items = []
    for cat in TARGET_CATEGORIES:
        n = len(per_cat[cat])
        logger.info(f"  {cat}: {n} items")
        items.extend(per_cat[cat])

    logger.info(f"Total benchmark items: {len(items)}")
    return items


def main() -> None:
    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    items = build_benchmark()

    data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "collection": CUAD_COLLECTION,
            "categories": TARGET_CATEGORIES,
            "items_per_category": ITEMS_PER_CATEGORY,
            "total_items": len(items),
        },
        "items": items,
    }

    BENCHMARK_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Benchmark saved to {BENCHMARK_PATH}")
    print(f"Saved {len(items)} items to {BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
