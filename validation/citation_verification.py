"""
Citation verification: checks that every [Source: ..., Page ...] citation in a
generated answer corresponds to a chunk that was actually in the retrieved context.

Tests conceptual review risk #3 (prompt-only citation enforcement).

Reads generated answers from validation/results/answers.json (produced by
faithfulness_eval.py) and produces validation/results/citations.json.

Run:
    python validation/faithfulness_eval.py   # generates answers first
    python validation/citation_verification.py
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import logger

ANSWERS_PATH = Path("validation/results/answers.json")
CITATIONS_PATH = Path("validation/results/citations.json")

# Matches: [Source: SOME DOC NAME, Page 3]  or  [Source: SOME DOC, Page -1]
CITATION_RE = re.compile(
    r"\[Source:\s*([^\],]+?),\s*Page\s*([^\]]+?)\]",
    re.IGNORECASE,
)


def parse_citations(answer_text: str) -> list[dict]:
    """Extract all [Source: ..., Page ...] citations from answer text."""
    citations = []
    for m in CITATION_RE.finditer(answer_text):
        citations.append({
            "document_name": m.group(1).strip(),
            "page_number": m.group(2).strip(),
            "span": m.group(0),
        })
    return citations


def _sentence_containing(answer_text: str, citation_span: str) -> str:
    """Return the sentence (or short window) that contains the citation span."""
    pos = answer_text.find(citation_span)
    if pos == -1:
        return ""
    # Find sentence boundaries ≤200 chars either side
    start = max(0, answer_text.rfind(".", 0, pos) + 1)
    end = answer_text.find(".", pos)
    end = end + 1 if end != -1 else min(len(answer_text), pos + 200)
    return answer_text[start:end].strip()


def _lexical_overlap(sentence: str, chunk_text: str) -> float:
    """Simple word-overlap ratio between sentence and chunk text (Jaccard on words)."""
    words_s = set(re.findall(r"[a-z0-9]+", sentence.lower()))
    words_c = set(re.findall(r"[a-z0-9]+", chunk_text.lower()))
    if not words_s:
        return 0.0
    intersection = words_s & words_c
    return len(intersection) / len(words_s)


def verify_item(item: dict) -> dict:
    """
    For a single answered item, check each citation in the answer:
    - Is the cited doc_name present in retrieved sources?
    - Does the sentence containing the citation have lexical overlap with that chunk?
    """
    answer = item.get("answer", "")
    sources = item.get("sources", [])
    retrieved_names = {s.get("document_name", "").strip() for s in sources}
    contexts = item.get("retrieved_contexts", [])

    citations = parse_citations(answer)
    verified = []

    for cit in citations:
        cited_name = cit["document_name"]
        in_retrieved = cited_name in retrieved_names

        # Find the matching chunk text for lexical overlap check
        matching_chunk = ""
        for s, ctx in zip(sources, contexts):
            if s.get("document_name", "").strip() == cited_name:
                matching_chunk = ctx
                break

        sentence = _sentence_containing(answer, cit["span"])
        overlap = _lexical_overlap(sentence, matching_chunk) if matching_chunk else 0.0

        verified.append({
            **cit,
            "in_retrieved_context": in_retrieved,
            "fabricated": not in_retrieved,
            "sentence": sentence[:200],
            "lexical_overlap": round(overlap, 3),
            "unsupported": overlap < 0.3 if in_retrieved else False,
        })

    return {
        "id": item.get("id", ""),
        "category": item.get("category", ""),
        "contract_name": item.get("contract_name", ""),
        "question": item.get("question", "")[:120],
        "n_citations": len(citations),
        "n_fabricated": sum(1 for v in verified if v["fabricated"]),
        "n_unsupported": sum(1 for v in verified if v.get("unsupported")),
        "citations": verified,
    }


def main() -> None:
    if not ANSWERS_PATH.exists():
        print(f"Answers not found at {ANSWERS_PATH}. Run faithfulness_eval.py first.")
        sys.exit(1)

    answered = json.loads(ANSWERS_PATH.read_text())
    # Only process items that have answers
    valid = [a for a in answered if a.get("answer") and not a.get("error")]
    logger.info(f"Verifying citations for {len(valid)} answered items")

    item_results = [verify_item(a) for a in valid]

    # Aggregate
    total_cit = sum(r["n_citations"] for r in item_results)
    total_fab = sum(r["n_fabricated"] for r in item_results)
    total_unsup = sum(r["n_unsupported"] for r in item_results)
    items_with_cit = sum(1 for r in item_results if r["n_citations"] > 0)

    fab_rate = total_fab / total_cit if total_cit else 0.0
    unsup_rate = total_unsup / total_cit if total_cit else 0.0

    # Flag fabricated citations
    fabricated_cases = [r for r in item_results if r["n_fabricated"] > 0]
    if fabricated_cases:
        print(f"\n[FLAGGED] Fabricated citations in {len(fabricated_cases)} answers:")
        for r in fabricated_cases:
            for cit in r["citations"]:
                if cit["fabricated"]:
                    print(f"  [{r['id']}] cited '{cit['document_name']}' not in retrieved context")
    else:
        print("\n[OK] No fabricated citations detected.")

    output = {
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "n_items": len(valid),
            "n_items_with_citations": items_with_cit,
        },
        "aggregate": {
            "total_citations": total_cit,
            "fabricated_citations": total_fab,
            "unsupported_citations": total_unsup,
            "fabricated_citation_rate": round(fab_rate, 4),
            "unsupported_citation_rate": round(unsup_rate, 4),
        },
        "items": item_results,
    }

    CITATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CITATIONS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"Results saved to {CITATIONS_PATH}")

    print("\n=== Citation Verification Results ===")
    print(f"  Total citations parsed:      {total_cit}")
    print(f"  Fabricated (not in context): {total_fab} ({fab_rate:.1%})")
    print(f"  Unsupported (low overlap):   {total_unsup} ({unsup_rate:.1%})")


if __name__ == "__main__":
    main()
