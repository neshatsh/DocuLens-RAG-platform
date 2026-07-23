"""
Ingest full CUAD v1 contract text files into ChromaDB.

Data source: data/cuad_v1/full_contract_txt/ (official Atticus Project release).
Download first:  python scripts/download_cuad_v1.py

Run: python scripts/ingest_cuad.py [--max 200] [--clear]
"""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import logger

DEFAULT_TXT_DIR = Path("data/cuad_v1/full_contract_txt")
DEFAULT_COLLECTION = "cuad_v1"


def _find_txt_dir(base: Path) -> Path | None:
    """Locate full_contract_txt/ regardless of zip extraction depth."""
    for candidate in sorted(base.rglob("full_contract_txt")):
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    return None


def collect_contract_files(txt_dir: Path, max_contracts: int) -> list[Path]:
    """Return up to max_contracts .txt paths, sorted for reproducibility.

    If txt_dir doesn't exist or is empty, auto-discovers full_contract_txt/
    under data/cuad_v1/ (handles nested zip extraction layout).
    """
    resolved = txt_dir
    if not txt_dir.exists() or not any(txt_dir.glob("*.txt")):
        auto = _find_txt_dir(Path("data/cuad_v1"))
        if auto is not None:
            logger.info(f"Auto-discovered txt dir: {auto}")
            resolved = auto
        else:
            raise FileNotFoundError(
                f"No .txt files found in {txt_dir} or data/cuad_v1/.\n"
                "Run:  python scripts/download_cuad_v1.py"
            )
    return sorted(resolved.glob("*.txt"))[:max_contracts]


def ingest_text_files(file_paths: list[Path], collection_name: str = DEFAULT_COLLECTION) -> int:
    """
    Ingest full contract .txt files through the existing pipeline:
    clean -> chunk -> embed -> store.

    Uses path.stem as document_name so it matches CUAD-QA title format
    exactly (e.g. "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT").
    Returns total chunks stored.
    """
    from src.ingestion.text_cleaner import TextCleaner
    from src.ingestion.chunker import SemanticChunker
    from src.ingestion.embedder import Embedder
    from src.retrieval.vector_store import VectorStore

    cleaner = TextCleaner()
    chunker = SemanticChunker()
    embedder = Embedder()
    store = VectorStore(collection_name=collection_name)

    total_chunks = 0
    for i, path in enumerate(file_paths):
        contract_name = path.stem  # no .txt — matches CUAD-QA title exactly
        doc_id = hashlib.md5(contract_name.encode()).hexdigest()[:16]
        logger.info(f"[{i+1}/{len(file_paths)}] Ingesting: {contract_name}")

        text = path.read_text(encoding="utf-8", errors="replace")
        cleaned = cleaner.clean(text)
        if not cleaned.strip():
            logger.warning(f"  Skipping empty: {path.name}")
            continue

        chunks = chunker.chunk_document(
            text=cleaned,
            document_id=doc_id,
            document_name=contract_name,
            metadata={"source": "CUAD_v1", "file": path.name},
        )
        if not chunks:
            logger.warning(f"  No chunks produced for: {path.name}")
            continue

        _, embeddings = embedder.embed_chunks(chunks)
        store.add_chunks(chunks, embeddings)
        total_chunks += len(chunks)
        logger.info(f"  -> {len(chunks)} chunks stored")

    logger.info(f"Done. Total chunks added: {total_chunks} | Store total: {store.count()}")
    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CUAD v1 full contracts into ChromaDB")
    parser.add_argument("--max", type=int, default=50, help="Max contracts to ingest")
    parser.add_argument(
        "--txt-dir", type=str, default=str(DEFAULT_TXT_DIR),
        help="Directory containing CUAD v1 .txt files",
    )
    parser.add_argument(
        "--collection", type=str, default=DEFAULT_COLLECTION,
        help="ChromaDB collection name (default: cuad_v1, separate from API collection)",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Wipe the collection before ingesting (use for a fresh benchmark run)",
    )
    args = parser.parse_args()

    if args.clear:
        from src.retrieval.vector_store import VectorStore
        store = VectorStore(collection_name=args.collection)
        store.clear()

    txt_dir = Path(args.txt_dir)
    files = collect_contract_files(txt_dir, max_contracts=args.max)
    logger.info(f"Ingesting {len(files)} contracts from {txt_dir}")
    ingest_text_files(files, collection_name=args.collection)


if __name__ == "__main__":
    main()
