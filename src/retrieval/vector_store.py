"""ChromaDB vector store wrapper for chunk storage and retrieval."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.utils.config import get_settings
from src.utils.logger import logger
from src.ingestion.chunker import TextChunk

settings = get_settings()


class VectorStore:
    """
    ChromaDB-backed vector store.

    Collections:
    - One collection per project (default: "doculens")
    - Documents stored with full metadata for filtering
    - Supports both embedding-based and metadata-filtered search
    """

    def __init__(
        self,
        collection_name: str = "doculens",
        persist_dir: str | None = None,
    ):
        persist_dir = persist_dir or str(settings.chroma_persist_dir)
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_name = collection_name
        logger.info(
            f"VectorStore ready: '{collection_name}' "
            f"({self.collection.count()} existing chunks)"
        )

    # ------------------------------------------------------------------ #
    # Write operations                                                     #
    # ------------------------------------------------------------------ #

    def add_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: np.ndarray,
    ) -> int:
        """Add chunks + their embeddings to the collection. Returns count added."""
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "document_name": c.document_name,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number or -1,
                "token_count": c.token_count,
                **{k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ]
        embedding_list = embeddings.tolist()

        # ChromaDB upserts by ID — safe to re-ingest
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embedding_list,
        )
        logger.info(f"Upserted {len(chunks)} chunks into '{self.collection_name}'")
        return len(chunks)

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to a document."""
        self.collection.delete(where={"document_id": document_id})
        logger.info(f"Deleted document: {document_id}")

    # ------------------------------------------------------------------ #
    # Read operations                                                      #
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most similar chunks.
        Returns list of dicts with keys: id, text, metadata, distance.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            hits.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine similarity
            })
        return hits

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Fetch all chunks for a given document ID."""
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        return [
            {"id": results["ids"][i], "text": results["documents"][i], "metadata": results["metadatas"][i]}
            for i in range(len(results["ids"]))
        ]

    def count(self) -> int:
        return self.collection.count()

    def list_documents(self) -> List[str]:
        """Return unique document_ids in the store."""
        results = self.collection.get(include=["metadatas"])
        seen = set()
        docs = []
        for meta in results["metadatas"]:
            doc_id = meta.get("document_id", "")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                docs.append(doc_id)
        return docs

    def list_document_names(self) -> List[str]:
        """Return unique document_names stored in this collection."""
        results = self.collection.get(include=["metadatas"])
        seen: set = set()
        names = []
        for meta in results["metadatas"]:
            name = meta.get("document_name", "")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names

    def get_document_id_by_name(self, document_name: str) -> Optional[str]:
        """Return the document_id for a given document_name, or None if not found."""
        results = self.collection.get(
            where={"document_name": document_name},
            include=["metadatas"],
            limit=1,
        )
        if results["metadatas"]:
            return results["metadatas"][0].get("document_id")
        return None

    def clear(self) -> None:
        """Delete and recreate the collection, wiping all data."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Cleared collection '{self.collection_name}'")
