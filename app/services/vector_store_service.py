"""
VectorStoreService
--------------------
Wraps a persistent FAISS index: add, query, list, and delete document
chunks by source filename. Drop-in replacement for the previous ChromaDB
version - same public API (add_chunks, query, list_documents,
delete_document, total_chunks).

Persists two files inside settings.vector_store_dir:
  - index.faiss    (the FAISS index itself)
  - metadata.pkl   (id -> {source, chunk_index, content} + next_id counter)
"""
import os
import pickle
import uuid
from functools import lru_cache
from typing import Dict, List, Optional

import faiss
import numpy as np

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class VectorStoreService:
    def __init__(self):
        os.makedirs(settings.vector_store_dir, exist_ok=True)
        self._index_path = os.path.join(settings.vector_store_dir, "index.faiss")
        self._meta_path = os.path.join(settings.vector_store_dir, "metadata.pkl")

        self.index: Optional[faiss.Index] = None  # lazily created on first add
        self.dim: Optional[int] = None
        self.metadata: Dict[int, dict] = {}  # int id -> {source, chunk_index, content}
        self._next_id: int = 0

        self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load(self):
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self.index = faiss.read_index(self._index_path)
            self.dim = self.index.d
            with open(self._meta_path, "rb") as f:
                saved = pickle.load(f)
                self.metadata = saved["metadata"]
                self._next_id = saved["next_id"]
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

    def _persist(self):
        faiss.write_index(self.index, self._index_path)
        with open(self._meta_path, "wb") as f:
            pickle.dump({"metadata": self.metadata, "next_id": self._next_id}, f)

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.dim = dim
            # IndexIDMap2 wraps a flat inner-product index so we can add/remove by explicit int64 ids.
            # Inner product on normalized embeddings == cosine similarity.
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        elif self.dim != dim:
            raise ValueError(
                f"Embedding dimension mismatch: index is {self.dim}-d, got {dim}-d vector."
            )

    # ------------------------------------------------------------------
    # public API (unchanged signatures)
    # ------------------------------------------------------------------
    def add_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        source: str,
    ) -> int:
        """Add a document's chunks + embeddings to the index."""
        if not chunks:
            return 0

        vectors = np.array(embeddings, dtype="float32")
        self._ensure_index(vectors.shape[1])

        ids = np.array(
            [self._next_id + i for i in range(len(chunks))], dtype="int64"
        )
        self.index.add_with_ids(vectors, ids)

        for i, chunk_id in enumerate(ids):
            self.metadata[int(chunk_id)] = {
                "source": source,
                "chunk_index": i,
                "content": chunks[i],
            }
        self._next_id += len(chunks)

        self._persist()
        logger.info(f"Added {len(chunks)} chunks from '{source}' to vector store")
        return len(chunks)

    def query(
        self, query_embedding: List[float], top_k: Optional[int] = None
    ) -> List[Dict]:
        """Return the top_k most similar chunks to the query embedding."""
        if self.index is None or self.index.ntotal == 0:
            return []

        k = top_k or settings.top_k_results
        k = min(k, self.index.ntotal)

        vec = np.array([query_embedding], dtype="float32")
        scores, ids = self.index.search(vec, k)

        hits = []
        for score, chunk_id in zip(scores[0], ids[0]):
            if chunk_id == -1:
                continue
            meta = self.metadata.get(int(chunk_id))
            if meta is None:
                continue
            hits.append(
                {
                    "content": meta["content"],
                    "source": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", -1),
                    "score": float(score),  # already cosine similarity (normalized IP)
                }
            )
        return hits

    def list_documents(self) -> Dict[str, int]:
        """Return {filename: chunk_count} for everything stored."""
        counts: Dict[str, int] = {}
        for meta in self.metadata.values():
            source = meta.get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        return counts

    def delete_document(self, source: str) -> int:
        """Delete all chunks belonging to a given source filename."""
        ids_to_remove = [
            chunk_id for chunk_id, meta in self.metadata.items()
            if meta.get("source") == source
        ]
        if not ids_to_remove:
            return 0

        id_array = np.array(ids_to_remove, dtype="int64")
        self.index.remove_ids(id_array)

        for chunk_id in ids_to_remove:
            del self.metadata[chunk_id]

        self._persist()
        logger.info(f"Deleted {len(ids_to_remove)} chunks for '{source}'")
        return len(ids_to_remove)

    def total_chunks(self) -> int:
        return self.index.ntotal if self.index is not None else 0


@lru_cache
def get_vector_store_service() -> VectorStoreService:
    return VectorStoreService()