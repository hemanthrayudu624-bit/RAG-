"""
RAGService
-----------
Orchestrates the other services to implement the end-to-end RAG flow:

  1. DocumentService  -> extract + chunk uploaded files
  2. EmbeddingService -> embed chunks / queries
  3. VectorStoreService -> store / retrieve chunks
  4. LLMService       -> generate the final answer with Groq

Keeping this as its own layer means routes stay thin and each
capability (parsing, embedding, storage, generation) stays swappable
in isolation.
"""
from collections import defaultdict
from typing import Dict, List

from app.services.document_service import DocumentService, get_document_service
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.llm_service import LLMService, get_llm_service
from app.services.vector_store_service import (
    VectorStoreService,
    get_vector_store_service,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# very small in-memory chat history store, keyed by session_id
_chat_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)
_MAX_HISTORY_TURNS = 6


class RAGService:
    def __init__(
        self,
        document_service: DocumentService,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        llm_service: LLMService,
    ):
        self.documents = document_service
        self.embeddings = embedding_service
        self.vector_store = vector_store_service
        self.llm = llm_service

    # ---------- ingestion ----------

    def ingest_file(self, file_path: str, filename: str) -> int:
        chunks = self.documents.process_file(file_path)
        if not chunks:
            return 0
        vectors = self.embeddings.embed_texts(chunks)
        return self.vector_store.add_chunks(chunks, vectors, source=filename)

    def delete_document(self, filename: str) -> int:
        return self.vector_store.delete_document(filename)

    def list_documents(self):
        return self.vector_store.list_documents()

    # ---------- chat ----------

    def answer_query(self, query: str, session_id: str = "default", top_k: int = None):
        query_vector = self.embeddings.embed_query(query)
        hits = self.vector_store.query(query_vector, top_k=top_k)

        context_chunks = [h["content"] for h in hits]
        history = _chat_histories[session_id][-_MAX_HISTORY_TURNS:]

        answer = self.llm.generate_answer(
            query=query, context_chunks=context_chunks, chat_history=history
        )

        _chat_histories[session_id].append({"role": "user", "content": query})
        _chat_histories[session_id].append({"role": "assistant", "content": answer})

        return answer, hits

    def total_chunks(self) -> int:
        return self.vector_store.total_chunks()


def get_rag_service() -> RAGService:
    """FastAPI dependency factory wiring together all underlying services."""
    return RAGService(
        document_service=get_document_service(),
        embedding_service=get_embedding_service(),
        vector_store_service=get_vector_store_service(),
        llm_service=get_llm_service(),
    )
