"""
EmbeddingService
-----------------
Wraps a local sentence-transformers model to turn text into vectors.

Groq does not currently expose an embeddings endpoint, so embeddings are
generated locally (fast, free, no API key needed) while Groq is used only
for the final answer generation (see llm_service.py).
"""
from functools import lru_cache
from typing import List

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingService:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts (used when indexing document chunks)."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Singleton so the (relatively heavy) model is only loaded once."""
    return EmbeddingService()
