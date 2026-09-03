"""
Centralized application configuration.
Reads values from environment variables / .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- Groq LLM ----
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.3
    groq_max_tokens: int = 1024

    # ---- Embeddings ----
    embedding_model: str = "all-MiniLM-L6-v2"

    # ---- Vector store ----
    vector_store_dir: str = "data/vectorstore"
    collection_name: str = "rag_documents"

    # ---- Chunking ----
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # ---- Retrieval ----
    top_k_results: int = 4

    # ---- App ----
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    upload_dir: str = "data/uploads"
    max_upload_size_mb: int = 20
    cors_origins: str = "*"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance, so the .env file is only parsed once."""
    return Settings()
