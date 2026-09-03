"""
Pydantic request/response models shared across routes and services.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's question")
    session_id: Optional[str] = Field(
        default="default", description="Conversation/session identifier"
    )
    top_k: Optional[int] = Field(
        default=None, description="Override number of chunks to retrieve"
    )


class SourceChunk(BaseModel):
    content: str
    source: str
    chunk_index: int
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk] = []
    session_id: str


class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    status: str


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int


class DocumentsListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_chunks: int


class DeleteResponse(BaseModel):
    filename: str
    status: str


class HealthResponse(BaseModel):
    status: str
    groq_model: str
    embedding_model: str
    total_chunks: int
