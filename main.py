"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models.schemas import HealthResponse
from app.routes import chat, documents
from app.services.rag_service import RAGService, get_rag_service

settings = get_settings()

app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation chatbot powered by Groq, "
    "built with a clean service-layer architecture.",
    version="1.0.0",
)

origins = (
    ["*"] if settings.cors_origins.strip() == "*" else
    [o.strip() for o in settings.cors_origins.split(",")]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/", tags=["Health"])
def root():
    return {"message": "RAG Chatbot API is running. See /docs for the API docs."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(rag_service: RAGService = Depends(get_rag_service)):
    return HealthResponse(
        status="ok",
        groq_model=settings.groq_model,
        embedding_model=settings.embedding_model,
        total_chunks=rag_service.total_chunks(),
    )
