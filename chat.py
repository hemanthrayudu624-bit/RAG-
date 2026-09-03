"""
Chat endpoints: ask a question, get an answer grounded in ingested docs.
"""
from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import ChatRequest, ChatResponse, SourceChunk
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    try:
        answer, hits = rag_service.answer_query(
            query=request.query,
            session_id=request.session_id or "default",
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    sources = [
        SourceChunk(
            content=h["content"][:500],
            source=h["source"],
            chunk_index=h["chunk_index"],
            score=round(h["score"], 4) if h.get("score") is not None else None,
        )
        for h in hits
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=request.session_id or "default",
    )
