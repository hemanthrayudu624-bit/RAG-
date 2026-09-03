"""
Document management endpoints: upload, list, delete knowledge-base files.
"""
import os
import shutil

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from app.config import get_settings
from app.models.schemas import (
    UploadResponse,
    DocumentsListResponse,
    DocumentInfo,
    DeleteResponse,
)
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(prefix="/documents", tags=["Documents"])
settings = get_settings()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    os.makedirs(settings.upload_dir, exist_ok=True)
    save_path = os.path.join(settings.upload_dir, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks_created = rag_service.ingest_file(save_path, file.filename)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to process file: {exc}"
        ) from exc

    return UploadResponse(
        filename=file.filename,
        chunks_created=chunks_created,
        status="indexed" if chunks_created else "no_text_found",
    )


@router.get("", response_model=DocumentsListResponse)
def list_documents(rag_service: RAGService = Depends(get_rag_service)):
    doc_counts = rag_service.list_documents()
    documents = [
        DocumentInfo(filename=name, chunk_count=count)
        for name, count in doc_counts.items()
    ]
    return DocumentsListResponse(
        documents=documents, total_chunks=rag_service.total_chunks()
    )


@router.delete("/{filename}", response_model=DeleteResponse)
def delete_document(
    filename: str, rag_service: RAGService = Depends(get_rag_service)
):
    deleted_count = rag_service.delete_document(filename)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"'{filename}' not found")

    file_path = os.path.join(settings.upload_dir, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return DeleteResponse(filename=filename, status="deleted")
