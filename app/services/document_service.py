"""
DocumentService
----------------
Responsible for turning raw uploaded files (txt, pdf, docx, md) into
clean text and splitting that text into overlapping chunks that are
ready to be embedded.
"""
import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DocumentService:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ---------- text extraction ----------

    def extract_text(self, file_path: str) -> str:
        """Extract raw text from a supported file type."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._extract_pdf(file_path)
        elif ext == ".docx":
            return self._extract_docx(file_path)
        elif ext in (".txt", ".md"):
            return self._extract_plain_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
        return "\n\n".join(text_parts)

    @staticmethod
    def _extract_docx(file_path: str) -> str:
        import docx

        doc = docx.Document(file_path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    @staticmethod
    def _extract_plain_text(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ---------- chunking ----------

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks suitable for embedding."""
        if not text or not text.strip():
            return []
        chunks = self.splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]

    def process_file(self, file_path: str) -> List[str]:
        """Extract + chunk a file in one step."""
        text = self.extract_text(file_path)
        chunks = self.chunk_text(text)
        logger.info(f"Processed '{file_path}' into {len(chunks)} chunks")
        return chunks


def get_document_service() -> DocumentService:
    return DocumentService()
