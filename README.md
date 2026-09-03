# RAG Chatbot (FastAPI + Groq)

A Retrieval-Augmented Generation chatbot built with a clean **service-layer
architecture**. Upload documents (PDF / DOCX / TXT / MD), and ask questions
that get answered using **Groq** for fast LLM inference, with a local
embedding model + ChromaDB for retrieval.

## Architecture

```
rag-chatbot/
├── app/
│   ├── main.py                     # FastAPI app + router wiring
│   ├── config.py                   # Settings loaded from .env
│   ├── models/
│   │   └── schemas.py              # Pydantic request/response models
│   ├── routes/                     # Thin HTTP layer
│   │   ├── chat.py                 # POST /chat
│   │   └── documents.py            # /documents upload, list, delete
│   ├── services/                   # Business logic (the core of the app)
│   │   ├── document_service.py     # Extract text + chunk files
│   │   ├── embedding_service.py    # Text -> vectors (sentence-transformers)
│   │   ├── vector_store_service.py # Chroma persistent vector DB
│   │   ├── llm_service.py          # Groq chat completion calls
│   │   └── rag_service.py          # Orchestrates the above 4 services
│   └── utils/
│       └── logger.py
├── data/
│   ├── uploads/                    # Saved uploaded files
│   └── vectorstore/                # Chroma persisted DB
├── requirements.txt
├── .env.example
└── README.md
```

**Why a service layer?** Each concern lives in its own file with a single
responsibility:
- `document_service.py` only knows how to turn files into text chunks.
- `embedding_service.py` only knows how to turn text into vectors.
- `vector_store_service.py` only knows how to store/query vectors.
- `llm_service.py` only knows how to call Groq.
- `rag_service.py` wires the four together so routes stay thin, and any
  piece (e.g. swapping Chroma for Pinecone, or Groq for another provider)
  can be replaced without touching the rest of the app.

## Setup

### 1. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your **Groq API key** (get one free at
https://console.groq.com/keys):

```
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

> Groq doesn't currently offer an embeddings endpoint, so embeddings are
> generated locally with `sentence-transformers` (free, no extra API key)
> while Groq is used purely for fast answer generation.

### 3. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now live at `http://localhost:8000`, with interactive docs at
`http://localhost:8000/docs`.

## API Endpoints

| Method | Endpoint                | Description                          |
|--------|--------------------------|--------------------------------------|
| GET    | `/health`                | Health check + config summary        |
| POST   | `/documents/upload`      | Upload & index a file (multipart)    |
| GET    | `/documents`             | List indexed documents + chunk counts|
| DELETE | `/documents/{filename}`  | Remove a document from the index     |
| POST   | `/chat`                  | Ask a question, get a grounded answer|

### Upload a document

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/your/document.pdf"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the document say about pricing?", "session_id": "user1"}'
```

Response:

```json
{
  "answer": "According to the document, pricing is...",
  "sources": [
    {"content": "...", "source": "document.pdf", "chunk_index": 3, "score": 0.82}
  ],
  "session_id": "user1"
}
```

## Notes

- Chat history is kept in-memory per `session_id` (resets on server
  restart). Swap in Redis/a DB for production use.
- Supported upload types: `.pdf`, `.docx`, `.txt`, `.md`.
- Change `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K_RESULTS` in `.env` to tune
  retrieval quality.
- Available Groq models can be checked at
  https://console.groq.com/docs/models — swap `GROQ_MODEL` in `.env`
  anytime without touching code.
