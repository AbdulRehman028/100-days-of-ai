# Day 61 - LexSearch (Full-Stack Legal Case Retrieval Engine)

LexSearch is a law-firm style document retrieval portal.
Users upload legal PDFs (cases, statutes, precedents), the backend chunks and indexes them, and the UI compares dense semantic retrieval against BM25 keyword retrieval side-by-side.

## Stack

- Backend: FastAPI, FAISS, Rank-BM25, Sentence-Transformers, pypdf, pytesseract
- Frontend: React, Tailwind CSS, Vite
- Fusion: Reciprocal Rank Fusion (RRF)
- Session UX: in-memory search history + saved results per session id

## Key Features

- PDF ingestion pipeline with OCR fallback (`pypdf` + `pdf2image` + `pytesseract`)
- Dual-mode retrieval:
  - Dense vector search (FAISS)
  - Sparse keyword search (BM25)
- Hybrid fused ranking via RRF
- Relevance-aware tokenization with common stopword filtering for cleaner BM25 rankings
- Side-by-side ranked views with highlighted snippets
- Metadata panel (case date, court, parties)
- Optional filename filter (`file_name_contains`) to restrict search to specific uploaded legal files
- Session search history and saved results
- OpenAPI docs from FastAPI (`/docs`)

## Project Structure

```text
Day-61-LexSearch/
  backend/
    app/
      core/config.py
      models/schemas.py
      services/
        embedding_service.py
        index_store.py
        ingestion.py
        retrieval.py
        session_store.py
      main.py
    data/
      uploads/
      indices/
    requirements.txt
  frontend/
    src/
      components/
      lib/
      styles/
      App.jsx
      main.jsx
    package.json
    tailwind.config.js
    vite.config.js
    .env.example
    vercel.json
```

## Run Locally

### 1) Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open API docs: `http://127.0.0.1:8000/docs`

### 2) Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Frontend runs on `http://127.0.0.1:5173`.

### 3) Generate a sample legal PDF and smoke-test upload/search

```bash
cd Day-61-LexSearch
python tools/generate_sample_legal_pdf.py
python tools/run_legal_smoke_test.py
```

This creates `test-data/legal_sample_case.pdf`, uploads it, and runs a hybrid search query.

## OCR Notes

For OCR fallback, install system tools:

- **Tesseract OCR** binary
- **Poppler** (required by `pdf2image`)

If OCR tools are unavailable, ingestion still works for text-based PDFs and returns warnings for OCR-dependent files.

## API Endpoints

- `GET /health`
- `GET /api/status`
- `POST /api/upload` (multipart PDFs)
- `POST /api/search`
- `GET /api/session/{session_id}/history`
- `DELETE /api/session/{session_id}/history`
- `GET /api/session/{session_id}/saved`
- `POST /api/session/{session_id}/saved`

## Deploy (Free)

### Frontend on Vercel

1. Import `frontend` directory as a Vercel project.
2. Set environment variable:
   - `VITE_API_BASE_URL=https://your-backend-url`
3. Deploy.

### Backend

Deploy FastAPI separately (Render/Fly.io/Railway or local VM), then point `VITE_API_BASE_URL` to it.

## Example Search Payload

```json
{
  "query": "breach of contract and liquidated damages",
  "top_k": 8,
  "mode": "hybrid",
  "session_id": "sess_123",
  "file_name_contains": "legal_sample_case"
}
```
