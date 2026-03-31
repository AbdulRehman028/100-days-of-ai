import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import UPLOAD_DIR
from app.models.schemas import (
    SaveResultRequest,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)
from app.services.embedding_service import EmbeddingService
from app.services.index_store import IndexStore
from app.services.ingestion import ingest_pdfs
from app.services.retrieval import RetrievalService
from app.services.session_store import SessionStore

app = FastAPI(
    title="LexSearch API",
    version="1.0.0",
    description="Legal case retrieval engine with dual-mode dense + BM25 search.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

index_store = IndexStore()
embedding_service = EmbeddingService()
retrieval_service = RetrievalService(index_store, embedding_service)
session_store = SessionStore()


@app.get("/health")
def health() -> dict:
    stats = index_store.stats()
    return {"status": "ok", **stats}


@app.get("/api/status")
def api_status() -> dict:
    stats = index_store.stats()
    return {
        "status": "ok",
        "documents": stats["documents"],
        "chunks": stats["chunks"],
        "embedding_model": embedding_service.model_name,
        "embedding_mode": embedding_service.mode,
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    saved_paths: List[Path] = []
    warnings: List[str] = []

    for upload in files:
        file_name = upload.filename or "unnamed.pdf"
        suffix = Path(file_name).suffix.lower()
        if suffix != ".pdf":
            warnings.append(f"{file_name}: skipped (only PDF is supported)")
            continue

        unique = f"{uuid.uuid4().hex[:8]}_{Path(file_name).name}"
        target = UPLOAD_DIR / unique

        with target.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)

        saved_paths.append(target)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid PDF files were uploaded")

    chunks, processed_documents, ingest_warnings = ingest_pdfs(saved_paths)
    warnings.extend(ingest_warnings)

    if not chunks:
        return UploadResponse(
            uploaded_files=len(saved_paths),
            processed_documents=processed_documents,
            created_chunks=0,
            warnings=warnings or ["No chunks produced from uploaded documents"],
        )

    texts = [c["text"] for c in chunks]
    embeddings = embedding_service.encode(texts)
    if embeddings.size == 0:
        raise HTTPException(status_code=500, detail="Embedding generation failed")

    try:
        index_store.add_chunks(chunks, embeddings)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return UploadResponse(
        uploaded_files=len(saved_paths),
        processed_documents=processed_documents,
        created_chunks=len(chunks),
        warnings=warnings,
    )


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    stats = index_store.stats()
    if stats["chunks"] == 0:
        return SearchResponse(
            mode=request.mode,
            query=request.query,
            dense_results=[],
            sparse_results=[],
            fused_results=[],
        )

    payload = retrieval_service.search(
        request.query,
        request.top_k,
        request.mode,
        request.file_name_contains,
    )

    if request.session_id:
        session_store.add_history(request.session_id, request.query, request.mode, request.top_k)

    return SearchResponse(
        mode=request.mode,
        query=request.query,
        dense_results=payload["dense_results"],
        sparse_results=payload["sparse_results"],
        fused_results=payload["fused_results"],
    )


@app.get("/api/session/{session_id}/history")
def session_history(session_id: str) -> dict:
    return {"items": session_store.get_history(session_id)}


@app.delete("/api/session/{session_id}/history")
def clear_session_history(session_id: str) -> dict:
    session_store.clear_history(session_id)
    return {"ok": True}


@app.get("/api/session/{session_id}/saved")
def session_saved(session_id: str) -> dict:
    return {"items": session_store.get_saved(session_id)}


@app.post("/api/session/{session_id}/saved")
def save_result(session_id: str, request: SaveResultRequest) -> dict:
    session_store.add_saved(session_id, request.result.model_dump())
    return {"ok": True}
