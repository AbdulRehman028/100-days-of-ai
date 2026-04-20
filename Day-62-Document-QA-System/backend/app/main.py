from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.models.schemas import AskRequest, AskResponse, UploadResponse
from app.services.rag_service import rag_service

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": settings.app_name}


@app.get("/api/status")
def status() -> dict:
    return {
        "ready": rag_service.is_ready,
        "indexed_chunks": rag_service.vector_store.total_chunks,
        "embedding_model": settings.embedding_model,
        "qa_model": settings.qa_model,
        "local_files_only": settings.local_files_only,
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    result = await rag_service.ingest_files(files)
    return UploadResponse(**result)


@app.post("/api/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    try:
        result = rag_service.answer_question(question=payload.question, top_k=payload.top_k)
        return AskResponse(**result)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(ex)}")
