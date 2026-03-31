from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


SearchMode = Literal["dense", "sparse", "hybrid"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(8, ge=1, le=30)
    mode: SearchMode = "hybrid"
    session_id: Optional[str] = None
    file_name_contains: Optional[str] = Field(default=None, max_length=120)


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    file_name: str
    score: float
    rank: int
    snippet: str
    highlighted_snippet: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    mode: SearchMode
    query: str
    dense_results: List[SearchResult]
    sparse_results: List[SearchResult]
    fused_results: List[SearchResult]


class UploadResponse(BaseModel):
    uploaded_files: int
    processed_documents: int
    created_chunks: int
    warnings: List[str]


class SaveResultRequest(BaseModel):
    result: SearchResult


class SessionHistoryItem(BaseModel):
    timestamp: str
    query: str
    mode: SearchMode
    top_k: int


class SessionSavedItem(BaseModel):
    timestamp: str
    result: SearchResult
