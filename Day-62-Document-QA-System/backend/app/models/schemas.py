from typing import List

from pydantic import BaseModel, Field


class UploadResult(BaseModel):
    file_name: str
    status: str
    chunks_indexed: int = 0
    message: str


class UploadResponse(BaseModel):
    processed_files: int
    results: List[UploadResult]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=12)


class SourceReference(BaseModel):
    source_document: str
    section: str
    relevance_score: float
    relevant_text: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    sources: List[SourceReference]
