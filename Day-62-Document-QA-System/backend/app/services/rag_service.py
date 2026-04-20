import uuid
import re
from typing import Dict, List

from fastapi import UploadFile
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from app.core.config import settings
from app.services.chunking import build_section_chunks
from app.services.document_parser import extract_text, validate_extension
from app.services.vector_store import VectorStore


class RAGService:
    def __init__(self) -> None:
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        settings.index_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = SentenceTransformer(
            settings.embedding_model,
            cache_folder=None,
            local_files_only=settings.local_files_only,
        )
        self.qa_pipeline = pipeline(
            "question-answering",
            model=settings.qa_model,
            tokenizer=settings.qa_model,
            local_files_only=settings.local_files_only,
        )
        self.vector_store = VectorStore(settings.index_dir)

    @property
    def is_ready(self) -> bool:
        return self.embedding_model is not None and self.qa_pipeline is not None

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return vectors.tolist()

    def _embed_query(self, query: str) -> List[float]:
        vector = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
        return vector[0].tolist()

    async def ingest_files(self, files: List[UploadFile]) -> Dict:
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        settings.index_dir.mkdir(parents=True, exist_ok=True)

        results = []
        processed_files = 0

        for file in files:
            if not validate_extension(file.filename):
                results.append(
                    {
                        "file_name": file.filename,
                        "status": "skipped",
                        "chunks_indexed": 0,
                        "message": "Unsupported format. Allowed: .pdf, .docx",
                    }
                )
                continue

            unique_name = f"{uuid.uuid4().hex}_{file.filename}"
            destination = settings.uploads_dir / unique_name

            content = await file.read()
            destination.write_bytes(content)

            try:
                extracted_text = extract_text(destination)
                if not extracted_text:
                    results.append(
                        {
                            "file_name": file.filename,
                            "status": "skipped",
                            "chunks_indexed": 0,
                            "message": "No readable text found in document.",
                        }
                    )
                    continue

                records = build_section_chunks(
                    text=extracted_text,
                    source_document=file.filename,
                    max_chars=settings.max_chunk_chars,
                    overlap_chars=settings.chunk_overlap_chars,
                )

                if not records:
                    results.append(
                        {
                            "file_name": file.filename,
                            "status": "skipped",
                            "chunks_indexed": 0,
                            "message": "No chunks generated from document.",
                        }
                    )
                    continue

                texts = [r["text"] for r in records]
                vectors = self._embed_texts(texts)
                self.vector_store.add(vectors=vectors, metadata_rows=records)

                processed_files += 1
                results.append(
                    {
                        "file_name": file.filename,
                        "status": "indexed",
                        "chunks_indexed": len(records),
                        "message": "Document indexed successfully.",
                    }
                )
            except Exception as ex:
                results.append(
                    {
                        "file_name": file.filename,
                        "status": "error",
                        "chunks_indexed": 0,
                        "message": f"Failed to index: {str(ex)}",
                    }
                )

        return {"processed_files": processed_files, "results": results}

    def _build_context(self, retrieved: List[Dict[str, str]]) -> str:
        context_blocks = []
        for idx, row in enumerate(retrieved, start=1):
            context_blocks.append(
                f"[Source {idx}]\\n"
                f"Document: {row['source_document']}\\n"
                f"Section: {row['section']}\\n"
                f"Content: {row['text']}"
            )
        return "\\n\\n".join(context_blocks)

    def answer_question(self, question: str, top_k: int) -> Dict:
        if self.vector_store.is_empty:
            return {
                "answer": "not found",
                "grounded": False,
                "sources": [],
            }

        query_vector = self._embed_query(question)
        retrieved = self.vector_store.search(query_vector, top_k=top_k)

        if not retrieved:
            return {
                "answer": "not found",
                "grounded": False,
                "sources": [],
            }

        query_tokens = [tok.lower() for tok in re.findall(r"[A-Za-z0-9]+", question) if len(tok) >= 3]
        is_short_query = len(question.split()) <= 2 and bool(query_tokens)

        best_score = max(item["score"] for item in retrieved)
        threshold = settings.similarity_threshold * 0.5 if is_short_query else settings.similarity_threshold
        if best_score < threshold:
            return {
                "answer": "not found",
                "grounded": False,
                "sources": [],
            }

        sources = [
            {
                "source_document": row["source_document"],
                "section": row["section"],
                "relevance_score": round(float(row["score"]), 4),
                "relevant_text": row["text"],
            }
            for row in retrieved[:3]
        ]

        # For terse queries like "skill", return grounded snippets directly.
        if is_short_query:
            snippets: List[str] = []
            for row in retrieved[:5]:
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", row["text"]):
                    sentence_lc = sentence.lower().strip()
                    if not sentence_lc:
                        continue
                    if any(tok in sentence_lc or tok.rstrip("s") in sentence_lc for tok in query_tokens):
                        snippets.append(sentence.strip())
                if len(snippets) >= 3:
                    break

            if snippets:
                return {
                    "answer": " | ".join(snippets[:3]),
                    "grounded": True,
                    "sources": sources,
                }

            return {
                "answer": retrieved[0]["text"][:320],
                "grounded": True,
                "sources": sources,
            }

        top_context = "\n\n".join(row["text"] for row in retrieved[:3])
        qa_result = self.qa_pipeline(question=question, context=top_context)
        answer = (qa_result.get("answer") or "").strip()
        confidence = float(qa_result.get("score", 0.0))

        if not answer or confidence < settings.qa_confidence_threshold:
            # Fallback: return top retrieved snippet to stay grounded instead of hallucinating.
            return {
                "answer": retrieved[0]["text"][:320],
                "grounded": True,
                "sources": sources,
            }

        return {
            "answer": answer,
            "grounded": True,
            "sources": sources,
        }


rag_service = RAGService()
