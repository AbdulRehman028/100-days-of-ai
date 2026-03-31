import re
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from pypdf import PdfReader

from app.core.config import CHUNK_OVERLAP, CHUNK_SIZE


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_pdf_text_with_ocr(path: Path) -> Tuple[str, str]:
    """
    OCR fallback using pdf2image + pytesseract.
    Returns (text, warning_message).
    warning_message is empty on success.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        return "", "OCR fallback unavailable (install pdf2image + pytesseract and system binaries)."

    try:
        images = convert_from_path(str(path), dpi=220)
    except Exception as exc:
        return "", f"OCR conversion failed: {exc}"

    page_texts = []
    for image in images:
        try:
            page_texts.append(pytesseract.image_to_string(image) or "")
        except Exception:
            continue

    text = "\n".join(page_texts).strip()
    if not text:
        return "", "OCR extracted no readable text."
    return text, ""


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            pivot = max(start + int(chunk_size * 0.55), start)
            boundary = text.rfind(" ", pivot, end)
            if boundary > start + 120:
                end = boundary

        chunk = text[start:end].strip()
        if len(chunk) >= 80:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def extract_metadata(text: str, file_name: str) -> Dict[str, str]:
    metadata = {
        "file_name": file_name,
        "case_date": "Unknown",
        "court": "Unknown",
        "parties": "Unknown",
    }

    date_match = re.search(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},\s+\d{4})\b",
        text,
        re.IGNORECASE,
    )
    if date_match:
        metadata["case_date"] = date_match.group(0)

    court_match = re.search(
        r"\b(supreme\s+court|high\s+court|district\s+court|court\s+of\s+appeals|session\s+court)\b",
        text,
        re.IGNORECASE,
    )
    if court_match:
        metadata["court"] = court_match.group(0).title()

    parties_match = re.search(
        r"\b([A-Z][A-Za-z0-9&.,'\-]*(?:\s+[A-Z][A-Za-z0-9&.,'\-]*){0,4}\s+v\.?\s+"
        r"[A-Z][A-Za-z0-9&.,'\-]*(?:\s+[A-Z][A-Za-z0-9&.,'\-]*){0,4})\b",
        text,
    )
    if parties_match:
        parties = parties_match.group(0).strip()
        parties = re.sub(r"\s+(date|dated|judgment|order|section)\b.*$", "", parties, flags=re.IGNORECASE)
        metadata["parties"] = parties.strip()

    return metadata


def ingest_pdfs(paths: List[Path]) -> Tuple[List[Dict], int, List[str]]:
    """
    Returns (chunks, processed_document_count, warnings).
    """
    all_chunks: List[Dict] = []
    warnings: List[str] = []
    processed_docs = 0

    for path in paths:
        try:
            text = extract_pdf_text(path)
        except Exception as exc:
            warnings.append(f"{path.name}: failed to read PDF ({exc})")
            continue

        if len(clean_text(text)) < 300:
            ocr_text, warning = extract_pdf_text_with_ocr(path)
            if warning:
                warnings.append(f"{path.name}: {warning}")
            if ocr_text:
                text = f"{text}\n{ocr_text}".strip()

        cleaned = clean_text(text)
        if len(cleaned) < 80:
            warnings.append(f"{path.name}: no usable text extracted")
            continue

        metadata = extract_metadata(cleaned, path.name)
        chunk_list = chunk_text(cleaned)
        if not chunk_list:
            warnings.append(f"{path.name}: chunking produced no chunks")
            continue

        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        for idx, chunk in enumerate(chunk_list):
            chunk_id = f"{doc_id}_c{idx:04d}"
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "file_name": path.name,
                    "text": chunk,
                    "metadata": metadata,
                }
            )

        processed_docs += 1

    return all_chunks, processed_docs, warnings
