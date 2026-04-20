import re
from pathlib import Path

from docx import Document
from pypdf import PdfReader


ALLOWED_EXTENSIONS = {".pdf", ".docx"}


def validate_extension(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_text_from_docx(file_path: Path) -> str:
    document = Document(str(file_path))
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs)


def clean_text(raw_text: str) -> str:
    text = raw_text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif suffix == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return clean_text(text)
