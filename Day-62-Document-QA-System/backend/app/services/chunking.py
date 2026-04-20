import re
from typing import Dict, List


def split_into_sections(text: str) -> List[Dict[str, str]]:
    lines = [line.strip() for line in text.split("\n")]
    sections: List[Dict[str, str]] = []

    current_title = "Introduction"
    current_lines: List[str] = []

    heading_pattern = re.compile(r"^(\d+(?:\.\d+)*)?[\)\.]?\s?[A-Z][A-Za-z0-9\s\-]{1,80}$")

    for line in lines:
        if not line:
            continue

        words = line.split()
        has_sentence_punct = any(ch in line for ch in [",", ".", ":", ";", "?", "!"])
        is_heading = (
            len(line) <= 60
            and len(words) <= 8
            and not has_sentence_punct
            and heading_pattern.match(line) is not None
        )

        if is_heading:
            if current_lines:
                sections.append({"title": current_title, "text": "\n".join(current_lines).strip()})
            current_title = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({"title": current_title, "text": "\n".join(current_lines).strip()})

    return sections


def chunk_section_text(section_text: str, max_chars: int, overlap_chars: int) -> List[str]:
    if len(section_text) <= max_chars:
        return [section_text]

    chunks: List[str] = []
    start = 0

    while start < len(section_text):
        end = min(start + max_chars, len(section_text))
        chunk = section_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(section_text):
            break

        start = max(0, end - overlap_chars)

    return chunks


def build_section_chunks(text: str, source_document: str, max_chars: int, overlap_chars: int) -> List[Dict[str, str]]:
    sections = split_into_sections(text)
    records: List[Dict[str, str]] = []

    for section in sections:
        section_chunks = chunk_section_text(section["text"], max_chars, overlap_chars)
        for idx, chunk in enumerate(section_chunks, start=1):
            records.append(
                {
                    "source_document": source_document,
                    "section": section["title"],
                    "chunk_id": f"{source_document}:{section['title']}:{idx}",
                    "text": chunk,
                }
            )

    return records
