import re
from typing import List


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

# Keep this intentionally small and practical to avoid removing legal terms.
COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "to",
    "under",
    "was",
    "were",
    "will",
    "with",
}


def tokenize_for_search(text: str) -> List[str]:
    tokens = [tok.lower() for tok in TOKEN_RE.findall(text)]
    return [tok for tok in tokens if len(tok) > 1 and tok not in COMMON_STOPWORDS]
