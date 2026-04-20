from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Day 62 - Document QA System"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    qa_model: str = "distilbert-base-uncased-distilled-squad"
    local_files_only: bool = True
    similarity_threshold: float = 0.2
    qa_confidence_threshold: float = 0.2

    data_dir: Path = Path("data")
    uploads_dir: Path = data_dir / "uploads"
    index_dir: Path = data_dir / "indices"

    max_chunk_chars: int = 1200
    chunk_overlap_chars: int = 150

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
