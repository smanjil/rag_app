import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed; rely on env vars being set externally


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip().strip("'\"") or default


PINECONE_API_KEY: str = _get("PINECONE_API_KEY")
PINECONE_INDEX_NAME: str = _get("PINECONE_INDEX_NAME", "ml-conference")
PINECONE_ENV: str = _get("PINECONE_ENV", "us-east-1")

EMBEDDING_MODEL: str = _get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM: int = 384

TALKS_JSON_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "mlprague" / "talks.json"

SEARCH_K_DEFAULT: int = 5
SEARCH_K_MAX: int = 20
