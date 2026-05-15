"""Pinecone vector search tool for ML Prague conference content."""

import threading
from typing import Any, List

from mcp_server.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_ENV,
    PINECONE_INDEX_NAME,
    SEARCH_K_DEFAULT,
    SEARCH_K_MAX,
)

_lock = threading.Lock()
_embedder = None
_index = None


def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is not None:
            return _embedder
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_index():
    global _index
    if _index is not None:
        return _index
    with _lock:
        if _index is not None:
            return _index
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is not set")
        from pinecone import Pinecone

        pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = pc.Index(PINECONE_INDEX_NAME)
    return _index


def search_conference(query: str, k: int = SEARCH_K_DEFAULT) -> List[dict]:
    """
    Search the ML Prague 2026 conference vector store.

    Returns up to k chunks ranked by semantic similarity, each with:
      - text: the chunk content
      - score: cosine similarity (0–1)
      - metadata: talk title, speaker, affiliation, date, slide index, timestamp
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    k = max(1, min(k, SEARCH_K_MAX))

    embedder = _get_embedder()
    vector = embedder.encode(query.strip()).tolist()
    assert len(vector) == EMBEDDING_DIM, f"Unexpected embedding dim: {len(vector)}"

    index = _get_index()
    response = index.query(vector=vector, top_k=k, include_metadata=True)

    results = []
    for match in response.matches:
        meta = match.metadata or {}
        results.append(
            {
                "text": meta.get("text", ""),
                "score": round(float(match.score), 4),
                "metadata": {
                    "talk_title": meta.get("talk_title", ""),
                    "speaker": meta.get("speaker", ""),
                    "affiliation": meta.get("affiliation", ""),
                    "date": meta.get("date", ""),
                    "slide_index": meta.get("slide_index"),
                    "timestamp_fmt": meta.get("timestamp_fmt", ""),
                    "city": meta.get("city", ""),
                    "country": meta.get("country", ""),
                },
            }
        )
    return results
