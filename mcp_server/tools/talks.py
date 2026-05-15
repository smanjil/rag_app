"""Tools for accessing ML Prague talk metadata from talks.json."""

import json
import threading
from pathlib import Path
from typing import Any, List, Optional

from mcp_server.config import TALKS_JSON_PATH

_cache: Optional[List[dict]] = None
_lock = threading.Lock()


def _load_talks() -> List[dict]:
    global _cache
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is not None:
            return _cache
        if not TALKS_JSON_PATH.exists():
            raise FileNotFoundError(f"talks.json not found at {TALKS_JSON_PATH}")
        _cache = json.loads(TALKS_JSON_PATH.read_text(encoding="utf-8"))
    return _cache


def list_talks() -> List[dict]:
    """Return all talks from the conference."""
    return _load_talks()


def get_talk_by_title(title: str) -> Optional[dict]:
    """
    Find a talk by exact or partial case-insensitive title match.
    Returns the best match or None if nothing is close enough.
    """
    if not title or not title.strip():
        return None

    needle = title.strip().lower()
    talks = _load_talks()

    # Exact match first
    for talk in talks:
        if talk.get("title", "").lower() == needle:
            return talk

    # Partial match — return first talk whose title contains the query
    for talk in talks:
        if needle in talk.get("title", "").lower():
            return talk

    # Word-overlap fallback: find talk with most shared words
    needle_words = set(needle.split())
    best, best_score = None, 0
    for talk in talks:
        haystack_words = set(talk.get("title", "").lower().split())
        score = len(needle_words & haystack_words)
        if score > best_score:
            best, best_score = talk, score

    # Require at least 2 overlapping words to avoid false positives
    return best if best_score >= 2 else None
