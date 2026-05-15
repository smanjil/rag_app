"""
Unit tests for MCP server tools.

Pinecone and SentenceTransformer are mocked so tests run without credentials
or network access.  The real talks.json is used for talk-lookup tests.
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy optional deps before any mcp_server import
# ---------------------------------------------------------------------------

def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


_dotenv = _stub_module("dotenv")
_dotenv.load_dotenv = lambda *a, **kw: None  # no-op

_pinecone = _stub_module("pinecone")
_st = _stub_module("sentence_transformers")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TALKS = [
    {
        "slide_start": 1,
        "slide_end": 45,
        "title": "Why Good Models Fail in Production",
        "speaker": "Alice Example",
        "affiliation": "TestCorp",
        "date": "2026-05-06",
    },
    {
        "slide_start": 47,
        "slide_end": 79,
        "title": "Sequential Recommendations for Search Reranking",
        "speaker": "Bob Sample",
        "affiliation": "AnotherCo",
        "date": "2026-05-06",
    },
]


@pytest.fixture(autouse=True)
def reset_tool_caches():
    """Reset module-level singletons between tests."""
    import mcp_server.tools.talks as talks_mod
    import mcp_server.tools.search as search_mod

    talks_mod._cache = None
    search_mod._embedder = None
    search_mod._index = None
    yield
    talks_mod._cache = None
    search_mod._embedder = None
    search_mod._index = None


@pytest.fixture()
def talks_json(tmp_path: Path) -> Path:
    p = tmp_path / "talks.json"
    p.write_text(json.dumps(SAMPLE_TALKS))
    return p


# ---------------------------------------------------------------------------
# list_talks
# ---------------------------------------------------------------------------

class TestListTalks:
    def test_returns_all_talks(self, talks_json):
        with patch("mcp_server.tools.talks.TALKS_JSON_PATH", talks_json):
            import mcp_server.tools.talks as mod
            result = mod.list_talks()
        assert len(result) == 2
        assert result[0]["speaker"] == "Alice Example"

    def test_caches_after_first_read(self, talks_json):
        with patch("mcp_server.tools.talks.TALKS_JSON_PATH", talks_json):
            import mcp_server.tools.talks as mod
            first = mod.list_talks()
            second = mod.list_talks()
        assert first is second  # same list object from cache

    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "no_such_file.json"
        with patch("mcp_server.tools.talks.TALKS_JSON_PATH", missing):
            import mcp_server.tools.talks as mod
            with pytest.raises(FileNotFoundError):
                mod.list_talks()


# ---------------------------------------------------------------------------
# get_talk_by_title
# ---------------------------------------------------------------------------

class TestGetTalkByTitle:
    def _run(self, title, talks_json):
        with patch("mcp_server.tools.talks.TALKS_JSON_PATH", talks_json):
            import mcp_server.tools.talks as mod
            return mod.get_talk_by_title(title)

    def test_exact_match(self, talks_json):
        result = self._run("Why Good Models Fail in Production", talks_json)
        assert result is not None
        assert result["speaker"] == "Alice Example"

    def test_case_insensitive(self, talks_json):
        result = self._run("why good models fail in production", talks_json)
        assert result is not None

    def test_partial_match(self, talks_json):
        result = self._run("Sequential Recommendations", talks_json)
        assert result is not None
        assert result["speaker"] == "Bob Sample"

    def test_word_overlap_fallback(self, talks_json):
        result = self._run("models fail production", talks_json)
        assert result is not None

    def test_no_match_returns_none(self, talks_json):
        result = self._run("completely unrelated xyz123", talks_json)
        assert result is None

    def test_empty_query_returns_none(self, talks_json):
        result = self._run("", talks_json)
        assert result is None

    def test_whitespace_only_returns_none(self, talks_json):
        result = self._run("   ", talks_json)
        assert result is None


# ---------------------------------------------------------------------------
# search_conference
# ---------------------------------------------------------------------------

class TestSearchConference:
    def _make_match(self, score, text, talk_title, speaker):
        m = MagicMock()
        m.score = score
        m.metadata = {
            "text": text,
            "talk_title": talk_title,
            "speaker": speaker,
            "affiliation": "TestCo",
            "date": "2026-05-06",
            "slide_index": 1,
            "timestamp_fmt": "00:01:00",
            "city": "Prague",
            "country": "Czechia",
        }
        return m

    def _mock_deps(self, matches):
        mock_vector = MagicMock()
        mock_vector.tolist.return_value = [0.0] * 384
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = mock_vector

        mock_index = MagicMock()
        mock_index.query.return_value = MagicMock(matches=matches)

        return mock_embedder, mock_index

    def test_returns_ranked_results(self):
        matches = [
            self._make_match(0.95, "chunk about production ML", "Why Good Models Fail", "Alice"),
            self._make_match(0.80, "another chunk", "Sequential Recs", "Bob"),
        ]
        mock_embedder, mock_index = self._mock_deps(matches)

        with (
            patch("mcp_server.tools.search._get_embedder", return_value=mock_embedder),
            patch("mcp_server.tools.search._get_index", return_value=mock_index),
        ):
            from mcp_server.tools.search import search_conference
            results = search_conference("production ML failures", k=2)

        assert len(results) == 2
        assert results[0]["score"] == 0.95
        assert results[0]["text"] == "chunk about production ML"
        assert results[0]["metadata"]["speaker"] == "Alice"

    def test_k_clamped_to_max(self):
        mock_embedder, mock_index = self._mock_deps([])
        mock_index.query.return_value = MagicMock(matches=[])

        with (
            patch("mcp_server.tools.search._get_embedder", return_value=mock_embedder),
            patch("mcp_server.tools.search._get_index", return_value=mock_index),
        ):
            from mcp_server.tools.search import search_conference
            from mcp_server.config import SEARCH_K_MAX
            search_conference("test", k=9999)

        _, call_kwargs = mock_index.query.call_args
        assert call_kwargs["top_k"] <= SEARCH_K_MAX

    def test_empty_query_raises(self):
        from mcp_server.tools.search import search_conference
        with pytest.raises(ValueError, match="non-empty"):
            search_conference("")

    def test_whitespace_query_raises(self):
        from mcp_server.tools.search import search_conference
        with pytest.raises(ValueError, match="non-empty"):
            search_conference("   ")

    def test_empty_results(self):
        mock_embedder, mock_index = self._mock_deps([])

        with (
            patch("mcp_server.tools.search._get_embedder", return_value=mock_embedder),
            patch("mcp_server.tools.search._get_index", return_value=mock_index),
        ):
            from mcp_server.tools.search import search_conference
            results = search_conference("obscure query no match")

        assert results == []
