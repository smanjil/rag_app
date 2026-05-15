"""
ML Prague 2026 MCP Server

Exposes conference content as MCP tools so any MCP-compatible client
(Claude Desktop, Claude Code, custom agents) can search and query talks
without bundling its own RAG logic.

Run:
    python -m mcp_server.server          # stdio transport (default for Claude Desktop)
    python -m mcp_server.server --sse    # SSE transport (for HTTP clients)
"""

import sys
from typing import Annotated, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from mcp_server.config import SEARCH_K_DEFAULT, SEARCH_K_MAX
from mcp_server.tools.search import search_conference as _search
from mcp_server.tools.talks import get_talk_by_title as _get_talk
from mcp_server.tools.talks import list_talks as _list_talks

mcp = FastMCP(
    name="ml-prague-2026",
    instructions=(
        "Tools for querying ML Prague 2026 conference content. "
        "Use search_conference for open-ended questions about talk content. "
        "Use list_talks to enumerate all sessions. "
        "Use get_talk to look up a specific talk by title."
    ),
)


@mcp.tool()
def list_talks() -> List[dict]:
    """List all talks presented at ML Prague 2026 with speaker and affiliation."""
    return _list_talks()


@mcp.tool()
def get_talk(
    title: Annotated[str, Field(description="Full or partial talk title (case-insensitive)")],
) -> Optional[dict]:
    """
    Look up a specific talk by title.
    Returns talk metadata (speaker, affiliation, date, slide range) or null if not found.
    """
    return _get_talk(title)


@mcp.tool()
def search_conference(
    query: Annotated[str, Field(description="Natural language query about conference content")],
    k: Annotated[
        int,
        Field(
            description=f"Number of results to return (1–{SEARCH_K_MAX})",
            ge=1,
            le=SEARCH_K_MAX,
        ),
    ] = SEARCH_K_DEFAULT,
) -> List[dict]:
    """
    Semantically search ML Prague 2026 conference transcripts and slides.
    Returns ranked chunks with talk metadata and similarity scores.
    """
    return _search(query, k)


if __name__ == "__main__":
    transport = "sse" if "--sse" in sys.argv else "stdio"
    mcp.run(transport=transport)
