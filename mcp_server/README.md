# ML Prague 2026 MCP Server

Standalone MCP server exposing ML Prague 2026 conference content as tools.
Separate from the RAG app — can be tested and deployed independently.

## Tools

| Tool | Description |
|------|-------------|
| `list_talks` | Return all talks (title, speaker, affiliation, date, slide range) |
| `get_talk` | Look up a talk by full or partial title |
| `search_conference` | Semantic search over transcripts and slides via Pinecone |

## Setup

```bash
# From rag_app root
python -m venv .venv_mcp
source .venv_mcp/bin/activate
pip install -r mcp_server/requirements-mcp.txt
```

The server reads credentials from the `.env` file in the `rag_app` root. Required:

```
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=ml-conference   # default
PINECONE_ENV=us-east-1              # default
```

## Running

```bash
# stdio transport — for Claude Desktop / Claude Code
python -m mcp_server.server

# SSE transport — for HTTP clients
python -m mcp_server.server --sse
```

### Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "ml-prague-2026": {
      "command": "/path/to/rag_app/.venv_mcp/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/rag_app"
    }
  }
}
```

### Claude Code (from rag_app root)

```bash
claude mcp add ml-prague-2026 -- .venv_mcp/bin/python -m mcp_server.server
```

## Tests

```bash
# No credentials needed — Pinecone and SentenceTransformer are mocked
pytest mcp_server/tests/ -v
```
