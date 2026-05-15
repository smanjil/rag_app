# RAG App — ML Prague 2026

Production-style RAG service + MCP server built on top of ML Prague 2026 conference content.

- **RAG service** (`rag_service/`) — FastAPI app: Pinecone retrieval, Mistral generation, Langfuse tracing, SQLite sessions, built-in UI
- **MCP server** (`mcp_server/`) — standalone Model Context Protocol server exposing conference content as tools for Claude Desktop / Claude Code
- **Ingest pipeline** (`scripts/process_mlprague.py`) — 5-phase pipeline: WhisperX transcription → slide frame extraction → Mistral Pixtral OCR → transcript-slide alignment → Pinecone ingestion

## Repository Layout

```
rag_app/
  rag_service/          # FastAPI RAG app
    main.py             # app lifecycle + routes
    config.py           # env and runtime config
    models.py           # request schemas
    database.py         # SQLite persistence
    logic.py            # prompt / eval / Langfuse helpers
  mcp_server/           # MCP server (standalone, separately testable)
    server.py           # FastMCP entrypoint — registers 3 tools
    config.py           # env loading, paths, constants
    tools/
      talks.py          # list_talks(), get_talk_by_title()
      search.py         # search_conference() via Pinecone
    tests/
      test_tools.py     # 15 unit tests (no credentials needed)
    requirements-mcp.txt
    README.md
  agents/               # offline RAG quality evaluation agent
  scripts/
    process_mlprague.py # end-to-end conference video ingest pipeline
    export_openapi.py   # export OpenAPI artifact
  data/mlprague/        # conference data (talks.json, transcripts, slide texts)
  openapi/openapi.json  # checked-in API contract
  tests/                # pytest suite for RAG service
  app.py                # compatibility entrypoint
  requirements.txt      # RAG service deps
  requirements-dev.txt  # dev/test + pipeline deps
  requirements-mcp.txt  # MCP server deps
```

## Data

Conference data lives under `data/mlprague/`:

| File | Description |
|------|-------------|
| `talks.json` | 10 talks with title, speaker, affiliation, date, slide range |
| `transcript.json` | WhisperX word-aligned transcript |
| `slide_texts.json` | Mistral Pixtral OCR output per slide frame |
| `aligned.json` | Merged transcript + slide text per timestamp |

Large binary files (video, slide frames) are gitignored.

## MCP Server

Exposes three tools to any MCP-compatible client:

| Tool | Description |
|------|-------------|
| `list_talks` | Return all talks (title, speaker, affiliation, date) |
| `get_talk` | Look up a talk by full or partial title |
| `search_conference` | Semantic search over transcripts and slides |

### Setup

```bash
python -m venv .venv_mcp
source .venv_mcp/bin/activate
pip install -r mcp_server/requirements-mcp.txt
```

### Run (stdio — for Claude Desktop)

```bash
python -m mcp_server.server
```

### Claude Desktop config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ml-prague-2026": {
      "command": "/path/to/rag_app/.venv_mcp/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/rag_app",
      "env": {
        "PYTHONPATH": "/path/to/rag_app"
      }
    }
  }
}
```

### Claude Code

```bash
claude mcp add ml-prague-2026 -- .venv_mcp/bin/python -m mcp_server.server
```

### MCP Tests

Run without credentials — Pinecone and SentenceTransformer are mocked:

```bash
pytest mcp_server/tests/ -v
```

## Ingest Pipeline

Processes raw conference video into Pinecone vectors in 5 phases:

```
Phase 1 — WhisperX transcription (GPU)
Phase 2 — Slide frame extraction + pHash deduplication
Phase 3 — Mistral Pixtral OCR per slide frame
Phase 4 — Transcript-slide alignment by timestamp
Phase 5 — Pinecone ingestion with enriched metadata
```

Each chunk stored in Pinecone includes: talk title, speaker, affiliation, date, city, country, slide index, and timestamp.

```bash
pip install -r requirements-pipeline.txt

# Full pipeline
python scripts/process_mlprague.py

# Resume from a specific phase
python scripts/process_mlprague.py --skip-transcription   # skip Phase 1
python scripts/process_mlprague.py --skip-slides          # skip Phases 1+2
```

## RAG Service

### Requirements

- Python 3.9+
- Pinecone API key + index (`ml-conference`, dimension 384, cosine)
- Mistral API key
- Langfuse keys (optional; `/history/langfuse` requires them)

### Environment Variables

Create `.env` in repo root:

```env
MISTRAL_API_KEY=...

PINECONE_API_KEY=...
PINECONE_INDEX_NAME=ml-conference
PINECONE_ENV=us-east-1

LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_PROMPT_NAME=rag-answer
LANGFUSE_PROMPT_LABEL=production
LANGFUSE_ENVIRONMENT=default

EVAL_FAITHFULNESS_THRESHOLD=0.3
EVAL_RELEVANCE_THRESHOLD=0.3
RETRIEVAL_K=10
RAG_DB_PATH=./rag_app.db
```

### Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open:
- UI: `http://localhost:8000/`
- Docs: `http://localhost:8000/docs`

### Docker

```bash
docker build -t rag-app:latest .
docker run --rm -p 8000:8000 --env-file .env -e RAG_DB_PATH=/app/storage/rag_app.db -v "$(pwd)/storage:/app/storage" rag-app:latest
```

### Docker Compose

```bash
mkdir -p storage
docker compose up --build
```

### API Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | UI page |
| `POST` | `/ask` | Ask a question — retrieves context, generates answer, evaluates, persists |
| `GET` | `/talks` | List all ML Prague 2026 talks |
| `POST` | `/feedback` | Submit rating by `qa_id` |
| `GET` | `/history` | In-process history |
| `POST` | `/history/update` | Override answer/rating |
| `GET` | `/history/langfuse` | Merged Langfuse traces and feedback |
| `GET` | `/sessions` | List SQLite sessions |
| `GET` | `/sessions/{session_id}/messages` | List messages in a session |

### Testing

```bash
pytest tests/ -v        # RAG service tests
pytest mcp_server/tests/ -v  # MCP server tests (no credentials needed)
pytest -v               # all tests
```

## RAG Quality Agent

Offline evaluation agent that runs against a live API instance:

```bash
python agents/rag_quality_agent.py --base-url http://127.0.0.1:8000
```

Options:

```bash
python agents/rag_quality_agent.py --limit 3
python agents/rag_quality_agent.py --update-baseline
python agents/rag_quality_agent.py --allow-regression
```

Checks: answer rate, reject rate, avg faithfulness/relevance, keyword recall, latency p50/p95.

## Storage

### SQLite

Default: `rag_app.db` (override with `RAG_DB_PATH`)

```bash
sqlite3 rag_app.db
SELECT session_id, title, updated_at FROM sessions ORDER BY updated_at DESC;
SELECT qa_id, question, rating, created_at FROM messages ORDER BY created_at DESC LIMIT 20;
```

### Langfuse

Traces include `rag-query`, `prompt`, `generation`, `decision`, `user-feedback`.

View via `GET /history/langfuse?limit=100` or in the Langfuse UI.

## Troubleshooting

- **`503 Service starting up`** — verify `.env` keys and network access to Pinecone/Mistral
- **MCP server `No module named 'mcp_server'`** — ensure `PYTHONPATH` is set to the repo root in the Claude Desktop config
- **Pinecone dimension mismatch** — index must be dimension 384 (all-MiniLM-L6-v2); delete and recreate if needed
- **`/history/langfuse` errors** — verify Langfuse keys; `limit` must be ≤ 100
- **Feedback `400`** — rating must be 1–5; when `qa_id` is omitted, both `question` and `answer` are required
