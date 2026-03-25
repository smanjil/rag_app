# RAG App

Production-style FastAPI RAG service with:
- Pinecone retrieval
- Mistral generation
- Langfuse tracing + prompt management
- SQLite-backed sessions/messages
- Built-in UI (`/`)
- Checked-in OpenAPI spec (`openapi/openapi.json`)
- Pytest suite for API + logic paths
- Docker + docker-compose runtime

## Architecture (Restructured)

- `app.py`: compatibility entrypoint (`uvicorn app:app`)
- `rag_service/main.py`: app lifecycle + routes
- `rag_service/config.py`: env and runtime configuration
- `rag_service/models.py`: request schemas
- `rag_service/database.py`: SQLite persistence helpers
- `rag_service/logic.py`: prompt/evaluation/rejection/langfuse helpers
- `scripts/export_openapi.py`: exports OpenAPI artifact
- `openapi/openapi.json`: generated API contract
- `tests/`: pytest suite

## What Changed

Recent upgrade includes:
- Refactor from single-file app to layered package modules.
- Lazy heavy imports (`langchain`, `pinecone`, `langfuse`) for better testability and OpenAPI export reliability.
- Startup hardening: dependency init failures log and retry lazily in request path.
- OpenAPI spec export and checked-in API contract file.
- Container runtime improvements (non-root user, healthcheck, compose file, persistent SQLite volume).
- Comprehensive pytest coverage for route behavior and evaluation gating logic.

## Requirements

- Python 3.11+
- Pinecone API key + index
- Mistral API key
- Langfuse keys (recommended; `/history/langfuse` requires them)

Install runtime deps:

```bash
pip install -r requirements.txt
```

Install test deps:

```bash
pip install -r requirements-dev.txt
```

## Environment Variables

Create `.env` in repo root:

```env
MISTRAL_API_KEY=...

PINECONE_API_KEY=...
PINECONE_INDEX_NAME=test-index
PINECONE_ENV=us-east-1

LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_PROMPT_NAME=rag-answer
LANGFUSE_PROMPT_LABEL=production
LANGFUSE_ENVIRONMENT=default

EVAL_FAITHFULNESS_THRESHOLD=0.7
EVAL_RELEVANCE_THRESHOLD=0.7
RETRIEVAL_K=3
RAG_DB_PATH=./rag_app.db
```

## Run Locally

1. Create env + install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ingest `data/*.txt` into Pinecone:

```bash
python ingest.py
```

3. Run app:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

4. Open:
- UI: `http://localhost:8000/`
- Docs: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## OpenAPI Specification File

Export checked-in spec:

```bash
python scripts/export_openapi.py
```

Artifact path:
- `openapi/openapi.json`

This file is OpenAPI-compatible and can be used by gateways, SDK generators, or API tooling.

## Container Run

## Docker

```bash
docker build -t rag-app:latest .
docker run --rm -p 8000:8000 --env-file .env -e RAG_DB_PATH=/app/storage/rag_app.db -v "$(pwd)/storage:/app/storage" rag-app:latest
```

## Docker Compose

```bash
mkdir -p storage
docker compose up --build
```

App available at `http://localhost:8000`.

## API Routes

- `GET /` UI page
- `POST /ask` ask question, retrieve context, generate answer, evaluate, persist
- `POST /feedback` submit rating by `qa_id` or by `question`+`answer`
- `GET /history` in-memory history for running process
- `POST /history/update` override answer/rating by `qa_id` or `trace_id`
- `GET /history/langfuse` merged Langfuse traces and feedback
- `GET /sessions` list SQLite sessions
- `GET /sessions/{session_id}/messages` list SQLite messages

## Testing

Run full suite:

```bash
pytest
```

Current test coverage includes:
- `/ask` happy path and persistence
- `/ask` rejection path for low evaluation scores
- `/ask` evaluator-failure fallback path
- `/feedback` success and validation failures
- `/history/update` validation failures
- `/history/langfuse` merge + override behavior
- logic tests for `should_reject` edge cases

Notes:
- Tests mock vectorstore/LLM and avoid Pinecone/Mistral network calls.
- Tests use temporary SQLite DB paths.

## RAG Quality Agent

This repo includes an offline quality-evaluation agent:

- Runner: `agents/rag_quality_agent.py`
- Scoring: `agents/scoring.py`
- Dataset: `agents/eval_dataset.jsonl`
- Reports: `reports/quality_report_*.json` and `reports/quality_report_*.md`

Run it against a running API instance:

```bash
python agents/rag_quality_agent.py --base-url http://127.0.0.1:8000
```

Useful options:

```bash
python agents/rag_quality_agent.py --limit 3
python agents/rag_quality_agent.py --update-baseline
python agents/rag_quality_agent.py --allow-regression
```

What it checks:
- answer rate
- reject rate
- average faithfulness/relevance
- keyword recall from eval dataset expectations
- model metadata presence in responses
- latency p50/p95

It fails with non-zero exit code when quality gates fail (or when regression vs baseline is detected unless `--allow-regression` is set).

## Where Results Are Stored

## SQLite (local durable store)

Default file:
- `rag_app.db` (or `RAG_DB_PATH` override)

Tables:
- `sessions`
- `messages`

Inspect manually:

```bash
sqlite3 rag_app.db
.tables
SELECT session_id, title, updated_at FROM sessions ORDER BY updated_at DESC;
SELECT qa_id, session_id, question, rating, trace_id, created_at FROM messages ORDER BY created_at DESC LIMIT 20;
```

## Langfuse (remote)

Traces/observations include:
- `rag-query`
- `prompt`
- `generation`
- `decision`
- `user-feedback`

View in Langfuse UI or via:
- `GET /history/langfuse?limit=100`

## API/UI Response Surface

Immediate `/ask` response includes:
- `answer`
- `context`
- `chunks` (with `similarity_score`)
- `similarity` summary
- `evaluation`
- `session_id`, `qa_id`, `trace_id`

## Troubleshooting

- `503 Service starting up`:
  - verify `.env` keys
  - verify external service/network access

- `/history/langfuse` errors:
  - verify Langfuse keys and base URL
  - ensure `limit <= 100`

- Feedback `400` errors:
  - rating must be `1..5`
  - when `qa_id` omitted, both `question` and `answer` are required

- Container build large/slow:
  - ensure `.dockerignore` is respected
  - avoid copying local `.venv` and DB artifacts
