# RAG App

Lightweight FastAPI RAG app with:
- Pinecone retrieval
- Mistral generation
- Langfuse tracing + prompt management
- Built-in web UI (`/`)

## Project Structure

- `app.py`: FastAPI app, RAG pipeline, tracing, feedback, history endpoints
- `ingest.py`: loads `data/*.txt`, chunks, embeds, uploads to Pinecone
- `ui/index.html`: frontend UI
- `data/`: source text files for indexing
- `.env`: local config and API keys

## Requirements

- Python 3.11 recommended
- Pinecone account + API key
- Mistral API key
- Langfuse project keys (optional but used by this app)

Install dependencies:

```bash
pip install -r requirements.txt
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

EVAL_FAITHFULNESS_THRESHOLD=0.7
EVAL_RELEVANCE_THRESHOLD=0.7
RETRIEVAL_K=3
```

## 1) Ingest Data into Pinecone

Put `.txt` files under `data/`, then run:

```bash
python ingest.py
```

## 2) Run the App

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open:
- UI: `http://localhost:8000/`
- Docs: `http://localhost:8000/docs`

## Prompt Management (Langfuse)

Create a text prompt in Langfuse Prompt Management:
- Name: `rag-answer` (or value of `LANGFUSE_PROMPT_NAME`)
- Label: `production` (or value of `LANGFUSE_PROMPT_LABEL`)
- Template:

```txt
Answer ONLY using the context below. If unsure, say you don't know.

Context:
{{context}}

Question:
{{question}}
```

The app fetches this prompt dynamically and falls back to local template if unavailable.

## API Endpoints

- `POST /ask`
  - Input: `{ "question": "..." }`
  - Returns: answer, context, chunks (with `similarity_score`), evaluation, similarity summary, trace_id

- `POST /feedback`
  - Input: `{ "qa_id": "...", "question": "...", "answer": "...", "rating": 1..5 }`

- `GET /history/langfuse?limit=100`
  - Returns Langfuse-derived history rows (rag-query + feedback merge)

- `POST /history/update`
  - Input: `{ "qa_id"|"trace_id", "answer"?, "rating"? }`
  - Applies local overrides for UI history edits

## UI Notes

- Ask tab: question, answer, eval, context, chunks, similarity
- History tab: records from Langfuse, rating controls, edit action

## Troubleshooting

- `503 Service starting up`: check `.env` keys and network access
- `history/langfuse` warning/error:
  - verify Langfuse keys/URL
  - `limit` must be <= 100
- Missing traces in Langfuse:
  - confirm correct project keys
  - check `LANGFUSE_BASE_URL`
  - ensure requests hit `/ask` and app can flush events

## Quick Test

```bash
curl -sS -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"what is s3"}' | jq
```
