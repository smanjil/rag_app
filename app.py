import os
import logging
import json
import re
import uuid
import base64
import sqlite3
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai.chat_models import ChatMistralAI

from langfuse import Langfuse
from pinecone import Pinecone


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

# ---- Thresholds ----
THRESHOLD_FAITHFULNESS = float(os.getenv("EVAL_FAITHFULNESS_THRESHOLD", 0.7))
THRESHOLD_RELEVANCE = float(os.getenv("EVAL_RELEVANCE_THRESHOLD", 0.7))


# ✅ ---- ADD HERE ----
def should_reject(eval_result):
    try:
        return (
            eval_result["faithfulness"] < THRESHOLD_FAITHFULNESS
            or eval_result["relevance"] < THRESHOLD_RELEVANCE
        )
    except:
        return True  # fail-safe


def _env(key: str) -> Optional[str]:
    val = os.getenv(key)
    if val is None:
        return None
    val = val.strip()
    if len(val) >= 2 and ((val[0] == val[-1]) and val[0] in {'"', "'"}):
        val = val[1:-1]
    return val or None


def initialize_clients() -> None:
    global _client, retriever, llm, vectorstore
    if (
        _client is not None
        and retriever is not None
        and llm is not None
        and vectorstore is not None
    ):
        return

    logger.info("Initializing clients")
    host = _env("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
    environment = _env("LANGFUSE_ENVIRONMENT") or "default"
    _client = Langfuse(
        secret_key=_env("LANGFUSE_SECRET_KEY"),
        public_key=_env("LANGFUSE_PUBLIC_KEY"),
        host=host,
        environment=environment,
    )

    logger.info("Initializing embeddings (may download model on first run)")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    logger.info("Initializing Pinecone retriever")
    pc = Pinecone(api_key=_env("PINECONE_API_KEY"))
    index_name = _env("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is required")
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    logger.info("Initializing Mistral chat model")
    llm = ChatMistralAI(
        model="mistral-small",
        temperature=0,
        api_key=_env("MISTRAL_API_KEY"),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _client, retriever, llm

    logger.info("Starting up: initializing clients")
    init_db()
    initialize_clients()

    logger.info("Startup complete")
    try:
        yield
    finally:
        try:
            if _client is not None and hasattr(_client, "flush"):
                _client.flush()
        except Exception:
            logger.exception("Langfuse flush failed on shutdown")


app = FastAPI(lifespan=lifespan)
_client: Optional[Langfuse] = None
retriever = None
vectorstore = None
llm: Optional[ChatMistralAI] = None
UI_HTML_PATH = Path(__file__).parent / "ui" / "index.html"
qa_history: list[dict] = []
history_overrides: dict[str, dict] = {}
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
DB_PATH = Path(os.getenv("RAG_DB_PATH", str(Path(__file__).parent / "rag_app.db")))

# Langfuse Prompt Management settings
PROMPT_NAME = os.getenv("LANGFUSE_PROMPT_NAME", "rag-answer")
PROMPT_LABEL = os.getenv("LANGFUSE_PROMPT_LABEL", "production")
LOCAL_PROMPT_VERSION = "local-v1.0"


def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                qa_id TEXT UNIQUE NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                context TEXT,
                evaluation_json TEXT,
                similarity_json TEXT,
                rating INTEGER,
                trace_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at)"
        )


def ensure_session(session_id: str, title: Optional[str] = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO sessions(session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title or "New chat", now, now),
            )
        else:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )


def store_message(
    *,
    session_id: str,
    qa_id: str,
    question: str,
    answer: str,
    context: str,
    evaluation: dict,
    similarity: dict,
    trace_id: Optional[str],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _db_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages(
                session_id, qa_id, question, answer, context,
                evaluation_json, similarity_json, trace_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(qa_id) DO UPDATE SET
                answer=excluded.answer,
                context=excluded.context,
                evaluation_json=excluded.evaluation_json,
                similarity_json=excluded.similarity_json,
                trace_id=excluded.trace_id,
                updated_at=excluded.updated_at
            """,
            (
                session_id,
                qa_id,
                question,
                answer,
                context,
                json.dumps(evaluation or {}),
                json.dumps(similarity or {}),
                trace_id,
                now,
                now,
            ),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )


def update_message_rating(qa_id: str, rating: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _db_conn() as conn:
        conn.execute(
            "UPDATE messages SET rating = ?, updated_at = ? WHERE qa_id = ?",
            (rating, now, qa_id),
        )


def llm_evaluate(question, context, answer):
    eval_prompt = f"""
You are an evaluator for a RAG system.

Given:
Question: {question}
Context: {context}
Answer: {answer}

Evaluate:
1. Faithfulness (0-1): Is answer grounded in context?
2. Relevance (0-1): Does it answer the question?

Return ONLY JSON:
{{
  "faithfulness": float,
  "relevance": float,
  "verdict": "good" | "bad"
}}
"""
    result = llm.invoke(eval_prompt)
    raw = (result.content or "").strip()

    # Accept plain JSON or JSON wrapped in markdown code fences.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {
        "faithfulness": 0.0,
        "relevance": 0.0,
        "verdict": "bad",
        "raw_eval_text": raw,
    }


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    qa_id: Optional[str] = None
    question: str
    answer: str
    rating: int  # 1 (bad) to 5 (good)


class HistoryUpdateRequest(BaseModel):
    qa_id: Optional[str] = None
    trace_id: Optional[str] = None
    answer: Optional[str] = None
    rating: Optional[int] = None


def evaluate_answer(question, context, answer):
    q_tokens = {w.lower() for w in question.split() if w.strip()}
    a_tokens = {w.lower() for w in answer.split() if w.strip()}
    c_tokens = {w.lower() for w in context.split() if w.strip()}

    relevance = 0.0
    if q_tokens:
        relevance = len(q_tokens & a_tokens) / len(q_tokens)

    faithfulness = 0.0
    if a_tokens:
        faithfulness = len(a_tokens & c_tokens) / len(a_tokens)

    verdict = "good" if (faithfulness >= 0.5 and relevance >= 0.5) else "bad"
    return {
        "faithfulness": round(faithfulness, 3),
        "relevance": round(relevance, 3),
        "verdict": verdict,
    }


def build_prompt(context, question):
    return f"""
[VERSION: {LOCAL_PROMPT_VERSION}]

Answer ONLY using the context below. If unsure, say you don't know.

Context:
{context}

Question:
{question}
"""


def resolve_prompt(context: str, question: str) -> tuple[str, dict]:
    fallback_prompt = build_prompt(context, question)

    if _client is None or not hasattr(_client, "get_prompt"):
        return fallback_prompt, {
            "source": "local-fallback",
            "name": PROMPT_NAME,
            "label": PROMPT_LABEL,
            "version": LOCAL_PROMPT_VERSION,
        }

    try:
        kwargs = {
            "name": PROMPT_NAME,
            "type": "text",
            "fallback": (
                "Answer ONLY using the context below. If unsure, say you don't know.\n\n"
                "Context:\n{{context}}\n\nQuestion:\n{{question}}\n"
            ),
        }
        if PROMPT_LABEL:
            kwargs["label"] = PROMPT_LABEL

        managed_prompt = _client.get_prompt(**kwargs)
        compiled = managed_prompt.compile(context=context, question=question)
        return compiled, {
            "source": "langfuse-managed",
            "name": getattr(managed_prompt, "name", PROMPT_NAME),
            "label": PROMPT_LABEL,
            "version": getattr(managed_prompt, "version", None),
        }
    except Exception:
        logger.exception("Failed to load managed prompt from Langfuse, using fallback")
        return fallback_prompt, {
            "source": "local-fallback",
            "name": PROMPT_NAME,
            "label": PROMPT_LABEL,
            "version": LOCAL_PROMPT_VERSION,
        }


def _as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def fetch_langfuse_traces(limit: int = 100) -> tuple[list[dict], Optional[str]]:
    host = (
        _env("LANGFUSE_BASE_URL")
        or _env("LANGFUSE_HOST")
        or "https://cloud.langfuse.com"
    ).rstrip("/")
    public_key = _env("LANGFUSE_PUBLIC_KEY")
    secret_key = _env("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return [], "Langfuse keys are not configured"

    safe_limit = max(1, min(int(limit), 100))
    url = f"{host}/api/public/traces?page=1&limit={safe_limit}"
    token = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("utf-8")
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Basic {token}",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        return [], f"Langfuse HTTP error {e.code}: {body[:300]}"
    except Exception as e:
        return [], f"Langfuse request failed: {e}"

    data = payload.get("data", [])
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"], None
    return [], "Unexpected Langfuse traces response format"


@app.get("/", response_class=HTMLResponse)
def ui():
    try:
        return UI_HTML_PATH.read_text(encoding="utf-8")
    except Exception:
        logger.exception("Failed to read UI HTML file: %s", UI_HTML_PATH)
        raise HTTPException(status_code=500, detail="UI file missing")


@app.post("/ask")
def ask(req: QueryRequest):
    init_db()
    if retriever is None or llm is None or vectorstore is None:
        try:
            initialize_clients()
        except Exception as e:
            logger.exception("Lazy initialization failed")
            raise HTTPException(
                status_code=503,
                detail=f"Service starting up; initialization failed: {e}",
            )

    # Langfuse SDK in this env does not support `langfuse.trace(...)`.
    # We create the `rag-query` trace later via `start_as_current_observation`.

    trace_id = None
    answer = ""
    eval_result = {}
    qa_id = uuid.uuid4().hex
    session_id = req.session_id or uuid.uuid4().hex
    ensure_session(session_id, title=req.question[:80])

    # ---- Retrieval ----
    docs = []
    raw_scores = []
    if vectorstore is not None and hasattr(vectorstore, "similarity_search_with_score"):
        docs_with_scores = vectorstore.similarity_search_with_score(
            req.question, k=RETRIEVAL_K
        )
        for doc, score in docs_with_scores:
            docs.append(doc)
            raw_scores.append(score)
    else:
        docs = retriever.invoke(req.question)
        raw_scores = [None] * len(docs)
    chunks = []
    for i, d in enumerate(docs):
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        safe_meta = {str(k): str(v) for k, v in meta.items()}
        score = raw_scores[i] if i < len(raw_scores) else None
        chunks.append(
            {
                "rank": i + 1,
                "content": d.page_content,
                "metadata": safe_meta,
                "similarity_score": score,
            }
        )
    context = "\n\n".join([d.page_content for d in docs])
    valid_scores = [float(s) for s in raw_scores if isinstance(s, (int, float))]
    similarity = {
        "k": RETRIEVAL_K,
        "scores": valid_scores,
        "best_score": max(valid_scores) if valid_scores else None,
        "avg_score": (sum(valid_scores) / len(valid_scores)) if valid_scores else None,
    }

    # ✅ Step 1 — Resolve prompt (Langfuse Prompt Management with fallback)
    prompt, prompt_meta = resolve_prompt(context, req.question)

    # ✅ Step 2 — Log to Langfuse (SDK-compatible)
    if _client is not None and hasattr(_client, "start_as_current_observation"):
        with _client.start_as_current_observation(
            name="prompt",
            as_type="span",
            input={
                "source": prompt_meta["source"],
                "name": prompt_meta["name"],
                "label": prompt_meta["label"],
                "version": prompt_meta["version"],
                "local_fallback_version": LOCAL_PROMPT_VERSION,
                "prompt": prompt,
                "question": req.question,
            },
        ):
            pass

    # ---- Generation ----
    response = llm.invoke(prompt)
    answer = response.content
    original_answer = answer

    # ✅ ---- EVALUATION GOES HERE ----
    eval_result = llm_evaluate(
        req.question,
        context,
        answer,
    )

    # ✅ ---- ADD HERE ----
    reject = should_reject(eval_result)
    decision_payload = {
        "rejected": reject,
        "scores": {
            "faithfulness": eval_result.get("faithfulness"),
            "relevance": eval_result.get("relevance"),
        },
        "thresholds": {
            "faithfulness": THRESHOLD_FAITHFULNESS,
            "relevance": THRESHOLD_RELEVANCE,
        },
    }

    if reject:
        answer = (
            "I'm not confident enough to answer based on the available information."
        )

    if _client is not None and hasattr(_client, "start_as_current_observation"):
        with _client.start_as_current_observation(
            name="rag-query",
            as_type="chain",
            input={"session_id": session_id, "qa_id": qa_id, "question": req.question},
        ):
            trace_id = _client.get_current_trace_id()
            logger.info("Langfuse trace started: %s", trace_id)
            try:
                if hasattr(_client, "update_current_span"):
                    _client.update_current_span(
                        output={
                            "answer": answer,
                            "evaluation": eval_result,
                            "decision": decision_payload,
                            "similarity": similarity,
                            "context_length": len(context),
                            "context": context,
                            "chunks": chunks,
                        }
                    )
                if hasattr(_client, "set_current_trace_io"):
                    _client.set_current_trace_io(
                        input={"session_id": session_id, "qa_id": qa_id, "question": req.question},
                        output={
                            "session_id": session_id,
                            "qa_id": qa_id,
                            "answer": answer,
                            "original_answer": original_answer,
                            "evaluation": eval_result,
                            "decision": decision_payload,
                            "similarity": similarity,
                            "context": context,
                            "chunks": chunks,
                        },
                    )
                if hasattr(_client, "start_as_current_observation"):
                    with _client.start_as_current_observation(
                        name="generation",
                        as_type="generation",
                        output={"original_answer": original_answer},
                    ):
                        pass
                if hasattr(_client, "start_as_current_observation"):
                    with _client.start_as_current_observation(
                        name="decision",
                        as_type="span",
                        output=decision_payload,
                    ):
                        pass

                logger.info("Langfuse trace updated: %s", trace_id)
            except Exception:
                logger.exception("Langfuse trace update failed")

    try:
        if _client is not None and hasattr(_client, "flush"):
            _client.flush()
    except Exception:
        logger.exception("Langfuse flush failed")

    res = {
        "session_id": session_id,
        "qa_id": qa_id,
        "answer": answer,
        "context": context,
        "chunks": chunks,
        "similarity": similarity,
        "evaluation": eval_result,
        "trace_id": trace_id,
    }
    store_message(
        session_id=session_id,
        qa_id=qa_id,
        question=req.question,
        answer=answer,
        context=context,
        evaluation=eval_result,
        similarity=similarity,
        trace_id=trace_id,
    )
    qa_history.append(
        {
            "qa_id": qa_id,
            "asked_at": datetime.now(timezone.utc).isoformat(),
            "question": req.question,
            "answer": answer,
            "rating": None,
            "trace_id": trace_id,
        }
    )
    print(res)

    return res


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    init_db()
    if req.rating < 1 or req.rating > 5:
        raise HTTPException(status_code=400, detail="rating must be between 1 and 5")

    try:
        if req.qa_id:
            patch = history_overrides.get(req.qa_id, {})
            patch["rating"] = req.rating
            patch["updated_at"] = datetime.now(timezone.utc).isoformat()
            history_overrides[req.qa_id] = patch
            update_message_rating(req.qa_id, req.rating)

        if req.qa_id:
            for item in reversed(qa_history):
                if item.get("qa_id") == req.qa_id:
                    item["rating"] = req.rating
                    item["rated_at"] = datetime.now(timezone.utc).isoformat()
                    break
        else:
            with _db_conn() as conn:
                row = conn.execute(
                    """
                    SELECT qa_id FROM messages
                    WHERE question = ? AND answer = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (req.question, req.answer),
                ).fetchone()
            if row:
                update_message_rating(row["qa_id"], req.rating)
            for item in reversed(qa_history):
                if item.get("question") == req.question and item.get("answer") == req.answer:
                    item["rating"] = req.rating
                    item["rated_at"] = datetime.now(timezone.utc).isoformat()
                    break

        if _client is not None and hasattr(_client, "start_as_current_observation"):
            with _client.start_as_current_observation(
                name="user-feedback",
                as_type="span",
                input={
                    "qa_id": req.qa_id,
                    "question": req.question,
                    "answer": req.answer,
                },
                output={"rating": req.rating},
            ):
                pass
            if hasattr(_client, "flush"):
                _client.flush()
    except Exception:
        logger.exception("Failed to record feedback in Langfuse")
        raise HTTPException(status_code=500, detail="failed to record feedback")

    return {"status": "feedback recorded"}


@app.get("/history")
def history():
    items = list(reversed(qa_history))
    for item in items:
        key = item.get("qa_id") or item.get("trace_id")
        if key and key in history_overrides:
            item.update(history_overrides[key])
    return {"items": items}


@app.post("/history/update")
def history_update(req: HistoryUpdateRequest):
    init_db()
    key = req.qa_id or req.trace_id
    if not key:
        raise HTTPException(status_code=400, detail="qa_id or trace_id is required")
    if req.answer is None and req.rating is None:
        raise HTTPException(status_code=400, detail="answer or rating is required")
    if req.rating is not None and (req.rating < 1 or req.rating > 5):
        raise HTTPException(status_code=400, detail="rating must be between 1 and 5")

    patch = history_overrides.get(key, {})
    if req.answer is not None:
        patch["answer"] = req.answer
    if req.rating is not None:
        patch["rating"] = req.rating
    patch["updated_at"] = datetime.now(timezone.utc).isoformat()
    history_overrides[key] = patch

    with _db_conn() as conn:
        if req.qa_id:
            if req.answer is not None:
                conn.execute(
                    "UPDATE messages SET answer = ?, updated_at = ? WHERE qa_id = ?",
                    (req.answer, patch["updated_at"], req.qa_id),
                )
            if req.rating is not None:
                conn.execute(
                    "UPDATE messages SET rating = ?, updated_at = ? WHERE qa_id = ?",
                    (req.rating, patch["updated_at"], req.qa_id),
                )
        elif req.trace_id:
            row = conn.execute(
                "SELECT qa_id FROM messages WHERE trace_id = ? ORDER BY created_at DESC LIMIT 1",
                (req.trace_id,),
            ).fetchone()
            if row:
                if req.answer is not None:
                    conn.execute(
                        "UPDATE messages SET answer = ?, updated_at = ? WHERE qa_id = ?",
                        (req.answer, patch["updated_at"], row["qa_id"]),
                    )
                if req.rating is not None:
                    conn.execute(
                        "UPDATE messages SET rating = ?, updated_at = ? WHERE qa_id = ?",
                        (req.rating, patch["updated_at"], row["qa_id"]),
                    )

    return {"status": "updated", "key": key, "patch": patch}


@app.get("/history/langfuse")
def history_langfuse(limit: int = 100):
    traces, err = fetch_langfuse_traces(limit=limit)

    # Create rows from rag-query only.
    by_key = {}
    for t in traces:
        if t.get("name") != "rag-query":
            continue
        input_obj = _as_dict(t.get("input"))
        output_obj = _as_dict(t.get("output"))
        qa_id = input_obj.get("qa_id") or output_obj.get("qa_id")
        key = qa_id or f"trace:{t.get('id')}"
        by_key[key] = {
            "qa_id": qa_id,
            "asked_at": t.get("timestamp") or t.get("createdAt"),
            "question": input_obj.get("question"),
            "answer": output_obj.get("answer"),
            "rating": None,
            "evaluation": output_obj.get("evaluation"),
            "similarity": output_obj.get("similarity"),
            "trace_id": t.get("id"),
        }

    # Attach feedback ratings to existing rag-query rows by qa_id only.
    for t in traces:
        if t.get("name") != "user-feedback":
            continue
        input_obj = _as_dict(t.get("input"))
        output_obj = _as_dict(t.get("output"))
        qa_id = input_obj.get("qa_id")
        if not qa_id:
            continue
        if qa_id in by_key:
            by_key[qa_id]["rating"] = output_obj.get("rating", by_key[qa_id]["rating"])

    for item_key, item in by_key.items():
        override_key = item.get("qa_id") or item.get("trace_id") or item_key
        if override_key and override_key in history_overrides:
            item.update(history_overrides[override_key])

    items = sorted(
        by_key.values(),
        key=lambda x: x.get("asked_at") or "",
        reverse=True,
    )
    return {"items": items, "error": err}


@app.get("/sessions")
def sessions():
    init_db()
    with _db_conn() as conn:
        rows = conn.execute(
            """
            SELECT s.session_id, s.title, s.created_at, s.updated_at, COUNT(m.id) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            GROUP BY s.session_id, s.title, s.created_at, s.updated_at
            ORDER BY s.updated_at DESC
            """
        ).fetchall()
    items = [dict(r) for r in rows]
    return {"items": items}


@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str):
    init_db()
    with _db_conn() as conn:
        rows = conn.execute(
            """
            SELECT qa_id, question, answer, context, evaluation_json, similarity_json,
                   rating, trace_id, created_at, updated_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        ).fetchall()
    items = []
    for r in rows:
        d = dict(r)
        d["evaluation"] = _as_dict(d.pop("evaluation_json"))
        d["similarity"] = _as_dict(d.pop("similarity_json"))
        items.append(d)
    return {"session_id": session_id, "items": items}
