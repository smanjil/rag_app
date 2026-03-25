import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .config import (
    DB_PATH,
    LANGFUSE_ENVIRONMENT,
    LANGFUSE_HOST,
    LOCAL_PROMPT_VERSION,
    RETRIEVAL_K,
    THRESHOLD_FAITHFULNESS,
    THRESHOLD_RELEVANCE,
    UI_HTML_PATH,
    env_value,
)
from .database import as_dict, db_conn, ensure_session, init_db, store_message, update_message_rating
from .logic import evaluate_answer, fetch_langfuse_traces, llm_evaluate, resolve_prompt, should_reject
from .models import FeedbackRequest, HistoryUpdateRequest, QueryRequest


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")


@dataclass
class RuntimeState:
    client: Optional[object] = None
    retriever: Optional[object] = None
    vectorstore: Optional[object] = None
    llm: Optional[object] = None
    qa_history: list[dict] = field(default_factory=list)
    history_overrides: dict[str, dict] = field(default_factory=dict)


state = RuntimeState()


def initialize_clients() -> None:
    if (
        state.client is not None
        and state.retriever is not None
        and state.llm is not None
        and state.vectorstore is not None
    ):
        return

    # Imported lazily to keep module import lightweight for tests and OpenAPI export.
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_mistralai.chat_models import ChatMistralAI
    from langchain_pinecone import PineconeVectorStore
    from langfuse import Langfuse
    from pinecone import Pinecone

    logger.info("Initializing clients")
    state.client = Langfuse(
        secret_key=env_value("LANGFUSE_SECRET_KEY"),
        public_key=env_value("LANGFUSE_PUBLIC_KEY"),
        host=LANGFUSE_HOST,
        environment=LANGFUSE_ENVIRONMENT,
    )

    logger.info("Initializing embeddings (may download model on first run)")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    logger.info("Initializing Pinecone retriever")
    pc = Pinecone(api_key=env_value("PINECONE_API_KEY"))
    index_name = env_value("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is required")

    index = pc.Index(index_name)
    state.vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    state.retriever = state.vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    logger.info("Initializing Mistral chat model")
    state.llm = ChatMistralAI(
        model="mistral-small",
        temperature=0,
        api_key=env_value("MISTRAL_API_KEY"),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting up: initializing services")
    init_db(DB_PATH)
    try:
        initialize_clients()
    except Exception:
        logger.exception("Startup dependency initialization failed; service will retry lazily")

    logger.info("Startup complete")
    try:
        yield
    finally:
        try:
            if state.client is not None and hasattr(state.client, "flush"):
                state.client.flush()
        except Exception:
            logger.exception("Langfuse flush failed on shutdown")


app = FastAPI(
    title="RAG App API",
    version="1.0.0",
    description="FastAPI RAG service with Pinecone retrieval, Mistral generation, Langfuse tracing, and SQLite persistence.",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    try:
        return UI_HTML_PATH.read_text(encoding="utf-8")
    except Exception:
        logger.exception("Failed to read UI HTML file: %s", UI_HTML_PATH)
        raise HTTPException(status_code=500, detail="UI file missing")


@app.post("/ask")
def ask(req: QueryRequest):
    init_db(DB_PATH)
    if state.retriever is None or state.llm is None or state.vectorstore is None:
        try:
            initialize_clients()
        except Exception as exc:
            logger.exception("Lazy initialization failed")
            raise HTTPException(
                status_code=503,
                detail=f"Service starting up; initialization failed: {exc}",
            )

    qa_id = uuid.uuid4().hex
    session_id = req.session_id or uuid.uuid4().hex
    ensure_session(DB_PATH, session_id, title=req.question[:80])

    docs = []
    raw_scores = []
    if state.vectorstore is not None and hasattr(state.vectorstore, "similarity_search_with_score"):
        docs_with_scores = state.vectorstore.similarity_search_with_score(req.question, k=RETRIEVAL_K)
        for doc, score in docs_with_scores:
            docs.append(doc)
            raw_scores.append(score)
    else:
        docs = state.retriever.invoke(req.question)
        raw_scores = [None] * len(docs)

    chunks = []
    for idx, doc in enumerate(docs):
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        safe_meta = {str(k): str(v) for k, v in meta.items()}
        score = raw_scores[idx] if idx < len(raw_scores) else None
        chunks.append(
            {
                "rank": idx + 1,
                "content": doc.page_content,
                "metadata": safe_meta,
                "similarity_score": score,
            }
        )

    context = "\n\n".join([doc.page_content for doc in docs])
    valid_scores = [float(s) for s in raw_scores if isinstance(s, (int, float))]
    similarity = {
        "k": RETRIEVAL_K,
        "scores": valid_scores,
        "best_score": max(valid_scores) if valid_scores else None,
        "avg_score": (sum(valid_scores) / len(valid_scores)) if valid_scores else None,
    }

    prompt, prompt_meta = resolve_prompt(state.client, context, req.question)

    if state.client is not None and hasattr(state.client, "start_as_current_observation"):
        with state.client.start_as_current_observation(
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

    response = state.llm.invoke(prompt)
    answer = response.content if isinstance(response.content, str) else str(response.content)
    original_answer = answer

    try:
        eval_result = llm_evaluate(state.llm, req.question, context, answer)
    except Exception:
        logger.exception("Evaluation failed; continuing without rejection")
        eval_result = {
            "faithfulness": None,
            "relevance": None,
            "verdict": "unknown",
            "error": "evaluation_failed",
        }

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
        "reason": "threshold_breach" if reject else "accepted_or_eval_unavailable",
    }

    if reject:
        answer = "I'm not confident enough to answer based on the available information."

    trace_id = None
    if state.client is not None and hasattr(state.client, "start_as_current_observation"):
        with state.client.start_as_current_observation(
            name="rag-query",
            as_type="chain",
            input={"session_id": session_id, "qa_id": qa_id, "question": req.question},
        ):
            trace_id = state.client.get_current_trace_id() if hasattr(state.client, "get_current_trace_id") else None
            try:
                if hasattr(state.client, "update_current_span"):
                    state.client.update_current_span(
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
                if hasattr(state.client, "set_current_trace_io"):
                    state.client.set_current_trace_io(
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
                if hasattr(state.client, "start_as_current_observation"):
                    with state.client.start_as_current_observation(
                        name="generation",
                        as_type="generation",
                        output={"original_answer": original_answer},
                    ):
                        pass
                    with state.client.start_as_current_observation(
                        name="decision",
                        as_type="span",
                        output=decision_payload,
                    ):
                        pass
            except Exception:
                logger.exception("Langfuse trace update failed")

    try:
        if state.client is not None and hasattr(state.client, "flush"):
            state.client.flush()
    except Exception:
        logger.exception("Langfuse flush failed")

    result = {
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
        db_path=DB_PATH,
        session_id=session_id,
        qa_id=qa_id,
        question=req.question,
        answer=answer,
        context=context,
        evaluation=eval_result,
        similarity=similarity,
        trace_id=trace_id,
    )
    state.qa_history.append(
        {
            "qa_id": qa_id,
            "asked_at": datetime.now(timezone.utc).isoformat(),
            "question": req.question,
            "answer": answer,
            "rating": None,
            "trace_id": trace_id,
        }
    )
    logger.debug("Returning /ask response for qa_id=%s", qa_id)
    return result


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    init_db(DB_PATH)
    if req.rating < 1 or req.rating > 5:
        raise HTTPException(status_code=400, detail="rating must be between 1 and 5")

    try:
        if req.qa_id:
            patch = state.history_overrides.get(req.qa_id, {})
            patch["rating"] = req.rating
            patch["updated_at"] = datetime.now(timezone.utc).isoformat()
            state.history_overrides[req.qa_id] = patch
            update_message_rating(DB_PATH, req.qa_id, req.rating)

        if req.qa_id:
            for item in reversed(state.qa_history):
                if item.get("qa_id") == req.qa_id:
                    item["rating"] = req.rating
                    item["rated_at"] = datetime.now(timezone.utc).isoformat()
                    break
        else:
            if not req.question or not req.answer:
                raise HTTPException(
                    status_code=400,
                    detail="question and answer are required when qa_id is not provided",
                )
            with db_conn(DB_PATH) as conn:
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
                update_message_rating(DB_PATH, row["qa_id"], req.rating)
            for item in reversed(state.qa_history):
                if item.get("question") == req.question and item.get("answer") == req.answer:
                    item["rating"] = req.rating
                    item["rated_at"] = datetime.now(timezone.utc).isoformat()
                    break

        if state.client is not None and hasattr(state.client, "start_as_current_observation"):
            with state.client.start_as_current_observation(
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
            if hasattr(state.client, "flush"):
                state.client.flush()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to record feedback in Langfuse")
        raise HTTPException(status_code=500, detail="failed to record feedback")

    return {"status": "feedback recorded"}


@app.get("/history")
def history():
    items = list(reversed(state.qa_history))
    for item in items:
        key = item.get("qa_id") or item.get("trace_id")
        if key and key in state.history_overrides:
            item.update(state.history_overrides[key])
    return {"items": items}


@app.post("/history/update")
def history_update(req: HistoryUpdateRequest):
    init_db(DB_PATH)
    key = req.qa_id or req.trace_id
    if not key:
        raise HTTPException(status_code=400, detail="qa_id or trace_id is required")
    if req.answer is None and req.rating is None:
        raise HTTPException(status_code=400, detail="answer or rating is required")
    if req.rating is not None and (req.rating < 1 or req.rating > 5):
        raise HTTPException(status_code=400, detail="rating must be between 1 and 5")

    patch = state.history_overrides.get(key, {})
    if req.answer is not None:
        patch["answer"] = req.answer
    if req.rating is not None:
        patch["rating"] = req.rating
    patch["updated_at"] = datetime.now(timezone.utc).isoformat()
    state.history_overrides[key] = patch

    with db_conn(DB_PATH) as conn:
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

    by_key = {}
    for trace in traces:
        if trace.get("name") != "rag-query":
            continue
        input_obj = as_dict(trace.get("input"))
        output_obj = as_dict(trace.get("output"))
        qa_id = input_obj.get("qa_id") or output_obj.get("qa_id")
        key = qa_id or f"trace:{trace.get('id')}"
        by_key[key] = {
            "qa_id": qa_id,
            "asked_at": trace.get("timestamp") or trace.get("createdAt"),
            "question": input_obj.get("question"),
            "answer": output_obj.get("answer"),
            "rating": None,
            "evaluation": output_obj.get("evaluation"),
            "similarity": output_obj.get("similarity"),
            "trace_id": trace.get("id"),
        }

    for trace in traces:
        if trace.get("name") != "user-feedback":
            continue
        input_obj = as_dict(trace.get("input"))
        output_obj = as_dict(trace.get("output"))
        qa_id = input_obj.get("qa_id")
        if qa_id and qa_id in by_key:
            by_key[qa_id]["rating"] = output_obj.get("rating", by_key[qa_id]["rating"])

    for item_key, item in by_key.items():
        override_key = item.get("qa_id") or item.get("trace_id") or item_key
        if override_key and override_key in state.history_overrides:
            item.update(state.history_overrides[override_key])

    items = sorted(by_key.values(), key=lambda x: x.get("asked_at") or "", reverse=True)
    return {"items": items, "error": err}


@app.get("/sessions")
def sessions():
    init_db(DB_PATH)
    with db_conn(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT s.session_id, s.title, s.created_at, s.updated_at, COUNT(m.id) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            GROUP BY s.session_id, s.title, s.created_at, s.updated_at
            ORDER BY s.updated_at DESC
            """
        ).fetchall()
    return {"items": [dict(row) for row in rows]}


@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str):
    init_db(DB_PATH)
    with db_conn(DB_PATH) as conn:
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
    for row in rows:
        item = dict(row)
        item["evaluation"] = as_dict(item.pop("evaluation_json"))
        item["similarity"] = as_dict(item.pop("similarity_json"))
        items.append(item)

    return {"session_id": session_id, "items": items}


__all__ = [
    "app",
    "ask",
    "feedback",
    "history",
    "history_update",
    "history_langfuse",
    "initialize_clients",
    "QueryRequest",
    "evaluate_answer",
    "state",
]
