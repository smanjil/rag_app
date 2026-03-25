import base64
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Optional

from .config import (
    LANGFUSE_HOST,
    LOCAL_PROMPT_VERSION,
    PROMPT_LABEL,
    PROMPT_NAME,
    THRESHOLD_FAITHFULNESS,
    THRESHOLD_RELEVANCE,
    env_value,
)

logger = logging.getLogger("rag_app")


def safe_eval_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 0.0 or score > 1.0:
        return None
    return score


def should_reject(eval_result: dict) -> bool:
    faithfulness = safe_eval_score(eval_result.get("faithfulness"))
    relevance = safe_eval_score(eval_result.get("relevance"))

    if faithfulness is None or relevance is None:
        return False

    return (
        faithfulness < THRESHOLD_FAITHFULNESS
        or relevance < THRESHOLD_RELEVANCE
    )


def llm_evaluate(llm, question: str, context: str, answer: str) -> dict:
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


def evaluate_answer(question: str, context: str, answer: str) -> dict:
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


def build_prompt(context: str, question: str) -> str:
    return f"""
[VERSION: {LOCAL_PROMPT_VERSION}]

Answer ONLY using the context below. If unsure, say you don't know.

Context:
{context}

Question:
{question}
"""


def resolve_prompt(client, context: str, question: str) -> tuple[str, dict]:
    fallback_prompt = build_prompt(context, question)

    if client is None or not hasattr(client, "get_prompt"):
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

        managed_prompt = client.get_prompt(**kwargs)
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


def fetch_langfuse_traces(limit: int = 100) -> tuple[list[dict], Optional[str]]:
    public_key = env_value("LANGFUSE_PUBLIC_KEY")
    secret_key = env_value("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return [], "Langfuse keys are not configured"

    safe_limit = max(1, min(int(limit), 100))
    url = f"{LANGFUSE_HOST.rstrip('/')}/api/public/traces?page=1&limit={safe_limit}"
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
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        return [], f"Langfuse HTTP error {exc.code}: {body[:300]}"
    except Exception as exc:
        return [], f"Langfuse request failed: {exc}"

    data = payload.get("data", [])
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"], None
    return [], "Unexpected Langfuse traces response format"
