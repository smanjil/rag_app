from __future__ import annotations

from typing import Any

from rag_service.logic import evaluate_answer


def _contains_keyword(text: str, keyword: str) -> bool:
    return keyword.strip().lower() in text.lower()


def score_sample(sample: dict[str, Any], response: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    question = sample.get("question", "")
    answer = response.get("answer", "") or ""
    context = response.get("context", "") or ""
    similarity = response.get("similarity", {}) or {}

    lexical = evaluate_answer(question, context, answer)

    expected_keywords = sample.get("expected_keywords", []) or []
    if expected_keywords:
        hits = sum(1 for kw in expected_keywords if _contains_keyword(answer, kw))
        keyword_recall = hits / len(expected_keywords)
    else:
        keyword_recall = 1.0

    best_similarity = similarity.get("best_score")
    if best_similarity is None and isinstance(similarity.get("scores"), list) and similarity["scores"]:
        best_similarity = similarity["scores"][0]

    rejected = "not confident enough" in answer.lower()
    generation_model = response.get("generation_model")
    evaluation_model = response.get("evaluation_model")

    return {
        "id": sample.get("id"),
        "question": question,
        "latency_ms": round(latency_ms, 2),
        "answer": answer,
        "trace_id": response.get("trace_id"),
        "session_id": response.get("session_id"),
        "qa_id": response.get("qa_id"),
        "answered": bool(answer.strip()),
        "rejected": rejected,
        "faithfulness": lexical.get("faithfulness", 0.0),
        "relevance": lexical.get("relevance", 0.0),
        "keyword_recall": round(keyword_recall, 3),
        "best_similarity": best_similarity,
        "generation_model": generation_model,
        "evaluation_model": evaluation_model,
        "model_present": bool(generation_model) and bool(evaluation_model),
    }


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scores)
    if total == 0:
        return {
            "total": 0,
            "answer_rate": 0.0,
            "reject_rate": 0.0,
            "avg_faithfulness": 0.0,
            "avg_relevance": 0.0,
            "avg_keyword_recall": 0.0,
            "avg_best_similarity": None,
            "model_present_rate": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
        }

    def _avg(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    answered = [1.0 if s["answered"] else 0.0 for s in scores]
    rejected = [1.0 if s["rejected"] else 0.0 for s in scores]
    faithfulness = [float(s.get("faithfulness", 0.0) or 0.0) for s in scores]
    relevance = [float(s.get("relevance", 0.0) or 0.0) for s in scores]
    keyword_recall = [float(s.get("keyword_recall", 0.0) or 0.0) for s in scores]
    model_present = [1.0 if s.get("model_present") else 0.0 for s in scores]

    best_similarity_vals = [
        float(s["best_similarity"])
        for s in scores
        if isinstance(s.get("best_similarity"), (int, float))
    ]

    lat = sorted(float(s.get("latency_ms", 0.0) or 0.0) for s in scores)

    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        idx = int(round((len(values) - 1) * p))
        return round(values[idx], 2)

    return {
        "total": total,
        "answer_rate": _avg(answered),
        "reject_rate": _avg(rejected),
        "avg_faithfulness": _avg(faithfulness),
        "avg_relevance": _avg(relevance),
        "avg_keyword_recall": _avg(keyword_recall),
        "avg_best_similarity": _avg(best_similarity_vals) if best_similarity_vals else None,
        "model_present_rate": _avg(model_present),
        "latency_p50_ms": _percentile(lat, 0.50),
        "latency_p95_ms": _percentile(lat, 0.95),
    }
