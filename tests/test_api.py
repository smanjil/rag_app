from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import rag_service.main as service


class Resp:
    def __init__(self, content):
        self.content = content


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


class FakeVectorStore:
    def similarity_search_with_score(self, question, k=3):
        docs = [
            (FakeDoc("S3 stores objects in buckets.", {"source": "aws.txt"}), 0.91),
            (FakeDoc("Docker packages apps in containers.", {"source": "docker.txt"}), 0.73),
        ]
        return docs[:k]


class GoodLLM:
    def invoke(self, prompt):
        if "You are an evaluator for a RAG system." in prompt:
            return Resp('{"faithfulness": 0.95, "relevance": 0.88, "verdict": "good"}')
        return Resp("S3 stores objects in buckets.")


class LowEvalLLM:
    def invoke(self, prompt):
        if "You are an evaluator for a RAG system." in prompt:
            return Resp('{"faithfulness": 0.1, "relevance": 0.2, "verdict": "bad"}')
        return Resp("S3 stores objects in buckets.")


class EvalFailLLM:
    def invoke(self, prompt):
        if "You are an evaluator for a RAG system." in prompt:
            raise RuntimeError("simulated evaluator failure")
        return Resp("S3 stores objects in buckets.")


@pytest.fixture
def test_client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "DB_PATH", Path(tmp_path) / "test.db")
    monkeypatch.setattr(service, "initialize_clients", lambda: None)

    service.state.client = None
    service.state.retriever = None
    service.state.vectorstore = FakeVectorStore()
    service.state.llm = GoodLLM()
    service.state.qa_history.clear()
    service.state.history_overrides.clear()

    with TestClient(service.app) as client:
        yield client


def test_ask_and_sessions_flow(test_client):
    response = test_client.post("/ask", json={"question": "what is s3"})
    assert response.status_code == 200
    payload = response.json()

    assert payload["answer"] == "S3 stores objects in buckets."
    assert payload["qa_id"]
    assert payload["session_id"]
    assert payload["similarity"]["best_score"] == pytest.approx(0.91)

    sessions = test_client.get("/sessions")
    assert sessions.status_code == 200
    assert len(sessions.json()["items"]) >= 1

    messages = test_client.get(f"/sessions/{payload['session_id']}/messages")
    assert messages.status_code == 200
    assert len(messages.json()["items"]) == 1


def test_feedback_by_qa_id_only(test_client):
    ask = test_client.post("/ask", json={"question": "what is s3"}).json()

    feedback = test_client.post("/feedback", json={"qa_id": ask["qa_id"], "rating": 5})
    assert feedback.status_code == 200

    messages = test_client.get(f"/sessions/{ask['session_id']}/messages").json()["items"]
    assert messages[0]["rating"] == 5


def test_feedback_validation_errors(test_client):
    invalid_rating = test_client.post("/feedback", json={"qa_id": "x", "rating": 7})
    assert invalid_rating.status_code == 400

    missing_fields = test_client.post("/feedback", json={"rating": 3})
    assert missing_fields.status_code == 400


def test_history_update_validation(test_client):
    no_key = test_client.post("/history/update", json={"answer": "x"})
    assert no_key.status_code == 400

    no_payload = test_client.post("/history/update", json={"qa_id": "abc"})
    assert no_payload.status_code == 400


def test_low_eval_rejects_answer(test_client):
    service.state.llm = LowEvalLLM()
    response = test_client.post("/ask", json={"question": "what is s3"})
    assert response.status_code == 200
    assert response.json()["answer"] == "I'm not confident enough to answer based on the available information."


def test_eval_failure_does_not_break_request(test_client):
    service.state.llm = EvalFailLLM()
    response = test_client.post("/ask", json={"question": "what is s3"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["evaluation"]["error"] == "evaluation_failed"
    assert payload["answer"] == "S3 stores objects in buckets."


def test_history_langfuse_merge_and_override(test_client, monkeypatch):
    traces = [
        {
            "id": "trace-1",
            "name": "rag-query",
            "timestamp": "2026-03-25T10:00:00Z",
            "input": {"qa_id": "qa-1", "question": "what is s3"},
            "output": {"qa_id": "qa-1", "answer": "a1", "evaluation": {}, "similarity": {}},
        },
        {
            "id": "trace-2",
            "name": "user-feedback",
            "timestamp": "2026-03-25T10:01:00Z",
            "input": {"qa_id": "qa-1"},
            "output": {"rating": 4},
        },
    ]

    service.state.history_overrides["qa-1"] = {"answer": "edited"}
    monkeypatch.setattr(service, "fetch_langfuse_traces", lambda limit=100: (traces, None))

    response = test_client.get("/history/langfuse?limit=10")
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["rating"] == 4
    assert items[0]["answer"] == "edited"
