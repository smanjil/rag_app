"""Compatibility entrypoint.

This module keeps `uvicorn app:app` working while the implementation lives in
`rag_service.main`.
"""

from rag_service.main import QueryRequest, app, ask, evaluate_answer, initialize_clients, state

__all__ = [
    "app",
    "QueryRequest",
    "ask",
    "evaluate_answer",
    "initialize_clients",
    "state",
]
