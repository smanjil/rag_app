from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    qa_id: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    rating: int


class HistoryUpdateRequest(BaseModel):
    qa_id: Optional[str] = None
    trace_id: Optional[str] = None
    answer: Optional[str] = None
    rating: Optional[int] = None
