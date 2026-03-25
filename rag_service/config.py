import os
from pathlib import Path
from typing import Optional


def env_value(key: str) -> Optional[str]:
    val = os.getenv(key)
    if val is None:
        return None
    val = val.strip()
    if len(val) >= 2 and ((val[0] == val[-1]) and val[0] in {'"', "'"}):
        val = val[1:-1]
    return val or None


BASE_DIR = Path(__file__).resolve().parent.parent
UI_HTML_PATH = BASE_DIR / "ui" / "index.html"
DB_PATH = Path(os.getenv("RAG_DB_PATH", str(BASE_DIR / "rag_app.db")))

THRESHOLD_FAITHFULNESS = float(os.getenv("EVAL_FAITHFULNESS_THRESHOLD", "0.7"))
THRESHOLD_RELEVANCE = float(os.getenv("EVAL_RELEVANCE_THRESHOLD", "0.7"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))

PROMPT_NAME = os.getenv("LANGFUSE_PROMPT_NAME", "rag-answer")
PROMPT_LABEL = os.getenv("LANGFUSE_PROMPT_LABEL", "production")
LOCAL_PROMPT_VERSION = "local-v1.0"

LANGFUSE_HOST = env_value("LANGFUSE_BASE_URL") or env_value("LANGFUSE_HOST") or "https://cloud.langfuse.com"
LANGFUSE_ENVIRONMENT = env_value("LANGFUSE_ENVIRONMENT") or "default"
