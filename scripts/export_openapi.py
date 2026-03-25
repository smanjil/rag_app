from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_service.main import app


if __name__ == "__main__":
    out_dir = ROOT / "openapi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "openapi.json"
    out_file.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
    print(f"wrote {out_file}")
