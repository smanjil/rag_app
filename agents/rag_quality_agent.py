from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.scoring import aggregate, score_sample


DEFAULT_GATES = {
    "min_answer_rate": 0.90,
    "max_reject_rate": 0.35,
    "min_avg_faithfulness": 0.50,
    "min_avg_relevance": 0.50,
    "min_avg_keyword_recall": 0.50,
    "min_model_present_rate": 1.00,
}

BASELINE_PATH = ROOT / "reports" / "baseline_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG quality agent runner")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="RAG API base URL")
    parser.add_argument("--dataset", default=str(ROOT / "agents" / "eval_dataset.jsonl"))
    parser.add_argument("--out-dir", default=str(ROOT / "reports"))
    parser.add_argument("--limit", type=int, default=0, help="Limit dataset rows (0 means all)")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--allow-regression", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return rows


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_eval(base_url: str, dataset: list[dict[str, Any]], timeout: float) -> tuple[list[dict[str, Any]], list[str]]:
    scored: list[dict[str, Any]] = []
    errors: list[str] = []

    for sample in dataset:
        question = sample.get("question", "")
        if not question:
            errors.append(f"sample {sample.get('id')} missing question")
            continue

        start = time.perf_counter()
        try:
            response = post_json(f"{base_url.rstrip('/')}/ask", {"question": question}, timeout)
            latency_ms = (time.perf_counter() - start) * 1000
            scored.append(score_sample(sample, response, latency_ms))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            errors.append(f"HTTP {exc.code} for {sample.get('id')}: {body[:200]}")
        except URLError as exc:
            errors.append(f"Connection error for {sample.get('id')}: {exc}")
        except Exception as exc:
            errors.append(f"Unhandled error for {sample.get('id')}: {exc}")

    return scored, errors


def check_gates(summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if summary["answer_rate"] < DEFAULT_GATES["min_answer_rate"]:
        failures.append("answer_rate below gate")
    if summary["reject_rate"] > DEFAULT_GATES["max_reject_rate"]:
        failures.append("reject_rate above gate")
    if summary["avg_faithfulness"] < DEFAULT_GATES["min_avg_faithfulness"]:
        failures.append("avg_faithfulness below gate")
    if summary["avg_relevance"] < DEFAULT_GATES["min_avg_relevance"]:
        failures.append("avg_relevance below gate")
    if summary["avg_keyword_recall"] < DEFAULT_GATES["min_avg_keyword_recall"]:
        failures.append("avg_keyword_recall below gate")
    if summary["model_present_rate"] < DEFAULT_GATES["min_model_present_rate"]:
        failures.append("model_present_rate below gate")
    return failures


def check_regression(summary: dict[str, Any], baseline_path: Path) -> list[str]:
    if not baseline_path.exists():
        return []

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    tolerance = 0.03
    for key in ["answer_rate", "avg_faithfulness", "avg_relevance", "avg_keyword_recall"]:
        prev = baseline.get(key)
        cur = summary.get(key)
        if isinstance(prev, (int, float)) and isinstance(cur, (int, float)) and cur < (prev - tolerance):
            failures.append(f"regression on {key}: baseline={prev}, current={cur}")

    return failures


def write_report(out_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"quality_report_{ts}.json"
    md_path = out_dir / f"quality_report_{ts}.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = payload["summary"]
    lines = [
        "# RAG Quality Report",
        "",
        f"- Generated at (UTC): {payload['generated_at']}"
        f"\n- Dataset size: {summary['total']}"
        f"\n- Base URL: {payload['base_url']}",
        "",
        "## Summary",
        "",
        f"- answer_rate: {summary['answer_rate']}",
        f"- reject_rate: {summary['reject_rate']}",
        f"- avg_faithfulness: {summary['avg_faithfulness']}",
        f"- avg_relevance: {summary['avg_relevance']}",
        f"- avg_keyword_recall: {summary['avg_keyword_recall']}",
        f"- avg_best_similarity: {summary['avg_best_similarity']}",
        f"- model_present_rate: {summary['model_present_rate']}",
        f"- latency_p50_ms: {summary['latency_p50_ms']}",
        f"- latency_p95_ms: {summary['latency_p95_ms']}",
        "",
        "## Gate Failures",
        "",
    ]

    failures = payload.get("failures", [])
    if failures:
        lines.extend([f"- {f}" for f in failures])
    else:
        lines.append("- none")

    if payload.get("errors"):
        lines.extend(["", "## Runtime Errors", ""])
        lines.extend([f"- {e}" for e in payload["errors"]])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)

    if not dataset_path.exists():
        print(f"dataset not found: {dataset_path}", file=sys.stderr)
        return 2

    dataset = read_jsonl(dataset_path)
    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]

    scored, errors = run_eval(args.base_url, dataset, timeout=args.timeout)
    summary = aggregate(scored)

    failures = check_gates(summary)
    if not args.allow_regression:
        failures.extend(check_regression(summary, BASELINE_PATH))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "gates": DEFAULT_GATES,
        "summary": summary,
        "failures": failures,
        "errors": errors,
        "samples": scored,
    }

    json_path, md_path = write_report(out_dir, payload)

    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote report json: {json_path}")
    print(f"wrote report md:   {md_path}")
    if args.update_baseline:
        print(f"updated baseline:  {BASELINE_PATH}")

    if errors:
        print(f"runtime errors: {len(errors)}")
    if failures:
        print("quality gate failed:")
        for f in failures:
            print(f"- {f}")
        return 1

    print("quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
