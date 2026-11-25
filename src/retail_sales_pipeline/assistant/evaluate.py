"""Evaluation bench for the text-to-SQL agent.

Every question in eval/questions.json either has a reference SQL query or expects a refusal.
An answer counts as correct when the rows it returns match the rows of the reference query
(execution accuracy): row order is ignored, extra columns are allowed, values are compared after
rounding numbers to 2 decimals. Refusal questions are correct when the agent refuses.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from retail_sales_pipeline.assistant.agent import Answer, Executor, rules_fingerprint
from retail_sales_pipeline.config import PROJECT_ROOT

BENCH_FILE = PROJECT_ROOT / "eval" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

Row = tuple[Any, ...]


def load_bench(path: Path = BENCH_FILE) -> list[dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in the bench")
    for item in items:
        if bool(item.get("ref_sql")) == bool(item.get("expect_refusal")):
            raise ValueError(f"{item['id']}: give either ref_sql or expect_refusal")
    return items


def bench_fingerprint(path: Path = BENCH_FILE) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def normalize(value: Any) -> Any:
    """Comparable form of a cell: numbers rounded half-up to 2 decimals, dates as ISO text."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float | Decimal):
        return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if isinstance(value, dt.date | dt.datetime):
        return value.isoformat()
    return str(value).strip()


def results_match(gold: Sequence[Row], predicted: Sequence[Row]) -> bool:
    """True when some choice of predicted columns reproduces the gold rows (as a multiset)."""
    gold_rows = [tuple(normalize(v) for v in row) for row in gold]
    pred_rows = [tuple(normalize(v) for v in row) for row in predicted]
    if len(gold_rows) != len(pred_rows):
        return False
    if not gold_rows:
        return True
    n_gold, n_pred = len(gold_rows[0]), len(pred_rows[0])

    def column(rows: list[Row], i: int) -> Counter:
        return Counter(row[i] for row in rows)

    # Candidate predicted columns for each gold column: same multiset of values.
    candidates = [
        [j for j in range(n_pred) if column(pred_rows, j) == column(gold_rows, i)]
        for i in range(n_gold)
    ]
    if any(not c for c in candidates):
        return False
    target = Counter(gold_rows)
    for choice in itertools.product(*candidates):
        if len(set(choice)) != len(choice):
            continue
        if Counter(tuple(row[j] for j in choice) for row in pred_rows) == target:
            return True
    return False


def grade(item: dict[str, Any], answer: Answer, execute: Executor) -> str:
    """Outcome label: correct, wrong_result, sql_error, false_refusal, missed_refusal."""
    if item.get("expect_refusal"):
        return "correct" if answer.refusal else "missed_refusal"
    if answer.refusal:
        return "false_refusal"
    if not answer.answered:
        return "sql_error"
    _, gold_rows = execute(item["ref_sql"])
    return "correct" if results_match(gold_rows, answer.rows) else "wrong_result"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline metrics, recomputable from the per-question records."""
    answerable = [r for r in records if not r["expect_refusal"]]
    refusals = [r for r in records if r["expect_refusal"]]
    latencies = [r["latency_s"] for r in records]

    def share(rows: list[dict[str, Any]], outcome: str | tuple[str, ...]) -> float:
        wanted = (outcome,) if isinstance(outcome, str) else outcome
        return round(sum(r["outcome"] in wanted for r in rows) / len(rows), 4) if rows else 0.0

    by_category = {
        category: share([r for r in records if r["category"] == category], "correct")
        for category in sorted({r["category"] for r in records})
    }
    return {
        "questions": len(records),
        "accuracy": share(records, "correct"),
        "execution_accuracy": share(answerable, "correct"),
        "valid_sql_rate": share(answerable, ("correct", "wrong_result")),
        "false_refusal_rate": share(answerable, "false_refusal"),
        "refusal_recall": share(refusals, "correct"),
        "accuracy_by_category": by_category,
        "outcomes": dict(Counter(r["outcome"] for r in records)),
        "latency_p50_s": round(statistics.median(latencies), 2) if latencies else 0.0,
        "latency_p95_s": round(_percentile(latencies, 0.95), 2),
        "repair_rate": share_attempts(records),
        "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in records)),
        "mean_completion_tokens": round(statistics.mean(r["completion_tokens"] for r in records)),
    }


def share_attempts(records: list[dict[str, Any]]) -> float:
    return round(sum(r["attempts"] > 1 for r in records) / len(records), 4) if records else 0.0


def run_bench(
    ask: Callable[[str], Answer],
    execute: Executor,
    model_name: str,
    bench: list[dict[str, Any]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    bench = bench if bench is not None else load_bench()
    records = []
    for item in bench:
        answer = ask(item["question"])
        outcome = grade(item, answer, execute)
        records.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "expect_refusal": bool(item.get("expect_refusal")),
                "outcome": outcome,
                "sql": answer.sql,
                "refusal": answer.refusal,
                "error": answer.error,
                "attempts": answer.attempts,
                "latency_s": round(answer.latency_s, 3),
                "prompt_tokens": answer.prompt_tokens,
                "completion_tokens": answer.completion_tokens,
            }
        )
        if progress:
            progress(f"{item['id']:<6} {outcome:<15} {item['question']}")
    return {
        "model": model_name,
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "bench_fingerprint": bench_fingerprint(),
        "rules_fingerprint": rules_fingerprint(),
        "summary": summarize(records),
        "records": records,
    }


def write_results(result: dict[str, Any], label: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Write eval/results/<model>_<label>.json (one file per prompt variant)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    result["label"] = label
    path = results_dir / f"{result['model']}_{label}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
