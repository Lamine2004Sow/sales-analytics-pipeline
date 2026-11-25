"""Text-to-SQL assistant: guardrails, result matching, agent loop and the recorded eval runs.

Everything here runs offline (fake model, fake executor, committed result files), so it runs in CI
without a database or an API key.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from retail_sales_pipeline.assistant.agent import (
    PROMPT_VARIANTS,
    RULES,
    SQLAgent,
    rules_fingerprint,
)
from retail_sales_pipeline.assistant.evaluate import (
    BENCH_FILE,
    RESULTS_DIR,
    bench_fingerprint,
    grade,
    load_bench,
    results_match,
    summarize,
)
from retail_sales_pipeline.assistant.guard import UnsafeSQLError, guard_sql
from retail_sales_pipeline.assistant.llm import Completion

TABLES = {"fact_sales": "order_id text, revenue_usd double precision", "dim_date": "year bigint"}

# --------------------------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT 1", "SELECT 1 LIMIT 1000"),
        ("SELECT * FROM fact_sales LIMIT 5", "SELECT * FROM fact_sales LIMIT 5"),
        ("SELECT * FROM fact_sales LIMIT 5000", "SELECT * FROM fact_sales LIMIT 1000"),
        (
            "WITH y AS (SELECT * FROM analytics.fact_sales) SELECT * FROM y",
            "WITH y AS (SELECT * FROM analytics.fact_sales) SELECT * FROM y LIMIT 1000",
        ),
    ],
)
def test_guard_accepts_selects_and_caps_rows(sql: str, expected: str) -> None:
    assert guard_sql(sql, TABLES) == expected


def test_guard_caps_set_operations_through_an_outer_query() -> None:
    sql = guard_sql("SELECT year FROM dim_date UNION SELECT 1", TABLES)
    assert sql.startswith("SELECT * FROM (") and sql.endswith(") AS capped LIMIT 1000")


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DELETE FROM fact_sales", "only SELECT"),
        ("UPDATE fact_sales SET revenue_usd = 0", "only SELECT"),
        ("DROP TABLE analytics.fact_sales", "only SELECT"),
        ("SELECT 1; DROP TABLE fact_sales", "exactly one statement"),
        ("SELECT * INTO copy FROM fact_sales", "INTO"),
        ("SELECT * FROM fact_sales FOR UPDATE", "LOCK"),
        ("SELECT pg_sleep(10)", "pg_sleep"),
        ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
        ("SELECT current_setting('data_directory')", "current_setting"),
        ("SELECT * FROM pg_catalog.pg_user", "pg_catalog"),
        ("SELECT * FROM information_schema.tables", "information_schema"),
        ("SELECT * FROM dim_customer", "whitelist"),
        ("SELEC x", "only SELECT"),
    ],
)
def test_guard_rejects_unsafe_sql(sql: str, reason: str) -> None:
    with pytest.raises(UnsafeSQLError, match=reason):
        guard_sql(sql, TABLES)


# --------------------------------------------------------------------------------------------
# Result matching (execution accuracy)
# --------------------------------------------------------------------------------------------


def test_results_match_ignores_row_order_and_extra_columns() -> None:
    gold = [("USA", Decimal("10.00")), ("UK", Decimal("5.50"))]
    predicted = [("UK", 5.5, 3), ("USA", 10.0, 7)]
    assert results_match(gold, predicted)


def test_results_match_rounds_numbers_to_two_decimals() -> None:
    assert results_match([(Decimal("35.07"),)], [(35.0712,)])
    assert results_match([(5,)], [(Decimal("5.00"),)])


def test_results_match_rejects_wrong_values_scale_and_row_counts() -> None:
    assert not results_match([(Decimal("35.07"),)], [(0.3507,)])  # fraction instead of percent
    assert not results_match([("USA",)], [("USA",), ("UK",)])
    assert not results_match([("USA", 1)], [("USA", 2)])


def test_results_match_needs_consistent_rows_not_just_columns() -> None:
    gold = [("A", 1), ("B", 2)]
    shuffled = [("A", 2), ("B", 1)]  # same column multisets, wrong pairing
    assert not results_match(gold, shuffled)


def test_results_match_handles_swapped_column_order() -> None:
    assert results_match([("A", 1), ("B", 2)], [(1, "A"), (2, "B")])


# --------------------------------------------------------------------------------------------
# Agent loop with a scripted model
# --------------------------------------------------------------------------------------------


class ScriptedModel:
    name = "scripted"

    def __init__(self, replies: list[dict]) -> None:
        self.replies = [json.dumps(r) for r in replies]
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        self.calls.append(list(messages))
        return Completion(self.replies.pop(0), prompt_tokens=100, completion_tokens=10, latency_s=0)


def _executor(sql: str) -> tuple[list[str], list[tuple]]:
    return ["n"], [(42,)]


def test_agent_answers_and_records_tokens() -> None:
    model = ScriptedModel([{"sql": "SELECT count(*) AS n FROM fact_sales", "refusal": None}])
    answer = SQLAgent(model, TABLES, _executor).ask("How many orders?")
    assert answer.answered and answer.rows == [(42,)]
    assert answer.sql.endswith("LIMIT 1000")
    assert (answer.attempts, answer.prompt_tokens) == (1, 100)


def test_agent_repairs_once_after_a_guardrail_rejection() -> None:
    model = ScriptedModel(
        [
            {"sql": "SELECT * FROM dim_customer", "refusal": None},
            {"sql": "SELECT count(*) FROM fact_sales", "refusal": None},
        ]
    )
    answer = SQLAgent(model, TABLES, _executor).ask("How many customers?")
    assert answer.answered and answer.attempts == 2
    assert "whitelist" in model.calls[1][-1]["content"]  # the error is fed back to the model


def test_agent_gives_up_after_two_failed_attempts() -> None:
    bad = {"sql": "DELETE FROM fact_sales", "refusal": None}
    answer = SQLAgent(ScriptedModel([bad, bad]), TABLES, _executor).ask("Clean up")
    assert not answer.answered and "guardrails" in answer.error


def test_agent_passes_refusals_through_without_executing() -> None:
    def never(sql: str) -> tuple[list[str], list[tuple]]:
        raise AssertionError("a refusal must not reach the database")

    model = ScriptedModel([{"sql": None, "refusal": "forecasts are not supported"}])
    answer = SQLAgent(model, TABLES, never).ask("Forecast 2025 revenue")
    assert answer.refusal == "forecasts are not supported" and not answer.answered


def test_grade_labels_each_outcome() -> None:
    model = ScriptedModel([{"sql": None, "refusal": "no"}])
    refused = SQLAgent(model, TABLES, _executor).ask("q")
    assert grade({"expect_refusal": True}, refused, _executor) == "correct"
    assert grade({"ref_sql": "SELECT 42"}, refused, _executor) == "false_refusal"


# --------------------------------------------------------------------------------------------
# Bench file and recorded runs (regression checks, no API call)
# --------------------------------------------------------------------------------------------


def test_ablation_drops_only_the_business_definitions() -> None:
    ablated = PROMPT_VARIANTS["no-business-rules"]
    assert PROMPT_VARIANTS["full"] == RULES
    assert "is_realised" not in ablated and "dim_geography" not in ablated
    assert "ROUND(SUM(revenue_usd)::numeric, 2)" in ablated  # SQL conventions kept
    assert "forecast" in ablated and '"refusal"' in ablated  # refusals and format kept


def test_bench_is_well_formed() -> None:
    bench = load_bench()
    categories = {item["category"] for item in bench}
    assert {"dashboard", "multi_step", "advanced", "french", "refusal"} <= categories
    assert sum(1 for item in bench if item["category"] == "dashboard") == 26


def _result_files() -> list[Path]:
    return sorted(RESULTS_DIR.glob("*.json"))


@pytest.mark.parametrize("path", _result_files(), ids=lambda p: p.stem)
def test_recorded_run_matches_bench_and_its_own_summary(path: Path) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["bench_fingerprint"] == bench_fingerprint(BENCH_FILE), (
        f"{path.name} was made on another version of eval/questions.json: rerun the bench"
    )
    assert result["summary"] == summarize(result["records"])


def test_current_prompt_has_a_recorded_run_above_the_floor() -> None:
    """Changing the prompt without rerunning `make eval-assistant` fails here."""
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in _result_files()]
    current = [r for r in runs if r["rules_fingerprint"] == rules_fingerprint()]
    assert current, "no eval run for the current prompt: run `make eval-assistant`"
    summary = current[-1]["summary"]
    assert summary["execution_accuracy"] >= 0.95
    assert summary["refusal_recall"] == 1.0
    assert summary["valid_sql_rate"] == 1.0
