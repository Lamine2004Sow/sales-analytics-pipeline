"""Text-to-SQL agent: question -> guarded SQL -> rows, or an explained refusal.

One model call writes the SQL (or refuses). If the guardrails or Postgres reject it, the error
is sent back once for a repair attempt. The schema description is read from the warehouse, and
the business rules below are the same definitions the dashboard uses.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import psycopg

from retail_sales_pipeline.assistant.guard import UnsafeSQLError, guard_sql
from retail_sales_pipeline.assistant.llm import ChatModel
from retail_sales_pipeline.config import WAREHOUSE_SCHEMA

MAX_ATTEMPTS = 2
STATEMENT_TIMEOUT_MS = 10_000

RULES = """You answer business questions about a retail sales warehouse by writing ONE PostgreSQL
SELECT query over the tables below (schema "analytics"). You never modify data.

Business definitions (same as the company dashboard):
- Revenue, profit, units, order counts, customers and average order value use realised orders
  only: fact_sales.is_realised = true (order_status 'Completed'), unless the question is about
  order statuses, cancellations or refunds, which use all orders.
- Money is in USD (columns ending in _usd). Round money and percentages to 2 decimals.
  Percentages are on a 0-100 scale. Measures are double precision, and PostgreSQL has no
  ROUND(double precision, int): cast first, as in ROUND(SUM(revenue_usd)::numeric, 2).
- Dates: join analytics.dim_date ON dim_date.date_key = fact_sales.date_key.
- Region: join analytics.dim_geography ON dim_geography.country = fact_sales.country.
- Customer segment and industry: read them from fact_sales (order level), not dim_customer.
- Return only the columns needed to answer, with readable aliases.

Refuse (set "sql" to null and explain in "refusal") when the question:
- asks for a forecast or prediction of future sales: a feasibility audit showed that no model
  beats the historical mean on this data;
- asks why something happened (causes): the warehouse holds no explanatory data;
- needs data that is not in the tables (weather, competitors, contact details...);
- asks to change, delete or create anything.

Reply with a JSON object only:
{"sql": "<query or null>", "explanation": "<one sentence on what the query computes>",
 "refusal": "<reason or null>"}"""


def rules_fingerprint() -> str:
    """Hash of the instructions: an eval run is only valid for the prompt it was made with."""
    return hashlib.sha256(RULES.encode("utf-8")).hexdigest()[:12]


def describe_schema(conn: psycopg.Connection, schema: str = WAREHOUSE_SCHEMA) -> dict[str, str]:
    """Table name -> 'col type, col type, ...' read from information_schema."""
    rows = conn.execute(
        "SELECT table_name, string_agg(column_name || ' ' || data_type, ', ' "
        "ORDER BY ordinal_position) FROM information_schema.columns "
        "WHERE table_schema = %s GROUP BY table_name ORDER BY table_name",
        (schema,),
    ).fetchall()
    return {table: columns for table, columns in rows}


def system_prompt(tables: dict[str, str]) -> str:
    schema_lines = "\n".join(f"- analytics.{name}({columns})" for name, columns in tables.items())
    return f"{RULES}\n\nTables:\n{schema_lines}"


@dataclass
class Answer:
    question: str
    sql: str | None = None
    explanation: str | None = None
    refusal: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    error: str | None = None
    attempts: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def answered(self) -> bool:
        return self.refusal is None and self.error is None and self.sql is not None


Executor = Callable[[str], tuple[list[str], list[tuple[Any, ...]]]]


def postgres_executor(conn: psycopg.Connection) -> Executor:
    """Run a query in a read-only transaction with a timeout; roll back afterwards."""

    def run(sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
        with conn.transaction(force_rollback=True), conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
            cur.execute(f"SET LOCAL search_path = {WAREHOUSE_SCHEMA}")
            cur.execute(sql.encode("utf-8"))
            columns = [d.name for d in cur.description or []]
            return columns, cur.fetchall()

    return run


def _parse_reply(content: str) -> dict[str, Any]:
    try:
        reply = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model reply is not JSON: {exc}") from exc
    if not isinstance(reply, dict):
        raise ValueError("model reply is not a JSON object")
    return reply


class SQLAgent:
    def __init__(self, model: ChatModel, tables: dict[str, str], execute: Executor) -> None:
        self.model = model
        self.tables = tables
        self.execute = execute
        self._system = system_prompt(tables)

    def ask(self, question: str) -> Answer:
        answer = Answer(question=question)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": question},
        ]
        start = time.perf_counter()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            answer.attempts = attempt
            completion = self.model.complete(messages)
            answer.prompt_tokens += completion.prompt_tokens
            answer.completion_tokens += completion.completion_tokens
            problem = self._try(answer, completion.content)
            if problem is None:
                break
            messages += [
                {"role": "assistant", "content": completion.content},
                {"role": "user", "content": f"That failed: {problem}. Reply with fixed JSON."},
            ]
        answer.latency_s = time.perf_counter() - start
        return answer

    def _try(self, answer: Answer, content: str) -> str | None:
        """Fill `answer` from one model reply. Returns an error to repair, or None when done."""
        answer.error = None
        try:
            reply = _parse_reply(content)
        except ValueError as exc:
            answer.error = str(exc)
            return answer.error
        answer.explanation = reply.get("explanation")
        if reply.get("refusal") or not reply.get("sql"):
            answer.refusal = reply.get("refusal") or "no SQL returned"
            answer.sql = None
            return None
        answer.refusal = None
        try:
            answer.sql = guard_sql(str(reply["sql"]), self.tables)
            answer.columns, answer.rows = self.execute(answer.sql)
        except UnsafeSQLError as exc:
            answer.sql, answer.error = str(reply["sql"]), f"rejected by guardrails: {exc}"
            return answer.error
        except psycopg.Error as exc:
            answer.error = f"database error: {str(exc).strip().splitlines()[0]}"
            return answer.error
        return None


def format_rows(columns: Sequence[str], rows: Sequence[tuple[Any, ...]], limit: int = 20) -> str:
    """Plain-text table for the CLI."""
    shown = [[str(v) for v in row] for row in rows[:limit]]
    widths = [max([len(c)] + [len(r[i]) for r in shown]) for i, c in enumerate(columns)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)) for r in shown]
    if len(rows) > limit:
        lines.append(f"... {len(rows) - limit} more rows")
    return "\n".join(lines)
