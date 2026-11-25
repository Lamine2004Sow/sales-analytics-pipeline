"""SQL guardrails: one read-only SELECT over whitelisted warehouse tables, with a row cap.

The model's SQL is parsed with sqlglot (Postgres dialect) and rejected before it reaches the
database when it is not a single query, writes or changes anything, reads a table outside the
whitelist, or calls a server function that can leak or stall (pg_*, dblink, set_config...).
Execution adds a second layer: a read-only transaction with a statement timeout.
"""

from __future__ import annotations

from collections.abc import Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

MAX_ROWS = 1000

# Statement types that change data, schema or session state, wherever they appear in the tree.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,
    exp.Set,
    exp.Copy,
    exp.Grant,
    exp.Into,
    exp.Lock,
)
_FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink", "set_config", "current_setting", "query_to")


class UnsafeSQLError(ValueError):
    """The SQL was rejected by the guardrails; the message says why."""


def _function_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def guard_sql(
    sql: str, allowed_tables: Iterable[str], schema: str = "analytics", max_rows: int = MAX_ROWS
) -> str:
    """Return the SQL to execute (with a LIMIT) or raise UnsafeSQLError."""
    allowed = {name.lower() for name in allowed_tables}
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except ParseError as exc:
        raise UnsafeSQLError(f"SQL does not parse: {exc.errors[0]['description']}") from exc
    if len(statements) != 1:
        raise UnsafeSQLError(f"expected exactly one statement, got {len(statements)}")
    tree = statements[0]

    if not isinstance(tree, exp.Query):
        raise UnsafeSQLError(f"only SELECT queries are allowed, got {tree.key.upper()}")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeSQLError(f"forbidden clause: {node.key.upper()}")
        if isinstance(node, exp.Func) and _function_name(node).startswith(
            _FORBIDDEN_FUNCTION_PREFIXES
        ):
            raise UnsafeSQLError(f"forbidden function: {_function_name(node)}")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name, db = table.name.lower(), table.db.lower()
        if not db and name in cte_names:
            continue
        if db and db != schema:
            raise UnsafeSQLError(f"schema {db!r} is not allowed (only {schema!r})")
        if name not in allowed:
            raise UnsafeSQLError(f"table {name!r} is not in the warehouse whitelist")

    return _with_row_cap(tree, max_rows)


def _with_row_cap(tree: exp.Query, max_rows: int) -> str:
    """Add LIMIT max_rows, or lower an existing larger literal LIMIT to it."""
    if isinstance(tree, exp.Select):
        limit = tree.args.get("limit")
        if limit is None:
            return tree.limit(max_rows).sql(dialect="postgres")
        value = limit.expression
        if isinstance(value, exp.Literal) and value.is_int and int(value.name) <= max_rows:
            return tree.sql(dialect="postgres")
        return tree.limit(max_rows).sql(dialect="postgres")
    # Set operations (UNION ...): cap through an outer query.
    return f"SELECT * FROM ({tree.sql(dialect='postgres')}) AS capped LIMIT {max_rows}"
