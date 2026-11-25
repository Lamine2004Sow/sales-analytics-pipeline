"""Load the pipeline outputs (star schema, KPI marts, quality report) into Postgres.

Every CSV in `processed/` and `marts/` becomes a table of the `analytics` schema, and the
quality report becomes a one-row `quality_report` table. The whole load runs in one transaction.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

import polars as pl
import psycopg
from psycopg.conninfo import make_conninfo

from retail_sales_pipeline.config import (
    ENV_FILE,
    FOREIGN_KEYS,
    MARTS_DIR,
    PRIMARY_KEYS,
    PROCESSED_DIR,
    REPORTS_DIR,
    WAREHOUSE_ENV_KEYS,
    WAREHOUSE_SCHEMA,
)

logger = logging.getLogger(__name__)

QUALITY_TABLE = "quality_report"


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines. Comments, blank lines and surrounding quotes are ignored."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("\"'")
    return values


def warehouse_dsn(env_file: Path = ENV_FILE) -> str:
    """Build a libpq connection string for the host-side warehouse (os.environ, then .env)."""
    file_values = _read_env_file(env_file)
    settings: dict[str, str] = {}
    for key in WAREHOUSE_ENV_KEYS:
        value = os.environ.get(key) or file_values.get(key)
        if not value:
            raise ValueError(f"{key} is not set: export it or add it to {env_file.name}")
        settings[key] = value
    return make_conninfo(
        host=settings["WAREHOUSE_HOST"],
        port=settings["WAREHOUSE_PORT"],
        dbname=settings["WAREHOUSE_DB"],
        user=settings["WAREHOUSE_USER"],
        password=settings["WAREHOUSE_PASSWORD"],
        connect_timeout="5",
    )


def pg_type(dtype: pl.DataType) -> str:
    """Map a Polars dtype to the Postgres column type used by the loader."""
    if dtype.is_integer():
        return "BIGINT"
    if dtype.is_float():
        return "DOUBLE PRECISION"
    if dtype == pl.Boolean:
        return "BOOLEAN"
    if dtype == pl.Date:
        return "DATE"
    if dtype == pl.Datetime:
        return "TIMESTAMP"
    return "TEXT"


def quote_ident(name: str) -> str:
    """Quote an SQL identifier (table, column, constraint)."""
    return '"' + name.replace('"', '""') + '"'


def qualified(schema: str, table: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table)}"


def create_table_sql(
    schema: str,
    table: str,
    columns: Mapping[str, pl.DataType],
    primary_key: tuple[str, ...] | None = None,
) -> str:
    """CREATE TABLE statement with one typed column per entry and an optional primary key."""
    definitions = [f"{quote_ident(name)} {pg_type(dtype)}" for name, dtype in columns.items()]
    if primary_key:
        keys = ", ".join(quote_ident(column) for column in primary_key)
        definitions.append(f"PRIMARY KEY ({keys})")
    body = ",\n    ".join(definitions)
    return f"CREATE TABLE {qualified(schema, table)} (\n    {body}\n)"


def add_foreign_key_sql(
    schema: str, table: str, column: str, ref_table: str, ref_column: str
) -> str:
    """ALTER TABLE statement adding a foreign key constraint named fk_<table>_<column>."""
    constraint = quote_ident(f"fk_{table}_{column}")
    return (
        f"ALTER TABLE {qualified(schema, table)} ADD CONSTRAINT {constraint} "
        f"FOREIGN KEY ({quote_ident(column)}) "
        f"REFERENCES {qualified(schema, ref_table)} ({quote_ident(ref_column)})"
    )


def read_csv_table(path: Path) -> pl.DataFrame:
    """Read a pipeline CSV, parsing ISO dates (for example `date`, `first_order_date`)."""
    return pl.read_csv(path, try_parse_dates=True, infer_schema_length=100_000)


def quality_report_frame(path: Path) -> pl.DataFrame:
    """Flatten reports/quality_report.json into a one-row frame (errors_by_rule kept as JSON)."""
    report = json.loads(path.read_text(encoding="utf-8"))
    return pl.DataFrame(
        {
            "total_rows": [int(report["total_rows"])],
            "valid_rows": [int(report["valid_rows"])],
            "rejected_rows": [int(report["rejected_rows"])],
            "valid_pct": [float(report["valid_pct"])],
            "duration_seconds": [float(report["duration_seconds"])],
            "errors_by_rule": [json.dumps(report.get("errors_by_rule", {}), ensure_ascii=False)],
        }
    )


def collect_tables(
    processed_dir: Path, marts_dir: Path, reports_dir: Path
) -> dict[str, pl.DataFrame]:
    """Every CSV of the two output directories plus the quality report, keyed by table name."""
    tables: dict[str, pl.DataFrame] = {}
    for directory in (processed_dir, marts_dir):
        for path in sorted(directory.glob("*.csv")):
            if path.stem in tables:
                raise ValueError(f"duplicate table name {path.stem!r} in {directory}")
            tables[path.stem] = read_csv_table(path)
    report_path = reports_dir / "quality_report.json"
    if report_path.is_file():
        tables[QUALITY_TABLE] = quality_report_frame(report_path)
    else:
        logger.warning("quality report missing, table %s skipped: %s", QUALITY_TABLE, report_path)
    return tables


def load_to_postgres(
    processed_dir: Path = PROCESSED_DIR,
    marts_dir: Path = MARTS_DIR,
    dsn: str | None = None,
    schema: str = WAREHOUSE_SCHEMA,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, int]:
    """Replace the warehouse tables with the current outputs. Returns row counts per table."""
    tables = collect_tables(Path(processed_dir), Path(marts_dir), Path(reports_dir))
    counts: dict[str, int] = {}
    dsn = dsn or warehouse_dsn()

    # One transaction: the psycopg connection commits on success and rolls back on error.
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)}")
        for table in reversed(list(tables)):
            cur.execute(f"DROP TABLE IF EXISTS {qualified(schema, table)} CASCADE")

        for table, frame in tables.items():
            cur.execute(create_table_sql(schema, table, frame.schema, PRIMARY_KEYS.get(table)))
            payload = frame.write_csv().encode("utf-8")
            with cur.copy(
                f"COPY {qualified(schema, table)} FROM STDIN (FORMAT csv, HEADER true)"
            ) as copy:
                copy.write(payload)
            counts[table] = frame.height
            logger.info("loaded %s.%s: %d rows", schema, table, frame.height)

        for table, column, ref_table, ref_column in FOREIGN_KEYS:
            if table in tables and ref_table in tables:
                cur.execute(add_foreign_key_sql(schema, table, column, ref_table, ref_column))

        for table in tables:
            cur.execute(f"ANALYZE {qualified(schema, table)}")

    logger.info("warehouse load complete: %d tables in schema %s", len(counts), schema)
    return counts
