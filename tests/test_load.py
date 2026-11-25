"""Tests for the Postgres loader: pure dtype/DDL helpers and a local-DB integration test."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import psycopg
import pytest

from retail_sales_pipeline.load import (
    add_foreign_key_sql,
    collect_tables,
    create_table_sql,
    load_to_postgres,
    pg_type,
    quote_ident,
    warehouse_dsn,
)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pl.Int64(), "BIGINT"),
        (pl.Int32(), "BIGINT"),
        (pl.UInt16(), "BIGINT"),
        (pl.Float64(), "DOUBLE PRECISION"),
        (pl.Float32(), "DOUBLE PRECISION"),
        (pl.Boolean(), "BOOLEAN"),
        (pl.Date(), "DATE"),
        (pl.Datetime("us"), "TIMESTAMP"),
        (pl.String(), "TEXT"),
    ],
)
def test_pg_type_maps_polars_dtypes(dtype: pl.DataType, expected: str) -> None:
    assert pg_type(dtype) == expected


def test_create_table_sql_types_columns_and_primary_key() -> None:
    columns = {
        "order_id": pl.String(),
        "date_key": pl.Int64(),
        "revenue_usd": pl.Float64(),
        "is_realised": pl.Boolean(),
        "date": pl.Date(),
    }
    ddl = create_table_sql("analytics", "fact_sales", columns, ("order_id",))
    assert ddl.startswith('CREATE TABLE "analytics"."fact_sales" (')
    assert '"order_id" TEXT' in ddl
    assert '"date_key" BIGINT' in ddl
    assert '"revenue_usd" DOUBLE PRECISION' in ddl
    assert '"is_realised" BOOLEAN' in ddl
    assert '"date" DATE' in ddl
    assert ddl.rstrip().endswith('PRIMARY KEY ("order_id")\n)')


def test_create_table_sql_without_primary_key() -> None:
    ddl = create_table_sql("analytics", "sales_yearly", {"year": pl.Int64()})
    assert "PRIMARY KEY" not in ddl


def test_quote_ident_escapes_double_quotes() -> None:
    assert quote_ident('we"ird') == '"we""ird"'


def test_add_foreign_key_sql() -> None:
    sql = add_foreign_key_sql("analytics", "fact_sales", "product", "dim_product", "product")
    assert sql == (
        'ALTER TABLE "analytics"."fact_sales" ADD CONSTRAINT "fk_fact_sales_product" '
        'FOREIGN KEY ("product") REFERENCES "analytics"."dim_product" ("product")'
    )


def test_collect_tables_reads_csv_and_quality_report(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    marts = tmp_path / "marts"
    reports = tmp_path / "reports"
    for directory in (processed, marts, reports):
        directory.mkdir()
    (processed / "dim_date.csv").write_text("date_key,date\n20150101,2015-01-01\n")
    (marts / "kpi_summary.csv").write_text("total_revenue_usd\n10.5\n")
    (reports / "quality_report.json").write_text(
        json.dumps(
            {
                "total_rows": 3,
                "valid_rows": 2,
                "rejected_rows": 1,
                "valid_pct": 66.67,
                "duration_seconds": 0.1,
                "errors_by_rule": {"revenue_inconsistent": 1},
            }
        )
    )
    tables = collect_tables(processed, marts, reports)
    assert set(tables) == {"dim_date", "kpi_summary", "quality_report"}
    assert tables["dim_date"].schema["date"] == pl.Date
    assert tables["quality_report"].row(0, named=True)["rejected_rows"] == 1
    assert json.loads(tables["quality_report"]["errors_by_rule"][0]) == {"revenue_inconsistent": 1}


def _warehouse_reachable() -> str | None:
    """Return the DSN if the local warehouse accepts connections, otherwise None."""
    try:
        dsn = warehouse_dsn()
        with psycopg.connect(dsn, connect_timeout=2):
            pass
    except (ValueError, psycopg.OperationalError):
        return None
    return dsn


@pytest.mark.integration
def test_load_to_postgres_round_trip(tmp_path: Path) -> None:
    dsn = _warehouse_reachable()
    if dsn is None:
        pytest.skip("local warehouse not reachable (run `make up` and set WAREHOUSE_* in .env)")

    processed = tmp_path / "processed"
    marts = tmp_path / "marts"
    reports = tmp_path / "reports"
    for directory in (processed, marts, reports):
        directory.mkdir()
    (processed / "dim_product.csv").write_text("product_key,product\n1,Product_A\n2,Product_B\n")
    (processed / "fact_sales.csv").write_text(
        "order_id,product,revenue_usd,is_realised,date\n"
        "ORD-2020-000001,Product_A,10.5,true,2020-01-02\n"
        "ORD-2020-000002,Product_B,,false,2020-01-03\n"
    )
    (reports / "quality_report.json").write_text(
        json.dumps(
            {
                "total_rows": 2,
                "valid_rows": 2,
                "rejected_rows": 0,
                "valid_pct": 100.0,
                "duration_seconds": 0.5,
                "errors_by_rule": {},
            }
        )
    )
    schema = "analytics_test"
    try:
        counts = load_to_postgres(
            processed_dir=processed,
            marts_dir=marts,
            dsn=dsn,
            schema=schema,
            reports_dir=reports,
        )
        assert counts == {"dim_product": 2, "fact_sales": 2, "quality_report": 1}
        with psycopg.connect(dsn) as conn:
            fact_count = conn.execute(f'SELECT count(*) FROM "{schema}".fact_sales').fetchone()
            assert fact_count == (2,)
            # Foreign key from fact_sales to dim_product must exist.
            fk = conn.execute(
                "SELECT count(*) FROM information_schema.table_constraints "
                "WHERE constraint_name = 'fk_fact_sales_product' AND table_schema = %s",
                (schema,),
            ).fetchone()
            assert fk == (1,)
    finally:
        with psycopg.connect(dsn) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
