"""End-to-end tests: run the pipeline on a small slice of the real raw CSV."""

import json
from pathlib import Path

import polars as pl
import pytest

from retail_sales_pipeline.cli import main
from retail_sales_pipeline.config import RAW_CSV
from retail_sales_pipeline.pipeline import DataQualityError, run_pipeline

if not RAW_CSV.is_file():
    pytest.skip(f"raw CSV not found: {RAW_CSV}", allow_module_level=True)

N_ROWS = 299
BAD_ORDER_ID = "ORD-2017-999999"
EXPECTED_FILES = [
    "processed/fact_sales.csv",
    "processed/dim_date.csv",
    "processed/dim_customer.csv",
    "processed/dim_product.csv",
    "processed/dim_geography.csv",
    "marts/kpi_summary.csv",
    "reports/quality_report.json",
    "reports/rejected_rows.csv",
    "reports/run_summary.json",
]


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Header + 299 real rows, plus one currency-mismatch row and one duplicate order_id."""
    base = pl.read_csv(RAW_CSV, n_rows=N_ROWS, infer_schema_length=100_000)
    assert base["country"][0] == "China"  # China must use CNY, so USD is a mismatch
    bad_currency = base.head(1).with_columns(
        pl.lit(BAD_ORDER_ID).alias("order_id"), pl.lit("USD").alias("currency")
    )
    duplicate = base.tail(1)  # exact copy of an existing order_id, appended last
    path = tmp_path / "raw.csv"
    pl.concat([base, bad_currency, duplicate], how="vertical").write_csv(path)
    return path


def _dirs(root: Path) -> dict[str, Path]:
    return {
        "processed_dir": root / "processed",
        "marts_dir": root / "marts",
        "reports_dir": root / "reports",
    }


def test_end_to_end_run(sample_csv, tmp_path):
    summary = run_pipeline(raw_csv=sample_csv, **_dirs(tmp_path))

    for rel in EXPECTED_FILES:
        assert (tmp_path / rel).is_file(), rel

    rows = summary["rows"]
    assert rows["rejected"] == 2
    assert rows["valid"] == N_ROWS
    assert rows["clean"] == rows["valid"]
    assert pl.read_csv(tmp_path / "processed" / "fact_sales.csv").height == rows["valid"]
    assert pl.read_csv(tmp_path / "reports" / "rejected_rows.csv").height == 2

    on_disk = json.loads((tmp_path / "reports" / "run_summary.json").read_text(encoding="utf-8"))
    assert on_disk["rows"] == rows


def test_rejected_threshold_raises_after_writing_reports(sample_csv, tmp_path):
    with pytest.raises(DataQualityError):
        run_pipeline(raw_csv=sample_csv, max_rejected_pct=0.1, **_dirs(tmp_path))

    assert (tmp_path / "reports" / "quality_report.json").is_file()
    assert not (tmp_path / "processed" / "fact_sales.csv").exists()


def test_missing_raw_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="raw sales CSV not found"):
        run_pipeline(raw_csv=tmp_path / "missing.csv", **_dirs(tmp_path))


def test_cli_run_returns_zero(sample_csv, tmp_path):
    code = main(
        [
            "run",
            "--input",
            str(sample_csv),
            "--output-dir",
            str(tmp_path / "data"),
            "--reports-dir",
            str(tmp_path / "reports"),
        ]
    )
    assert code == 0
    assert (tmp_path / "data" / "processed" / "fact_sales.csv").is_file()
    assert (tmp_path / "data" / "marts" / "kpi_summary.csv").is_file()


def test_cli_missing_input_returns_two(tmp_path):
    code = main(
        [
            "run",
            "--input",
            str(tmp_path / "missing.csv"),
            "--output-dir",
            str(tmp_path / "data"),
            "--reports-dir",
            str(tmp_path / "reports"),
        ]
    )
    assert code == 2
