"""End-to-end run: raw CSV -> validation -> cleaning -> star schema -> KPI marts -> run summary."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import polars as pl

from retail_sales_pipeline.config import (
    MARTS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_CSV,
    REPORTS_DIR,
)
from retail_sales_pipeline.export import write_tables
from retail_sales_pipeline.transform import (
    build_star_schema,
    clean,
    compute_kpis,
    count_customer_attribute_conflicts,
)
from retail_sales_pipeline.validate import (
    ValidationResult,
    validate_dataframe,
    write_validation_outputs,
)

logger = logging.getLogger(__name__)


class DataQualityError(RuntimeError):
    """Raised when the share of rejected rows is above the allowed threshold."""


def extract(raw_csv: Path) -> pl.DataFrame:
    """Read the raw sales CSV, raising FileNotFoundError with a clear message if it is absent."""
    if not raw_csv.is_file():
        raise FileNotFoundError(f"raw sales CSV not found: {raw_csv}")
    return pl.read_csv(raw_csv, infer_schema_length=100_000)


def _display_path(path: Path) -> str:
    """Path relative to the project root when possible, otherwise as given."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


@contextmanager
def _timed(step: str, durations: dict[str, float]) -> Iterator[None]:
    """Record the duration of a step (2 decimals) and log one INFO line when it completes."""
    started = time.perf_counter()
    yield
    elapsed = time.perf_counter() - started
    durations[step] = round(elapsed, 2)
    logger.info("%s done in %.2fs", step, elapsed)


def run_pipeline(
    raw_csv: Path = RAW_CSV,
    processed_dir: Path = PROCESSED_DIR,
    marts_dir: Path = MARTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    max_rejected_pct: float = 5.0,
) -> dict[str, Any]:
    """Run every pipeline step and return the run summary (also written to run_summary.json).

    Raises FileNotFoundError if `raw_csv` is missing and DataQualityError if the rejected share
    of rows exceeds `max_rejected_pct` (validation outputs are written first in that case).
    """
    raw_csv = Path(raw_csv)
    total_started = time.perf_counter()
    durations: dict[str, float] = {}
    written: list[str] = []

    with _timed("extract", durations):
        raw = extract(raw_csv)

    with _timed("validate", durations):
        result: ValidationResult = validate_dataframe(raw)

    with _timed("write_validation", durations):
        write_validation_outputs(result, reports_dir)
    written += [
        _display_path(reports_dir / "quality_report.json"),
        _display_path(reports_dir / "rejected_rows.csv"),
    ]

    rejected = result.report["rejected_rows"]
    rejected_pct = 100 * rejected / raw.height if raw.height else 0.0
    if rejected_pct > max_rejected_pct:
        raise DataQualityError(
            f"{rejected} of {raw.height} rows rejected ({rejected_pct:.2f}%), above the "
            f"{max_rejected_pct}% threshold; see {reports_dir / 'rejected_rows.csv'}"
        )

    with _timed("clean", durations):
        cleaned = clean(result.valid)

    with _timed("star_schema", durations):
        star_tables = build_star_schema(cleaned)

    with _timed("write_processed", durations):
        written += [_display_path(p) for p in write_tables(star_tables, processed_dir)]

    with _timed("kpis", durations):
        kpi_tables = compute_kpis(cleaned)

    with _timed("write_marts", durations):
        written += [_display_path(p) for p in write_tables(kpi_tables, marts_dir)]

    summary: dict[str, Any] = {
        "raw_csv": _display_path(raw_csv),
        "total_s": round(time.perf_counter() - total_started, 2),
        "durations_s": durations,
        "rows": {
            "raw": raw.height,
            "valid": result.valid.height,
            "rejected": rejected,
            "clean": cleaned.height,
        },
        "customer_attribute_conflicts": count_customer_attribute_conflicts(cleaned),
        "written_files": written,
        "kpi_summary": kpi_tables["kpi_summary"].row(0, named=True),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return summary
