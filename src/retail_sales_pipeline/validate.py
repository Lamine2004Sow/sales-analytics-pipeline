"""Row-level validation of the raw sales CSV with Pydantic, plus a quality report."""

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from pydantic import ValidationError

from retail_sales_pipeline.config import RAW_CSV, REPORTS_DIR
from retail_sales_pipeline.schemas import SalesRecord

VALUE_ERROR_PREFIX = "Value error, "
RULE_SEPARATOR = "; "


@dataclass
class ValidationResult:
    """Outcome of validating a DataFrame row by row."""

    valid: pl.DataFrame  # valid rows, original columns and dtypes
    rejected: pl.DataFrame  # rejected rows + row_number (1-based) + errors
    report: dict


def _error_messages(exc: ValidationError) -> tuple[list[str], list[str]]:
    """Return (row-level messages, rule/field keys) for one failed row."""
    messages: list[str] = []
    keys: list[str] = []
    for err in exc.errors():
        msg = err["msg"].removeprefix(VALUE_ERROR_PREFIX)
        loc = ".".join(str(part) for part in err["loc"])
        if loc:
            # Field-level error (type, constraint, Literal...).
            messages.append(f"{loc}: {msg}")
            keys.append(loc)
        else:
            # Model-level error: one message per business rule, joined by "; ".
            for rule in msg.split(RULE_SEPARATOR):
                messages.append(rule)
                keys.append(rule.split(":", 1)[0].strip())
    return messages, keys


def validate_dataframe(df: pl.DataFrame) -> ValidationResult:
    """Validate every row of `df` against SalesRecord and split valid/rejected rows."""
    started = time.perf_counter()

    # Duplicate order_id: the first occurrence is kept for pydantic, later ones are rejected.
    first_seen = df.select(pl.col("order_id").is_first_distinct()).to_series().to_list()

    mask: list[bool] = []
    rejected_numbers: list[int] = []
    rejected_errors: list[str] = []
    errors_by_rule: Counter[str] = Counter()

    for index, (row, is_first) in enumerate(zip(df.iter_rows(named=True), first_seen, strict=True)):
        messages: list[str] = []
        if not is_first:
            messages.append("duplicate_order_id")
            errors_by_rule["duplicate_order_id"] += 1

        try:
            SalesRecord.model_validate(row)
        except ValidationError as exc:
            row_messages, keys = _error_messages(exc)
            messages.extend(row_messages)
            errors_by_rule.update(keys)

        mask.append(not messages)
        if messages:
            rejected_numbers.append(index + 1)
            rejected_errors.append(RULE_SEPARATOR.join(messages))

    mask_series = pl.Series("mask", mask, dtype=pl.Boolean)
    valid = df.filter(mask_series)
    rejected = df.filter(~mask_series).with_columns(
        pl.Series("row_number", rejected_numbers, dtype=pl.Int64),
        pl.Series("errors", rejected_errors, dtype=pl.String),
    )

    total = df.height
    valid_rows = valid.height
    report = {
        "total_rows": total,
        "valid_rows": valid_rows,
        "rejected_rows": total - valid_rows,
        "valid_pct": round(100 * valid_rows / total, 2) if total else 0.0,
        "errors_by_rule": dict(errors_by_rule.most_common()),
        "duration_seconds": round(time.perf_counter() - started, 2),
    }
    return ValidationResult(valid=valid, rejected=rejected, report=report)


def write_validation_outputs(result: ValidationResult, reports_dir: Path = REPORTS_DIR) -> None:
    """Write quality_report.json and rejected_rows.csv (header written even when empty)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "quality_report.json").write_text(
        json.dumps(result.report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result.rejected.write_csv(reports_dir / "rejected_rows.csv")


if __name__ == "__main__":
    raw = pl.read_csv(RAW_CSV, infer_schema_length=100000)
    outcome = validate_dataframe(raw)
    print(json.dumps(outcome.report, indent=2, ensure_ascii=False))
