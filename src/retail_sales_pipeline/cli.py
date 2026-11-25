"""Command line interface: `retail-pipeline run` and `retail-pipeline validate`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import psycopg

from retail_sales_pipeline.config import DATA_DIR, MARTS_DIR, PROCESSED_DIR, RAW_CSV, REPORTS_DIR
from retail_sales_pipeline.load import load_to_postgres
from retail_sales_pipeline.pipeline import DataQualityError, extract, run_pipeline
from retail_sales_pipeline.validate import validate_dataframe, write_validation_outputs

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
EXIT_OK = 0
EXIT_DATA_QUALITY = 1
EXIT_MISSING_INPUT = 2
EXIT_WAREHOUSE_UNAVAILABLE = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retail-pipeline", description="Retail sales ETL: validate, clean, export."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the full pipeline")
    run.add_argument("--input", type=Path, default=RAW_CSV, help="raw sales CSV")
    run.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="base data directory; processed/ and marts/ are written below it",
    )
    run.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR, help="quality report and run summary"
    )
    run.add_argument(
        "--max-rejected-pct",
        type=float,
        default=5.0,
        help="fail (exit 1) when the rejected share of rows exceeds this (default: 5.0)",
    )
    run.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    validate = commands.add_parser("validate", help="validate only and print the quality report")
    validate.add_argument("--input", type=Path, default=RAW_CSV, help="raw sales CSV")
    validate.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR, help="where to write the quality report"
    )
    validate.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    load = commands.add_parser("load", help="load the outputs into the Postgres warehouse")
    load.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DIR, help="star schema CSV directory"
    )
    load.add_argument("--marts-dir", type=Path, default=MARTS_DIR, help="KPI mart CSV directory")
    load.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR, help="directory of quality_report.json"
    )
    load.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def _cmd_load(args: argparse.Namespace) -> int:
    try:
        counts = load_to_postgres(
            processed_dir=args.processed_dir,
            marts_dir=args.marts_dir,
            reports_dir=args.reports_dir,
        )
    except (psycopg.OperationalError, ValueError) as exc:
        # ValueError: a WAREHOUSE_* setting is missing (raised by warehouse_dsn).
        print(
            f"error: cannot load into the Postgres warehouse ({exc}).\n"
            "hint: run `make up` first, and check WAREHOUSE_* in .env",
            file=sys.stderr,
        )
        return EXIT_WAREHOUSE_UNAVAILABLE
    for table, rows in counts.items():
        print(f"{table}: {rows} rows")
    print(f"Loaded {len(counts)} tables into the warehouse.")
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    summary = run_pipeline(
        raw_csv=args.input,
        processed_dir=args.output_dir / "processed",
        marts_dir=args.output_dir / "marts",
        reports_dir=args.reports_dir,
        max_rejected_pct=args.max_rejected_pct,
    )
    rows = summary["rows"]
    revenue = summary["kpi_summary"]["total_revenue_usd"]
    print(
        f"Rows: raw={rows['raw']} valid={rows['valid']} "
        f"rejected={rows['rejected']} clean={rows['clean']}"
    )
    print(f"Realised revenue: {revenue:,.2f} USD")
    print(f"Duration: {summary['total_s']:.2f}s (outputs: {args.output_dir}, {args.reports_dir})")
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    result = validate_dataframe(extract(args.input))
    write_validation_outputs(result, args.reports_dir)
    print(json.dumps(result.report, indent=2, ensure_ascii=False))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns 0 on success, 1 on a data quality failure, 2 on a missing input."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=LOG_FORMAT)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "load":
            return _cmd_load(args)
        return _cmd_validate(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MISSING_INPUT
    except DataQualityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA_QUALITY


if __name__ == "__main__":
    raise SystemExit(main())
