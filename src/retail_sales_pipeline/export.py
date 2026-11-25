"""CSV export of star-schema and KPI tables (inputs for the Power BI dashboard)."""

from pathlib import Path

import polars as pl


def write_tables(tables: dict[str, pl.DataFrame], out_dir: Path) -> list[Path]:
    """Write each table as `<name>.csv` in `out_dir` and return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, table in tables.items():
        path = out_dir / f"{name}.csv"
        table.write_csv(path)
        paths.append(path)
    return paths
