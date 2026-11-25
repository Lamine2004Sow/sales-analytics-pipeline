"""Shared constants: paths, allowed categorical values, business mappings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "global_sales_100k_50features.csv"
PROCESSED_DIR = DATA_DIR / "processed"  # star schema for Power BI
MARTS_DIR = DATA_DIR / "marts"  # pre-aggregated KPI tables
REPORTS_DIR = PROJECT_ROOT / "reports"

# Country -> sales region (the raw data has no region column).
COUNTRY_TO_REGION: dict[str, str] = {
    "USA": "Americas",
    "Canada": "Americas",
    "Brazil": "Americas",
    "UK": "EMEA",
    "Germany": "EMEA",
    "UAE": "EMEA",
    "China": "APAC",
    "India": "APAC",
    "Japan": "APAC",
    "Australia": "APAC",
}

COUNTRY_TO_CURRENCY: dict[str, str] = {
    "USA": "USD",
    "Canada": "CAD",
    "Brazil": "BRL",
    "UK": "GBP",
    "Germany": "EUR",
    "UAE": "AED",
    "China": "CNY",
    "India": "INR",
    "Japan": "JPY",
    "Australia": "AUD",
}

SEASONS = ("Winter", "Spring", "Summer", "Fall")
MONTH_TO_SEASON: dict[int, str] = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Fall",
    10: "Fall",
    11: "Fall",
}

# Only completed orders count as realised revenue in the KPIs.
REALISED_STATUS = "Completed"

# Relative tolerance for recomputed amounts (source values are rounded to cents).
AMOUNT_REL_TOL = 0.01
AMOUNT_ABS_TOL = 0.05

# Local warehouse (docker compose service `warehouse`), see docs/metabase.md.
ENV_FILE = PROJECT_ROOT / ".env"
WAREHOUSE_SCHEMA = "analytics"
WAREHOUSE_ENV_KEYS = (
    "WAREHOUSE_HOST",
    "WAREHOUSE_PORT",
    "WAREHOUSE_DB",
    "WAREHOUSE_USER",
    "WAREHOUSE_PASSWORD",
)
PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "fact_sales": ("order_id",),
    "dim_date": ("date_key",),
    "dim_customer": ("customer_id",),
    "dim_product": ("product",),
    "dim_geography": ("country",),
}
# (table, column, referenced table, referenced column): used for Metabase joins.
FOREIGN_KEYS: tuple[tuple[str, str, str, str], ...] = (
    ("fact_sales", "date_key", "dim_date", "date_key"),
    ("fact_sales", "customer_id", "dim_customer", "customer_id"),
    ("fact_sales", "product", "dim_product", "product"),
    ("fact_sales", "country", "dim_geography", "country"),
)
