"""Polars transformation: clean raw sales, build the star schema and the KPI tables.

Run as a script to rebuild the processed star schema and the KPI marts from the raw CSV.
"""

import time

import polars as pl

from retail_sales_pipeline.config import (
    COUNTRY_TO_REGION,
    MARTS_DIR,
    MONTH_TO_SEASON,
    PROCESSED_DIR,
    RAW_CSV,
    REALISED_STATUS,
)
from retail_sales_pipeline.export import write_tables

FLAG_COLUMNS = [
    "is_weekend",
    "is_repeat_customer",
    "is_churned",
    "discount_sensitive",
    "return_flag",
]
CANCELLED_STATUS = "Cancelled"
REFUNDED_STATUS = "Refunded"
CONFLICT_ATTRIBUTES = ["customer_segment", "industry"]

# Customer-level attributes: snapshot values repeated on every order, taken from the latest order.
CUSTOMER_SNAPSHOT_COLUMNS = [
    "customer_segment",
    "industry",
    "priority_segment",
    "recency_days",
    "frequency_orders",
    "monetary_total_usd",
    "avg_order_value_usd",
    "rfm_r_score",
    "rfm_f_score",
    "rfm_m_score",
    "rfm_total_score",
    "customer_tenure_days",
    "is_repeat_customer",
    "purchase_frequency_per_year",
    "clv_3yr_usd",
    "churn_probability",
    "is_churned",
    "discount_sensitive",
    "nps_score",
    "engagement_score",
]

FACT_COLUMNS = [
    "order_id",
    "date_key",
    "customer_id",
    "customer_segment",
    "industry",
    "product",
    "country",
    "channel",
    "payment_method",
    "order_status",
    "is_realised",
    "quantity",
    "unit_price_usd",
    "discount_pct",
    "discount_amount_usd",
    "revenue_usd",
    "revenue_local",
    "currency",
    "tax_amount_usd",
    "shipping_cost_usd",
    "gross_profit_usd",
    "net_profit_usd",
    "profit_margin_pct",
    "return_flag",
    "support_tickets",
    "days_to_delivery",
]

DATE_COLUMNS = [
    "date_key",
    "date",
    "year",
    "quarter",
    "month",
    "month_name",
    "year_month",
    "week",
    "day_of_week",
    "day_name",
    "is_weekend",
    "season",
]

PRODUCT_KPI_COLUMNS = [
    "product",
    "revenue_usd",
    "gross_profit_usd",
    "net_profit_usd",
    "units",
    "orders",
    "gross_margin_pct",
    "revenue_share_pct",
    "rank_revenue",
    "rank_profit",
    "rank_units",
]
REGION_KPI_COLUMNS = [
    "region",
    "country",
    "revenue_usd",
    "gross_profit_usd",
    "orders",
    "customers",
    "gross_margin_pct",
    "revenue_share_pct",
]
REGION_TOTAL_KPI_COLUMNS = ["region", *REGION_KPI_COLUMNS[2:]]
MONTHLY_KPI_COLUMNS = [
    "year_month",
    "year",
    "month",
    "revenue_usd",
    "gross_profit_usd",
    "orders",
    "units",
    "revenue_mom_pct",
    "revenue_yoy_pct",
    "revenue_rolling_3m",
]
YEARLY_KPI_COLUMNS = [
    "year",
    "revenue_usd",
    "gross_profit_usd",
    "orders",
    "customers",
    "revenue_yoy_pct",
]
SEGMENT_KPI_COLUMNS = [
    "customer_segment",
    "revenue_usd",
    "gross_profit_usd",
    "orders",
    "customers",
    "avg_order_value_usd",
    "revenue_per_customer_usd",
    "gross_margin_pct",
    "avg_nps",
    "avg_churn_probability",
    "revenue_share_pct",
]
CHANNEL_KPI_COLUMNS = [
    "channel",
    "revenue_usd",
    "gross_profit_usd",
    "orders",
    "gross_margin_pct",
    "revenue_share_pct",
]


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """Return a typed, deduplicated copy of the raw sales rows with derived columns."""
    return (
        df.with_columns(pl.col(pl.String).str.strip_chars())
        .unique(maintain_order=True)
        .unique(subset=["order_id"], keep="first", maintain_order=True)
        .with_columns(
            pl.col("transaction_date").str.to_date("%Y-%m-%d"),
            pl.col(FLAG_COLUMNS).cast(pl.Boolean),
            region=pl.col("country").replace_strict(COUNTRY_TO_REGION, default="Other"),
            is_realised=pl.col("order_status") == REALISED_STATUS,
            net_profit_usd=(pl.col("gross_profit_usd") - pl.col("shipping_cost_usd")).round(2),
            discount_amount_usd=(
                (pl.col("unit_price_usd") - pl.col("unit_price_after_discount_usd"))
                * pl.col("quantity")
            ).round(2),
        )
        .with_columns(date_key=pl.col("transaction_date").dt.strftime("%Y%m%d").cast(pl.Int32))
    )


def count_customer_attribute_conflicts(clean_df: pl.DataFrame) -> dict[str, int]:
    """Count customers with more than one distinct value for each customer attribute."""
    per_customer = clean_df.group_by("customer_id").agg(
        pl.col(attr).n_unique().alias(attr) for attr in CONFLICT_ATTRIBUTES
    )
    return {attr: int((per_customer[attr] > 1).sum()) for attr in CONFLICT_ATTRIBUTES}


def _mode_per_key(df: pl.DataFrame, key: str, value: str) -> pl.DataFrame:
    """Most frequent value of `value` per `key`; ties go to the value seen first."""
    return (
        df.group_by(key, value)
        .agg(n=pl.len(), first_seen=pl.col("transaction_date").min())
        .sort(["n", "first_seen"], descending=[True, False])
        .unique(subset=key, keep="first")
        .select(key, value)
    )


def _dim_date(df: pl.DataFrame) -> pl.DataFrame:
    """Continuous calendar covering the whole transaction period."""
    start, end = df["transaction_date"].min(), df["transaction_date"].max()
    return (
        pl.DataFrame({"date": pl.date_range(start, end, interval="1d", eager=True)})
        .with_columns(
            date_key=pl.col("date").dt.strftime("%Y%m%d").cast(pl.Int32),
            year=pl.col("date").dt.year(),
            quarter=pl.col("date").dt.quarter(),
            month=pl.col("date").dt.month(),
            month_name=pl.col("date").dt.strftime("%b"),
            year_month=pl.col("date").dt.strftime("%Y-%m"),
            week=pl.col("date").dt.week(),
            day_of_week=pl.col("date").dt.weekday() - 1,  # Polars: Mon=1 -> 0=Mon
            day_name=pl.col("date").dt.strftime("%A"),
        )
        .with_columns(
            is_weekend=pl.col("day_of_week") >= 5,
            season=pl.col("month").cast(pl.Int64).replace_strict(MONTH_TO_SEASON),
        )
        .select(DATE_COLUMNS)
    )


def _dim_product(df: pl.DataFrame) -> pl.DataFrame:
    """Distinct products with a 1-based surrogate key."""
    return (
        df.select(pl.col("product").unique().sort())
        .with_columns(product_key=pl.int_range(1, pl.len() + 1, dtype=pl.Int32))
        .select("product_key", "product")
    )


def _dim_customer(df: pl.DataFrame) -> pl.DataFrame:
    """One row per customer: latest-order snapshot plus order history."""
    latest = (
        df.sort(["transaction_date", "order_id"])
        .unique(subset="customer_id", keep="last", maintain_order=True)
        .select("customer_id", *CUSTOMER_SNAPSHOT_COLUMNS)
    )
    history = df.group_by("customer_id").agg(
        first_order_date=pl.col("transaction_date").min(),
        last_order_date=pl.col("transaction_date").max(),
        orders_count=pl.len(),  # all statuses
    )
    return latest.join(history, on="customer_id", how="left", validate="1:1").sort("customer_id")


def _dim_geography(df: pl.DataFrame) -> pl.DataFrame:
    """One row per country with its region, currency and (modal) GDP per capita."""
    gdp = _mode_per_key(df, "country", "gdp_per_capita_usd")
    currency = _mode_per_key(df, "country", "currency")
    return (
        gdp.join(currency, on="country", validate="1:1")
        .with_columns(region=pl.col("country").replace_strict(COUNTRY_TO_REGION, default="Other"))
        .select("country", "region", "currency", "gdp_per_capita_usd")
        .sort("country")
    )


def build_star_schema(clean_df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split cleaned sales rows into one fact table and its dimensions."""
    return {
        "fact_sales": clean_df.select(FACT_COLUMNS),
        "dim_date": _dim_date(clean_df),
        "dim_product": _dim_product(clean_df),
        "dim_customer": _dim_customer(clean_df),
        "dim_geography": _dim_geography(clean_df),
    }


def _tidy(df: pl.DataFrame) -> pl.DataFrame:
    """Round every float column to 2 decimals."""
    return df.with_columns(pl.col(pl.Float64).round(2))


def _status_share_pct(clean_df: pl.DataFrame, status: str) -> float:
    return (clean_df["order_status"] == status).sum() / clean_df.height * 100


def _kpi_summary(clean_df: pl.DataFrame, realised: pl.DataFrame) -> pl.DataFrame:
    """Single-row headline KPIs (realised = Completed orders unless stated)."""
    revenue = realised["revenue_usd"].sum()
    gross = realised["gross_profit_usd"].sum()
    orders = realised.height
    return pl.DataFrame(
        {
            "total_revenue_usd": [revenue],
            "total_gross_profit_usd": [gross],
            "total_net_profit_usd": [realised["net_profit_usd"].sum()],
            "gross_margin_pct": [gross / revenue * 100 if revenue else 0.0],
            "orders": [orders],
            "units": [int(realised["quantity"].sum())],
            "customers": [realised["customer_id"].n_unique()],
            "avg_order_value_usd": [revenue / orders if orders else 0.0],
            "total_orders_all_statuses": [clean_df.height],
            "cancellation_rate_pct": [_status_share_pct(clean_df, CANCELLED_STATUS)],
            "refund_rate_pct": [_status_share_pct(clean_df, REFUNDED_STATUS)],
            # Returns are only flagged on Refunded orders, so the rate is over all orders.
            "return_rate_pct": [clean_df["return_flag"].mean() * 100 if clean_df.height else 0.0],
            "first_date": [realised["transaction_date"].min()],
            "last_date": [realised["transaction_date"].max()],
        }
    )


def _grouped_sales(realised: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Common sales metrics for realised orders grouped by `by`."""
    total_revenue = realised["revenue_usd"].sum()
    return (
        realised.group_by(by)
        .agg(
            pl.col("revenue_usd").sum(),
            pl.col("gross_profit_usd").sum(),
            pl.col("net_profit_usd").sum(),
            pl.col("quantity").sum().alias("units"),
            pl.len().alias("orders"),
            pl.col("customer_id").n_unique().alias("customers"),
            pl.col("nps_score").mean().alias("avg_nps"),
            pl.col("churn_probability").mean().alias("avg_churn_probability"),
        )
        .with_columns(
            gross_margin_pct=pl.col("gross_profit_usd") / pl.col("revenue_usd") * 100,
            revenue_share_pct=pl.col("revenue_usd") / total_revenue * 100,
            avg_order_value_usd=pl.col("revenue_usd") / pl.col("orders"),
            revenue_per_customer_usd=pl.col("revenue_usd") / pl.col("customers"),
        )
    )


def _sales_by_product(realised: pl.DataFrame) -> pl.DataFrame:
    return (
        _grouped_sales(realised, ["product"])
        .with_columns(
            rank_revenue=pl.col("revenue_usd").rank(method="min", descending=True),
            rank_profit=pl.col("gross_profit_usd").rank(method="min", descending=True),
            rank_units=pl.col("units").rank(method="min", descending=True),
        )
        .sort("revenue_usd", descending=True)
        .select(PRODUCT_KPI_COLUMNS)
    )


def _sales_by_region(realised: pl.DataFrame) -> pl.DataFrame:
    return (
        _grouped_sales(realised, ["region", "country"])
        .sort(["region", "revenue_usd"], descending=[False, True])
        .select(REGION_KPI_COLUMNS)
    )


def _sales_by_region_total(realised: pl.DataFrame) -> pl.DataFrame:
    return (
        _grouped_sales(realised, ["region"])
        .sort("revenue_usd", descending=True)
        .select(REGION_TOTAL_KPI_COLUMNS)
    )


def _growth_pct(periods: int) -> pl.Expr:
    """Percent change of revenue_usd versus `periods` rows earlier (null if base is 0)."""
    prev = pl.col("revenue_usd").shift(periods)
    return pl.when(prev != 0).then((pl.col("revenue_usd") / prev - 1) * 100).otherwise(None)


def _sales_monthly(realised: pl.DataFrame) -> pl.DataFrame:
    """Monthly sales on a continuous month spine (empty months count as zero)."""
    monthly = realised.group_by(
        pl.col("transaction_date").dt.month_start().alias("month_start")
    ).agg(
        pl.col("revenue_usd").sum(),
        pl.col("gross_profit_usd").sum(),
        pl.len().alias("orders"),
        pl.col("quantity").sum().alias("units"),
    )
    spine = pl.DataFrame(
        {
            "month_start": pl.date_range(
                monthly["month_start"].min(),
                monthly["month_start"].max(),
                interval="1mo",
                eager=True,
            )
        }
    )
    return (
        spine.join(monthly, on="month_start", how="left")
        .with_columns(pl.col("revenue_usd", "gross_profit_usd", "orders", "units").fill_null(0))
        .sort("month_start")
        .with_columns(
            year_month=pl.col("month_start").dt.strftime("%Y-%m"),
            year=pl.col("month_start").dt.year(),
            month=pl.col("month_start").dt.month(),
            revenue_mom_pct=_growth_pct(1),
            revenue_yoy_pct=_growth_pct(12),
            revenue_rolling_3m=pl.col("revenue_usd").rolling_sum(window_size=3, min_samples=3),
        )
        .select(MONTHLY_KPI_COLUMNS)
    )


def _sales_yearly(realised: pl.DataFrame) -> pl.DataFrame:
    return (
        realised.group_by(pl.col("transaction_date").dt.year().alias("year"))
        .agg(
            pl.col("revenue_usd").sum(),
            pl.col("gross_profit_usd").sum(),
            pl.len().alias("orders"),
            pl.col("customer_id").n_unique().alias("customers"),
        )
        .sort("year")
        .with_columns(revenue_yoy_pct=_growth_pct(1))
        .select(YEARLY_KPI_COLUMNS)
    )


def _sales_by_segment(realised: pl.DataFrame) -> pl.DataFrame:
    return (
        _grouped_sales(realised, ["customer_segment"])
        .sort("revenue_usd", descending=True)
        .select(SEGMENT_KPI_COLUMNS)
    )


def _sales_by_channel(realised: pl.DataFrame) -> pl.DataFrame:
    return (
        _grouped_sales(realised, ["channel"])
        .sort("revenue_usd", descending=True)
        .select(CHANNEL_KPI_COLUMNS)
    )


def _order_status_breakdown(clean_df: pl.DataFrame) -> pl.DataFrame:
    """Order counts and revenue per status, across all orders (not only realised)."""
    return (
        clean_df.group_by("order_status")
        .agg(orders=pl.len(), revenue_usd=pl.col("revenue_usd").sum())
        .with_columns(share_pct=pl.col("orders") / clean_df.height * 100)
        .sort("orders", descending=True)
    )


def compute_kpis(clean_df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Compute KPI marts from cleaned sales rows."""
    realised = clean_df.filter(pl.col("is_realised"))
    tables = {
        "kpi_summary": _kpi_summary(clean_df, realised),
        "sales_by_product": _sales_by_product(realised),
        "sales_by_region": _sales_by_region(realised),
        "sales_by_region_total": _sales_by_region_total(realised),
        "sales_monthly": _sales_monthly(realised),
        "sales_yearly": _sales_yearly(realised),
        "sales_by_segment": _sales_by_segment(realised),
        "sales_by_channel": _sales_by_channel(realised),
        "order_status_breakdown": _order_status_breakdown(clean_df),
    }
    return {name: _tidy(table) for name, table in tables.items()}


def main() -> None:
    """Rebuild the processed star schema and the KPI marts from the raw CSV."""
    t0 = time.perf_counter()
    raw = pl.read_csv(RAW_CSV, infer_schema_length=100_000)
    cleaned = clean(raw)
    t1 = time.perf_counter()

    star_paths = write_tables(build_star_schema(cleaned), PROCESSED_DIR)
    t2 = time.perf_counter()

    kpis = compute_kpis(cleaned)
    kpi_paths = write_tables(kpis, MARTS_DIR)
    t3 = time.perf_counter()

    print(f"Rows: raw={raw.height} clean={cleaned.height}")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(kpis["kpi_summary"])
    print(f"Customer attribute conflicts: {count_customer_attribute_conflicts(cleaned)}")
    print(f"Star schema: {len(star_paths)} tables -> {PROCESSED_DIR}")
    print(f"KPI marts: {len(kpi_paths)} tables -> {MARTS_DIR}")
    print(
        f"Timings: read+clean={t1 - t0:.2f}s star={t2 - t1:.2f}s "
        f"kpis={t3 - t2:.2f}s total={t3 - t0:.2f}s"
    )


if __name__ == "__main__":
    main()
