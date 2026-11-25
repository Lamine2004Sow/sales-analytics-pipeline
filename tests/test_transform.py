"""Tests for the Polars cleaning, star schema, KPI and export layer."""

from datetime import date

import polars as pl
import pytest

from retail_sales_pipeline.export import write_tables
from retail_sales_pipeline.transform import (
    build_star_schema,
    clean,
    compute_kpis,
    count_customer_attribute_conflicts,
)

HEADER = (
    "order_id,customer_id,transaction_date,country,currency,year,quarter,month,day_of_week,"
    "is_weekend,season,product,industry,channel,customer_segment,payment_method,order_status,"
    "quantity,unit_price_usd,discount_pct,unit_price_after_discount_usd,revenue_usd,revenue_local,"
    "tax_rate_pct,tax_amount_usd,shipping_cost_usd,gross_profit_usd,profit_margin_pct,"
    "recency_days,frequency_orders,monetary_total_usd,avg_order_value_usd,rfm_r_score,"
    "rfm_f_score,rfm_m_score,rfm_total_score,customer_tenure_days,is_repeat_customer,"
    "purchase_frequency_per_year,clv_3yr_usd,churn_probability,is_churned,discount_sensitive,"
    "nps_score,engagement_score,priority_segment,return_flag,support_tickets,days_to_delivery,"
    "gdp_per_capita_usd"
).split(",")

VALID_ROW = (
    "ORD-2024-000001,21330,2024-04-12,Australia,AUD,2024,2,4,4,0,Spring,Product_J,Education,"
    "Direct Sales,SMB,Credit Card,Completed,8,668.04,20.0,534.43,4275.44,6626.93,10.0,427.54,"
    "119.36,1667.42,39.0,163,5,7055.51,1411.1,5,3,5,13,2889,1,0.632,2674.21,0.122,0,1,6.2,45.3,"
    "High Value,0,1,21,55000"
)


def _parse(value: str) -> int | float | str:
    """Convert a CSV cell to int, float or str, in that order."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def make_row(**overrides: object) -> dict[str, object]:
    """Return one raw sales row built from VALID_ROW, with selected fields overridden."""
    row = {col: _parse(val) for col, val in zip(HEADER, VALID_ROW.split(","), strict=True)}
    unknown = set(overrides) - set(row)
    assert not unknown, f"unknown columns: {unknown}"
    return {**row, **overrides}


def make_df(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


@pytest.fixture
def raw() -> pl.DataFrame:
    """Six rows: a duplicate row, a duplicate order_id, a cancelled and a refunded order."""
    return make_df(
        make_row(),  # ORD-000001, customer 21330, Completed, 2024-04-12
        make_row(),  # exact duplicate of the row above
        make_row(  # same customer, later order with a different segment and snapshot
            order_id="ORD-2024-000002",
            transaction_date="2024-06-01",
            customer_segment="Enterprise",
            recency_days=10,
            revenue_usd=1000.0,
            gross_profit_usd=400.0,
            shipping_cost_usd=50.0,
            product="Product_A",
            return_flag=1,
        ),
        make_row(order_id="ORD-2024-000002", revenue_usd=9999.0),  # duplicate order_id, dropped
        make_row(  # cancelled order, whitespace around the country name
            order_id="ORD-2024-000003",
            customer_id=30000,
            transaction_date="2024-06-05",
            country="  Germany ",
            currency="EUR",
            order_status="Cancelled",
            industry="Retail",
            revenue_usd=500.0,
            gross_profit_usd=200.0,
            product="Product_A",
        ),
        make_row(  # refunded order
            order_id="ORD-2024-000004",
            customer_id=40000,
            transaction_date="2024-07-01",
            country="UK",
            currency="GBP",
            order_status="Refunded",
            revenue_usd=300.0,
            product="Product_B",
        ),
        make_row(  # completed order for customer 40000
            order_id="ORD-2024-000005",
            customer_id=40000,
            transaction_date="2024-07-02",
            country="UK",
            currency="GBP",
            revenue_usd=700.0,
            product="Product_B",
        ),
    )


def test_clean_strips_dedups_and_types(raw: pl.DataFrame) -> None:
    cleaned = clean(raw)

    assert cleaned.height == 5
    assert cleaned["order_id"].n_unique() == 5
    assert "Germany" in cleaned["country"].to_list()
    assert cleaned["transaction_date"].dtype == pl.Date
    assert cleaned["is_weekend"].dtype == pl.Boolean
    assert cleaned["return_flag"].dtype == pl.Boolean
    regions = dict(zip(cleaned["country"].to_list(), cleaned["region"].to_list(), strict=False))
    assert regions == {"Australia": "APAC", "Germany": "EMEA", "UK": "EMEA"}
    assert clean(make_df(make_row(country="Narnia")))["region"].to_list() == ["Other"]


def test_clean_derived_columns(raw: pl.DataFrame) -> None:
    row = clean(raw).filter(pl.col("order_id") == "ORD-2024-000001").row(0, named=True)

    assert row["is_realised"] is True
    assert row["net_profit_usd"] == pytest.approx(1667.42 - 119.36, abs=0.005)
    assert row["discount_amount_usd"] == pytest.approx((668.04 - 534.43) * 8, abs=0.005)
    assert row["date_key"] == 20240412


def test_dim_customer_uses_most_recent_snapshot(raw: pl.DataFrame) -> None:
    star = build_star_schema(clean(raw))
    dim = star["dim_customer"]

    assert dim["customer_id"].n_unique() == dim.height == 3
    row = dim.filter(pl.col("customer_id") == 21330).row(0, named=True)
    assert row["recency_days"] == 10
    assert row["customer_segment"] == "Enterprise"
    assert row["first_order_date"] == date(2024, 4, 12)
    assert row["last_order_date"] == date(2024, 6, 1)
    assert row["orders_count"] == 2


def test_customer_attribute_conflicts(raw: pl.DataFrame) -> None:
    assert count_customer_attribute_conflicts(clean(raw)) == {
        "customer_segment": 1,
        "industry": 0,
    }


def test_dim_date_is_continuous(raw: pl.DataFrame) -> None:
    cleaned = clean(raw)
    dim = build_star_schema(cleaned)["dim_date"]

    span = (cleaned["transaction_date"].max() - cleaned["transaction_date"].min()).days + 1
    assert dim.height == span
    assert dim["date_key"].is_unique().all()
    day = dim.filter(pl.col("date") == date(2024, 4, 12)).row(0, named=True)
    assert day["day_of_week"] == 4  # Friday, Monday=0
    assert day["day_name"] == "Friday"
    assert day["month_name"] == "Apr"
    assert day["season"] == "Spring"
    assert day["year_month"] == "2024-04"


def test_fact_sales_links_to_dim_date(raw: pl.DataFrame) -> None:
    star = build_star_schema(clean(raw))
    fact, dim_date = star["fact_sales"], star["dim_date"]

    assert fact.height == 5
    assert fact.join(dim_date, on="date_key", how="anti").is_empty()
    assert fact["is_realised"].sum() == 3


def test_kpi_summary_excludes_non_completed(raw: pl.DataFrame) -> None:
    kpis = compute_kpis(clean(raw))
    summary = kpis["kpi_summary"].row(0, named=True)

    # Realised: ORD-000001 (4275.44), ORD-000002 (1000.0), ORD-000005 (700.0).
    assert summary["total_revenue_usd"] == pytest.approx(5975.44)
    assert summary["orders"] == 3
    assert summary["units"] == 24  # quantity 8 on each realised order
    assert summary["customers"] == 2
    assert summary["total_orders_all_statuses"] == 5
    assert summary["cancellation_rate_pct"] == pytest.approx(20.0)
    assert summary["refund_rate_pct"] == pytest.approx(20.0)
    assert summary["return_rate_pct"] == pytest.approx(20.0)  # 1 return over 5 orders


def test_revenue_shares_sum_to_100(raw: pl.DataFrame) -> None:
    kpis = compute_kpis(clean(raw))

    assert kpis["sales_by_product"]["revenue_share_pct"].sum() == pytest.approx(100, abs=0.05)
    assert kpis["sales_by_region_total"]["revenue_share_pct"].sum() == pytest.approx(100, abs=0.05)
    assert kpis["sales_by_product"]["rank_revenue"].to_list() == [1, 2, 3]


def test_monthly_mom_and_yoy_on_two_months() -> None:
    raw = make_df(
        make_row(order_id="A1", transaction_date="2024-04-10", revenue_usd=100.0),
        make_row(order_id="A2", transaction_date="2024-05-20", revenue_usd=150.0),
    )
    monthly = compute_kpis(clean(raw))["sales_monthly"]

    assert monthly["year_month"].to_list() == ["2024-04", "2024-05"]
    assert monthly["revenue_mom_pct"][0] is None
    assert monthly["revenue_mom_pct"][1] == pytest.approx(50.0)
    assert monthly["revenue_yoy_pct"].null_count() == 2
    assert monthly["revenue_rolling_3m"].null_count() == 2


def test_write_tables(tmp_path, raw: pl.DataFrame) -> None:
    tables = {
        "fact_sales": build_star_schema(clean(raw))["fact_sales"],
        "tiny": pl.DataFrame({"a": [1]}),
    }

    paths = write_tables(tables, tmp_path / "out")

    assert [p.name for p in paths] == ["fact_sales.csv", "tiny.csv"]
    assert all(p.exists() for p in paths)
    assert paths[1].read_text().splitlines() == ["a", "1"]
    assert "order_id" in paths[0].read_text().splitlines()[0]
