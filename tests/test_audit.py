"""Tests for the data-audit functions (forecastability, backtests, leakage, fingerprints)."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from retail_sales_pipeline.audit import (
    _folds,
    backtest_forecasters,
    derived_column_check,
    forecastability,
    leakage_demo,
    monthly_series,
    summarise_backtest,
    synthetic_fingerprints,
)
from retail_sales_pipeline.transform import clean

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

TEMPLATE = (
    "ORD-2024-000001,21330,2024-04-12,Australia,AUD,2024,2,4,4,0,Spring,Product_J,Education,"
    "Direct Sales,SMB,Credit Card,Completed,8,668.04,20.0,534.43,4275.44,6626.93,10.0,427.54,"
    "119.36,1667.42,39.0,163,5,7055.51,1411.1,5,3,5,13,2889,1,0.632,2674.21,0.122,0,1,6.2,45.3,"
    "High Value,0,1,21,55000"
).split(",")


def _parse(value: str) -> int | float | str:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


TEMPLATE_ROW = {col: _parse(val) for col, val in zip(HEADER, TEMPLATE, strict=True)}


def raw_orders(n: int, seed: int = 0) -> pl.DataFrame:
    """Raw-shaped sales rows where revenue_usd = quantity x price after discount, as in the data."""
    rng = np.random.default_rng(seed)
    start = date(2021, 1, 1)
    rows = []
    for i in range(n):
        unit = round(float(rng.uniform(10, 100)), 2)
        disc = float(rng.choice([0.0, 5.0, 10.0, 20.0]))
        after = round(unit * (1 - disc / 100), 2)
        qty = int(rng.integers(1, 10))
        day = start + timedelta(days=int(rng.integers(0, 1400)))
        rows.append(
            {
                **TEMPLATE_ROW,
                "order_id": f"ORD-{i:06d}",
                "customer_id": int(rng.integers(1, 200)),
                "transaction_date": day.isoformat(),
                "product": f"Product_{rng.integers(0, 4)}",
                "country": str(rng.choice(["USA", "UK", "Japan"])),
                "order_status": "Completed" if rng.random() > 0.1 else "Cancelled",
                "quantity": qty,
                "unit_price_usd": unit,
                "discount_pct": disc,
                "unit_price_after_discount_usd": after,
                "revenue_usd": round(qty * after, 2),
                "churn_probability": round(float(rng.uniform(0, 1)), 3),
                "is_churned": int(rng.random() > 0.5),
            }
        )
    return pl.DataFrame(rows)


def monthly_sine(n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, n)


def test_folds_never_use_future() -> None:
    for train_idx, test_idx in _folds(120, n_splits=5, test_size=12):
        assert train_idx.max() < test_idx.min()
        assert len(test_idx) == 12
        assert test_idx.max() < 120


@pytest.fixture(scope="module")
def noise_backtest() -> pl.DataFrame:
    y = 100 + np.random.default_rng(0).normal(0, 5, 120)
    return summarise_backtest(backtest_forecasters(y, n_splits=2))


@pytest.fixture(scope="module")
def sine_backtest() -> pl.DataFrame:
    return summarise_backtest(backtest_forecasters(monthly_sine(), n_splits=2))


def test_white_noise_is_flagged_and_seasonal_naive_does_not_beat_mean(
    noise_backtest: pl.DataFrame,
) -> None:
    y = 100 + np.random.default_rng(0).normal(0, 5, 120)
    result = forecastability(y)
    assert result["is_white_noise"] is True
    assert set(result) >= {"cv", "acf", "ljung_box_p_12", "seasonal_strength", "trend_strength"}

    summary = noise_backtest
    skill = summary.filter(summary["model"] == "seasonal_naive")["skill_vs_mean_pct"].item()
    assert skill < 10


def test_seasonal_trending_series_is_predictable(sine_backtest: pl.DataFrame) -> None:
    result = forecastability(monthly_sine())
    assert result["is_white_noise"] is False
    assert result["seasonal_strength"] > 0.9

    summary = sine_backtest
    skills = dict(zip(summary["model"], summary["skill_vs_mean_pct"], strict=True))
    assert skills["seasonal_naive"] > 30
    assert skills["holt_winters"] > 30


def test_backtest_summary_has_mean_row_with_zero_skill(sine_backtest: pl.DataFrame) -> None:
    summary = sine_backtest
    assert summary["mae_mean"].is_sorted(nulls_last=True)
    assert summary.filter(summary["model"] == "mean")["skill_vs_mean_pct"].item() == pytest.approx(
        0
    )


def test_monthly_series_fills_gaps_with_zero() -> None:
    clean_df = clean(raw_orders(300)).filter(pl.col("transaction_date") >= date(2021, 1, 1))
    series = monthly_series(clean_df)
    months = series["month_start"]
    assert months.dtype == pl.Date
    assert (months.dt.day() == 1).all()
    gaps = months.diff().drop_nulls().unique().to_list()
    assert all(28 <= g.days <= 31 for g in gaps)  # consecutive month starts, none skipped
    assert series["revenue_usd"].null_count() == 0
    assert (series["revenue_usd"] >= 0).all()


def test_leaky_features_explain_revenue_and_honest_ones_do_not() -> None:
    clean_df = clean(raw_orders(600))
    result = leakage_demo(clean_df)
    r2 = {(row["feature_set"], row["model"]): row["r2"] for row in result.iter_rows(named=True)}
    assert r2[("leaky", "hist_gbr")] > 0.95
    assert r2[("honest", "ridge")] < r2[("leaky", "hist_gbr")]
    assert result.filter(pl.col("feature_set") == "none")["model"].item() == "mean"
    assert (result["n_train"] > 0).all() and (result["n_test"] > 0).all()


def test_fingerprints_keys_and_identity_on_synthetic_data() -> None:
    clean_df = clean(raw_orders(500))
    fp = synthetic_fingerprints(clean_df)
    expected = {
        "null_count_total",
        "duplicate_rows",
        "revenue_identity_share",
        "product_margin_sd_pp",
        "month_of_year_revenue_cv",
        "customers_multi_order",
        "customers_segment_conflict",
        "customers_churn_prob_conflict",
        "churn_prob_vs_is_churned_corr",
        "verdict",
    }
    assert expected <= set(fp)
    assert fp["revenue_identity_share"] == pytest.approx(1.0)
    assert isinstance(fp["verdict"], str)


def test_fingerprints_accept_raw_frames() -> None:
    fp = synthetic_fingerprints(raw_orders(200))
    assert fp["n_rows"] == 200


def test_derived_column_check_returns_requested_columns() -> None:
    clean_df = clean(raw_orders(400))
    result = derived_column_check(clean_df, columns=["frequency_orders", "rfm_m_score"])
    assert result["column"].to_list() == ["frequency_orders", "rfm_m_score"]
    assert result["customer_actual"].to_list() == ["order_count", "sum_realised_revenue_usd"]
