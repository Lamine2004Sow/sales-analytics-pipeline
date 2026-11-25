"""Tests for the Pydantic validation layer (schemas.py and validate.py)."""

import io
import json

import polars as pl
import pytest
from pydantic import ValidationError

from retail_sales_pipeline.schemas import SalesRecord
from retail_sales_pipeline.validate import validate_dataframe, write_validation_outputs

HEADER = (
    "order_id,customer_id,transaction_date,country,currency,year,quarter,month,day_of_week,"
    "is_weekend,season,product,industry,channel,customer_segment,payment_method,order_status,"
    "quantity,unit_price_usd,discount_pct,unit_price_after_discount_usd,revenue_usd,revenue_local,"
    "tax_rate_pct,tax_amount_usd,shipping_cost_usd,gross_profit_usd,profit_margin_pct,recency_days,"
    "frequency_orders,monetary_total_usd,avg_order_value_usd,rfm_r_score,rfm_f_score,rfm_m_score,"
    "rfm_total_score,customer_tenure_days,is_repeat_customer,purchase_frequency_per_year,"
    "clv_3yr_usd,churn_probability,is_churned,discount_sensitive,nps_score,engagement_score,"
    "priority_segment,return_flag,support_tickets,days_to_delivery,gdp_per_capita_usd"
)
LINE = (
    "ORD-2024-000001,21330,2024-04-12,Australia,AUD,2024,2,4,4,0,Spring,Product_J,Education,"
    "Direct Sales,SMB,Credit Card,Completed,8,668.04,20.0,534.43,4275.44,6626.93,10.0,427.54,"
    "119.36,1667.42,39.0,163,5,7055.51,1411.1,5,3,5,13,2889,1,0.632,2674.21,0.122,0,1,6.2,45.3,"
    "High Value,0,1,21,55000"
)


def _typed_row(line: str) -> dict:
    """Parse a CSV line into a typed row, exactly as read_csv would produce it."""
    df = pl.read_csv(io.StringIO(HEADER + "\n" + line), infer_schema_length=None)
    return df.row(0, named=True)


@pytest.fixture
def valid_row() -> dict:
    return _typed_row(LINE)


def test_valid_row_passes(valid_row):
    SalesRecord.model_validate(valid_row)


@pytest.mark.parametrize(
    ("field", "value", "rule"),
    [
        ("day_of_week", 3, "date_mismatch"),
        ("season", "Winter", "season_mismatch"),
        ("currency", "USD", "currency_mismatch"),
        ("unit_price_after_discount_usd", 600.0, "discount_inconsistent"),
        ("revenue_usd", 5000.0, "revenue_inconsistent"),
        ("tax_amount_usd", 999.0, "tax_inconsistent"),
        ("profit_margin_pct", 50.0, "margin_inconsistent"),
        ("gross_profit_usd", 7000.0, "profit_exceeds_revenue"),
    ],
)
def test_each_business_rule_is_reported(valid_row, field, value, rule):
    valid_row[field] = value
    with pytest.raises(ValidationError) as excinfo:
        SalesRecord.model_validate(valid_row)
    assert rule in str(excinfo.value)


def test_all_failed_rules_are_collected(valid_row):
    valid_row["season"] = "Winter"
    valid_row["currency"] = "USD"
    with pytest.raises(ValidationError) as excinfo:
        SalesRecord.model_validate(valid_row)
    message = str(excinfo.value)
    assert "season_mismatch" in message
    assert "currency_mismatch" in message


def test_bad_literal_is_rejected(valid_row):
    valid_row["country"] = "France"
    with pytest.raises(ValidationError) as excinfo:
        SalesRecord.model_validate(valid_row)
    assert any(err["loc"] == ("country",) for err in excinfo.value.errors())


def test_negative_quantity_is_rejected(valid_row):
    valid_row["quantity"] = -1
    with pytest.raises(ValidationError) as excinfo:
        SalesRecord.model_validate(valid_row)
    assert any(err["loc"] == ("quantity",) for err in excinfo.value.errors())


def test_flag_must_be_zero_or_one(valid_row):
    valid_row["is_churned"] = 2
    with pytest.raises(ValidationError):
        SalesRecord.model_validate(valid_row)


def test_missing_field_is_rejected(valid_row):
    del valid_row["gdp_per_capita_usd"]
    with pytest.raises(ValidationError) as excinfo:
        SalesRecord.model_validate(valid_row)
    assert any(err["type"] == "missing" for err in excinfo.value.errors())


def test_extra_field_is_forbidden(valid_row):
    valid_row["unexpected_column"] = 1
    with pytest.raises(ValidationError) as excinfo:
        SalesRecord.model_validate(valid_row)
    assert any(err["type"] == "extra_forbidden" for err in excinfo.value.errors())


def test_validate_dataframe_counts_and_keeps_schema(valid_row):
    other = dict(valid_row, order_id="ORD-2024-000002")
    duplicate = dict(valid_row)  # same order_id as valid_row
    bad_currency = dict(valid_row, order_id="ORD-2024-000003", currency="USD")
    df = pl.DataFrame([valid_row, other, duplicate, bad_currency])

    result = validate_dataframe(df)

    assert result.report["total_rows"] == 4
    assert result.report["valid_rows"] == 2
    assert result.report["rejected_rows"] == 2
    assert result.report["valid_pct"] == 50.0
    assert result.report["errors_by_rule"]["duplicate_order_id"] == 1
    assert result.report["errors_by_rule"]["currency_mismatch"] == 1
    assert result.valid.columns == df.columns
    assert result.valid.schema == df.schema
    assert result.valid["order_id"].to_list() == ["ORD-2024-000001", "ORD-2024-000002"]
    assert sorted(result.rejected["row_number"].to_list()) == [3, 4]
    assert "duplicate_order_id" in result.rejected.filter(pl.col("row_number") == 3)["errors"][0]


def test_write_validation_outputs(tmp_path, valid_row):
    df = pl.DataFrame([valid_row, dict(valid_row, currency="USD")])
    result = validate_dataframe(df)

    write_validation_outputs(result, reports_dir=tmp_path)

    report = json.loads((tmp_path / "quality_report.json").read_text(encoding="utf-8"))
    assert report["rejected_rows"] == 1
    csv_text = (tmp_path / "rejected_rows.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("order_id,customer_id,")
    assert "row_number" in csv_text.splitlines()[0]


def test_write_validation_outputs_empty_rejected_has_header(tmp_path, valid_row):
    result = validate_dataframe(pl.DataFrame([valid_row]))

    write_validation_outputs(result, reports_dir=tmp_path)

    lines = (tmp_path / "rejected_rows.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("order_id,")
