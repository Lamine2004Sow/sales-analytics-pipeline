"""Pydantic v2 schema for one raw sales row, with business-coherence rules."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from retail_sales_pipeline.config import (
    AMOUNT_ABS_TOL,
    AMOUNT_REL_TOL,
    COUNTRY_TO_CURRENCY,
    MONTH_TO_SEASON,
)

Country = Literal[
    "Australia", "Brazil", "Canada", "China", "Germany", "India", "Japan", "UAE", "UK", "USA"
]
Currency = Literal["AED", "AUD", "BRL", "CAD", "CNY", "EUR", "GBP", "INR", "JPY", "USD"]
Season = Literal["Winter", "Spring", "Summer", "Fall"]
Product = Literal[
    "Product_A",
    "Product_B",
    "Product_C",
    "Product_D",
    "Product_E",
    "Product_F",
    "Product_G",
    "Product_H",
    "Product_I",
    "Product_J",
]
Industry = Literal[
    "Education",
    "Energy",
    "Finance",
    "Healthcare",
    "Logistics",
    "Manufacturing",
    "Media",
    "Real Estate",
    "Retail",
    "Technology",
]
Channel = Literal["Direct Sales", "Marketplace", "Online", "Partner", "Reseller"]
CustomerSegment = Literal["Enterprise", "Government", "Non-Profit", "SMB", "Startup"]
PaymentMethod = Literal["Bank Transfer", "Credit Card", "Crypto", "Invoice", "PayPal"]
OrderStatus = Literal["Cancelled", "Completed", "Pending", "Refunded"]
PrioritySegment = Literal["High Value", "Low", "Medium"]
Flag = Literal[0, 1]

# Tolerance for profit_margin_pct, which is a percentage point value.
MARGIN_ABS_TOL = 0.1


def _close(a: float, b: float) -> bool:
    """Return True if a and b match within the configured amount tolerance."""
    return abs(a - b) <= max(AMOUNT_ABS_TOL, AMOUNT_REL_TOL * abs(b))


class SalesRecord(BaseModel):
    """One row of the raw sales CSV, validated field by field then rule by rule."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    order_id: str = Field(pattern=r"^ORD-\d{4}-\d{6}$")
    customer_id: int = Field(gt=0)
    transaction_date: date
    country: Country
    currency: Currency
    year: int = Field(ge=2000, le=2100)
    quarter: int = Field(ge=1, le=4)
    month: int = Field(ge=1, le=12)
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: Flag
    season: Season
    product: Product
    industry: Industry
    channel: Channel
    customer_segment: CustomerSegment
    payment_method: PaymentMethod
    order_status: OrderStatus
    quantity: int = Field(ge=1)
    unit_price_usd: float = Field(gt=0)
    discount_pct: float = Field(ge=0, le=100)
    unit_price_after_discount_usd: float = Field(ge=0)
    revenue_usd: float = Field(gt=0)
    revenue_local: float = Field(gt=0)
    tax_rate_pct: float = Field(ge=0, le=100)
    tax_amount_usd: float = Field(ge=0)
    shipping_cost_usd: float = Field(ge=0)
    gross_profit_usd: float
    profit_margin_pct: float = Field(ge=-100, le=100)
    recency_days: int = Field(ge=0)
    frequency_orders: int = Field(ge=1)
    monetary_total_usd: float = Field(ge=0)
    avg_order_value_usd: float = Field(ge=0)
    rfm_r_score: int = Field(ge=1, le=5)
    rfm_f_score: int = Field(ge=1, le=5)
    rfm_m_score: int = Field(ge=1, le=5)
    rfm_total_score: int = Field(ge=3, le=15)
    customer_tenure_days: int = Field(ge=0)
    is_repeat_customer: Flag
    purchase_frequency_per_year: float = Field(ge=0)
    clv_3yr_usd: float = Field(ge=0)
    churn_probability: float = Field(ge=0, le=1)
    is_churned: Flag
    discount_sensitive: Flag
    nps_score: float = Field(ge=0, le=10)
    engagement_score: float = Field(ge=0, le=100)
    priority_segment: PrioritySegment
    return_flag: Flag
    support_tickets: int = Field(ge=0)
    days_to_delivery: int = Field(ge=0)
    gdp_per_capita_usd: int = Field(ge=0)

    @model_validator(mode="after")
    def check_business_rules(self) -> "SalesRecord":
        """Check cross-field coherence. All failed rules are reported together."""
        errors: list[str] = []

        # NOTE: the year inside order_id is NOT checked against `year`: in the data,
        # Dec 29-31 rows carry the ISO-week year (next year).
        weekday = self.transaction_date.weekday()
        if (
            self.year != self.transaction_date.year
            or self.quarter != (self.transaction_date.month - 1) // 3 + 1
            or self.month != self.transaction_date.month
            or self.day_of_week != weekday
            or self.is_weekend != int(self.day_of_week >= 5)
        ):
            errors.append(
                "date_mismatch: year/quarter/month/day_of_week/is_weekend "
                "do not match transaction_date"
            )

        if self.season != MONTH_TO_SEASON[self.month]:
            errors.append(f"season_mismatch: {self.season} != {MONTH_TO_SEASON[self.month]}")

        expected_currency = COUNTRY_TO_CURRENCY[self.country]
        if self.currency != expected_currency:
            errors.append(f"currency_mismatch: {self.currency} != {expected_currency}")

        expected_after_discount = self.unit_price_usd * (1 - self.discount_pct / 100)
        if not _close(self.unit_price_after_discount_usd, expected_after_discount):
            errors.append(
                "discount_inconsistent: unit_price_after_discount_usd "
                f"{self.unit_price_after_discount_usd} != {expected_after_discount:.2f}"
            )

        expected_revenue = self.quantity * self.unit_price_after_discount_usd
        if not _close(self.revenue_usd, expected_revenue):
            errors.append(
                f"revenue_inconsistent: revenue_usd {self.revenue_usd} != {expected_revenue:.2f}"
            )

        expected_tax = self.revenue_usd * self.tax_rate_pct / 100
        if not _close(self.tax_amount_usd, expected_tax):
            errors.append(
                f"tax_inconsistent: tax_amount_usd {self.tax_amount_usd} != {expected_tax:.2f}"
            )

        expected_margin = self.gross_profit_usd / self.revenue_usd * 100
        if abs(self.profit_margin_pct - expected_margin) > MARGIN_ABS_TOL:
            errors.append(
                "margin_inconsistent: profit_margin_pct "
                f"{self.profit_margin_pct} != {expected_margin:.2f}"
            )

        if self.gross_profit_usd > self.revenue_usd:
            errors.append(
                f"profit_exceeds_revenue: gross_profit_usd {self.gross_profit_usd} > "
                f"revenue_usd {self.revenue_usd}"
            )

        if errors:
            raise ValueError("; ".join(errors))
        return self
