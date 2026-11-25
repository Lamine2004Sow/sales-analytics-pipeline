"""Provision Metabase: admin user, warehouse connection, SQL cards and the dashboard.

Idempotent: running it again reuses the database connection and the collection, recreates the
cards with the same names and replaces the dashboard layout. Requires the stack from `make up`
and the warehouse loaded with `make load`. Reads WAREHOUSE_* and MB_* from the environment or .env.

Usage: uv run python scripts/metabase_setup.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests

from retail_sales_pipeline.config import ENV_FILE, MARTS_DIR, PROCESSED_DIR, WAREHOUSE_SCHEMA
from retail_sales_pipeline.load import QUALITY_TABLE, _read_env_file

logger = logging.getLogger("metabase_setup")

API_TIMEOUT_S = 30
HEALTH_TIMEOUT_S = 180
SYNC_TIMEOUT_S = 180
POLL_INTERVAL_S = 3

SITE_NAME = "Retail Sales Analytics"
COLLECTION_NAME = "Retail Sales Pipeline"
DATABASE_NAME = "Retail Warehouse"
DASHBOARD_NAME = "Retail Sales Overview"
# Host and port of the warehouse inside the compose network (Metabase connects through it).
NETWORK_WAREHOUSE_HOST = "warehouse"
NETWORK_WAREHOUSE_PORT = 5432
GRID_WIDTH = 24  # Metabase dashboards use a 24-column grid


class MetabaseError(RuntimeError):
    """A Metabase API call failed or a check did not pass."""


def _setting(key: str, default: str | None = None) -> str | None:
    """Environment first, then .env for missing keys."""
    value = os.environ.get(key)
    if value:
        return value
    return _read_env_file(ENV_FILE).get(key, default)


def _required(key: str) -> str:
    value = _setting(key)
    if not value:
        raise MetabaseError(f"{key} is not set: export it or add it to .env")
    return value


class Metabase:
    """Minimal REST client: JSON in and out, raises MetabaseError on HTTP errors."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = requests.Session()

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.http.request(
            method, f"{self.base_url}{path}", timeout=API_TIMEOUT_S, **kwargs
        )
        if response.status_code >= 400:
            raise MetabaseError(
                f"{method} {path} -> HTTP {response.status_code}: {response.text[:500]}"
            )
        return response.json() if response.content else None

    def get(self, path: str, **kwargs: Any) -> Any:
        return self._call("GET", path, **kwargs)

    def post(self, path: str, json: Any = None) -> Any:
        return self._call("POST", path, json=json if json is not None else {})

    def put(self, path: str, json: Any) -> Any:
        return self._call("PUT", path, json=json)

    def delete(self, path: str) -> Any:
        return self._call("DELETE", path)

    def login(self, email: str, password: str) -> None:
        session = self.post("/api/session", json={"username": email, "password": password})
        self.http.headers["X-Metabase-Session"] = session["id"]


def wait_for_health(mb: Metabase) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while True:
        try:
            if mb.get("/api/health").get("status") == "ok":
                return
        except (requests.RequestException, MetabaseError) as exc:
            logger.debug("health not ready yet: %s", exc)
        if time.monotonic() > deadline:
            raise MetabaseError(f"Metabase not healthy after {HEALTH_TIMEOUT_S}s: run `make up`")
        time.sleep(POLL_INTERVAL_S)


def ensure_admin(mb: Metabase, email: str, password: str) -> None:
    """Run the first-time setup if needed, then log in as the admin."""
    properties = mb.get("/api/session/properties")
    if not properties.get("has-user-setup"):
        mb.post(
            "/api/setup",
            json={
                "token": properties["setup-token"],
                "user": {
                    "first_name": "Retail",
                    "last_name": "Admin",
                    "email": email,
                    "password": password,
                    "site_name": SITE_NAME,
                },
                "prefs": {"site_name": SITE_NAME, "site_locale": "en", "allow_tracking": False},
            },
        )
        logger.info("first-time setup done for the local admin account")
    mb.login(email, password)


def _items(response: Any) -> list[dict[str, Any]]:
    """List endpoints return either a bare list or {"data": [...]} depending on version."""
    return response["data"] if isinstance(response, dict) else response


def ensure_database(mb: Metabase) -> int:
    for database in _items(mb.get("/api/database")):
        if database["name"] == DATABASE_NAME:
            return database["id"]
    created = mb.post(
        "/api/database",
        json={
            "engine": "postgres",
            "name": DATABASE_NAME,
            "details": {
                "host": NETWORK_WAREHOUSE_HOST,
                "port": NETWORK_WAREHOUSE_PORT,
                "dbname": _required("WAREHOUSE_DB"),
                "user": _required("WAREHOUSE_USER"),
                "password": _required("WAREHOUSE_PASSWORD"),
                "ssl": False,
                "schema-filters-type": "inclusion",
                "schema-filters-patterns": WAREHOUSE_SCHEMA,
            },
            "is_full_sync": True,
            "auto_run_queries": True,
        },
    )
    logger.info("created database connection %s (id %s)", DATABASE_NAME, created["id"])
    return created["id"]


def expected_tables() -> set[str]:
    names = {path.stem for path in PROCESSED_DIR.glob("*.csv")}
    names |= {path.stem for path in MARTS_DIR.glob("*.csv")}
    names.add(QUALITY_TABLE)
    return names


def sync_and_wait(mb: Metabase, database_id: int) -> None:
    """Sync the schema and wait until every expected warehouse table is visible to Metabase."""
    mb.post(f"/api/database/{database_id}/sync_schema")
    expected = expected_tables()
    deadline = time.monotonic() + SYNC_TIMEOUT_S
    while True:
        metadata = mb.get(f"/api/database/{database_id}/metadata")
        visible = {
            t["name"] for t in metadata.get("tables", []) if t.get("schema") == WAREHOUSE_SCHEMA
        }
        missing = expected - visible
        if not missing:
            logger.info("Metabase sees all %d warehouse tables", len(expected))
            return
        if time.monotonic() > deadline:
            raise MetabaseError(f"tables not synced after {SYNC_TIMEOUT_S}s: {sorted(missing)}")
        time.sleep(POLL_INTERVAL_S)


def ensure_collection(mb: Metabase) -> int:
    for collection in _items(mb.get("/api/collection")):
        if collection["name"] == COLLECTION_NAME and not collection.get("archived"):
            return collection["id"]
    created = mb.post("/api/collection", json={"name": COLLECTION_NAME, "color": "#509EE3"})
    return created["id"]


def _collection_items(mb: Metabase, collection_id: int, model: str) -> list[dict[str, Any]]:
    response = mb.get(f"/api/collection/{collection_id}/items", params={"models": model})
    return [item for item in _items(response) if item.get("model") == model]


# ---------------------------------------------------------------------------------------------
# Cards: native SQL questions. Every name is unique across the dashboard.
# ---------------------------------------------------------------------------------------------

MONTH_JOIN = "JOIN analytics.dim_date d ON d.date_key = f.date_key"
GEO_JOIN = "JOIN analytics.dim_geography g ON g.country = f.country"
PRODUCT_JOIN = "JOIN analytics.dim_product p ON p.product = f.product"


def realised(*joins: str) -> str:
    """FROM clause of the fact table, its joins, then the realised-orders filter."""
    return "FROM analytics.fact_sales f " + " ".join(joins) + " WHERE f.is_realised"


@dataclass(frozen=True)
class Card:
    name: str
    sql: str
    display: str
    size: tuple[int, int]  # (size_x, size_y) in grid units
    settings: dict[str, Any]


@dataclass(frozen=True)
class Page:
    name: str
    cards: tuple[Card, ...]


def _currency(column: str, decimals: int = 0) -> dict[str, Any]:
    return {
        f'["name","{column}"]': {
            "number_style": "currency",
            "currency": "USD",
            "currency_style": "symbol",
            "decimals": decimals,
        }
    }


def _scalar(column: str, column_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"scalar.field": column, "column_settings": column_settings or {}}


def _graph(dimension: str, metric: str, column_settings: dict[str, Any] | None = None) -> dict:
    return {
        "graph.dimensions": [dimension],
        "graph.metrics": [metric],
        "column_settings": column_settings or {},
    }


def _pie(dimension: str, metric: str) -> dict[str, Any]:
    return {"pie.dimension": dimension, "pie.metric": metric}


PAGES: tuple[Page, ...] = (
    Page(
        "Overview",
        (
            Card(
                "Total revenue",
                "SELECT ROUND(SUM(f.revenue_usd)::numeric, 2) AS total_revenue_usd " + realised(),
                "scalar",
                (4, 3),
                _scalar("total_revenue_usd", _currency("total_revenue_usd")),
            ),
            Card(
                "Gross profit",
                "SELECT ROUND(SUM(f.gross_profit_usd)::numeric, 2) AS gross_profit_usd "
                + realised(),
                "scalar",
                (4, 3),
                _scalar("gross_profit_usd", _currency("gross_profit_usd")),
            ),
            Card(
                "Gross margin %",
                "SELECT ROUND((100 * SUM(f.gross_profit_usd) / NULLIF(SUM(f.revenue_usd), 0))"
                "::numeric, 2) AS gross_margin_pct " + realised(),
                "scalar",
                (4, 3),
                _scalar("gross_margin_pct"),
            ),
            Card(
                "Orders",
                "SELECT COUNT(DISTINCT f.order_id) AS orders " + realised(),
                "scalar",
                (4, 3),
                _scalar("orders"),
            ),
            Card(
                "Average order value",
                "SELECT ROUND((SUM(f.revenue_usd) / NULLIF(COUNT(DISTINCT f.order_id), 0))"
                "::numeric, 2) AS avg_order_value_usd " + realised(),
                "scalar",
                (4, 3),
                _scalar("avg_order_value_usd", _currency("avg_order_value_usd", 2)),
            ),
            Card(
                "Customers",
                "SELECT COUNT(DISTINCT f.customer_id) AS customers " + realised(),
                "scalar",
                (4, 3),
                _scalar("customers"),
            ),
            Card(
                "Revenue by month",
                "SELECT d.year_month, ROUND(SUM(f.revenue_usd)::numeric, 2) AS revenue_usd "
                + realised(MONTH_JOIN)
                + " GROUP BY d.year_month ORDER BY d.year_month",
                "line",
                (24, 6),
                _graph("year_month", "revenue_usd", _currency("revenue_usd")),
            ),
        ),
    ),
    Page(
        "Products",
        (
            Card(
                "Revenue by product",
                "SELECT p.product, ROUND(SUM(f.revenue_usd)::numeric, 2) AS revenue_usd "
                + realised(PRODUCT_JOIN)
                + " GROUP BY p.product ORDER BY revenue_usd DESC",
                "row",
                (12, 6),
                _graph("product", "revenue_usd", _currency("revenue_usd")),
            ),
            Card(
                "Gross margin % by product",
                "SELECT p.product, "
                "ROUND((100 * SUM(f.gross_profit_usd) / NULLIF(SUM(f.revenue_usd), 0))"
                "::numeric, 2) AS gross_margin_pct "
                + realised(PRODUCT_JOIN)
                + " GROUP BY p.product ORDER BY gross_margin_pct DESC",
                "bar",
                (12, 6),
                _graph("product", "gross_margin_pct"),
            ),
            Card(
                "Product table",
                "SELECT p.product, ROUND(SUM(f.revenue_usd)::numeric, 2) AS revenue_usd, "
                "ROUND(SUM(f.gross_profit_usd)::numeric, 2) AS gross_profit_usd, "
                "SUM(f.quantity) AS units, "
                "ROUND((100 * SUM(f.gross_profit_usd) / NULLIF(SUM(f.revenue_usd), 0))"
                "::numeric, 2) AS gross_margin_pct, "
                "RANK() OVER (ORDER BY SUM(f.revenue_usd) DESC) AS revenue_rank "
                + realised(PRODUCT_JOIN)
                + " GROUP BY p.product ORDER BY revenue_rank",
                "table",
                (24, 8),
                {},
            ),
        ),
    ),
    Page(
        "Geography",
        (
            Card(
                "Revenue by region",
                "SELECT g.region, ROUND(SUM(f.revenue_usd)::numeric, 2) AS revenue_usd "
                + realised(GEO_JOIN)
                + " GROUP BY g.region ORDER BY revenue_usd DESC",
                "pie",
                (12, 6),
                _pie("region", "revenue_usd"),
            ),
            Card(
                "Revenue by country",
                "SELECT g.country, ROUND(SUM(f.revenue_usd)::numeric, 2) AS revenue_usd "
                + realised(GEO_JOIN)
                + " GROUP BY g.country ORDER BY revenue_usd DESC",
                "bar",
                (12, 6),
                _graph("country", "revenue_usd", _currency("revenue_usd")),
            ),
            Card(
                "Region and country table",
                "SELECT g.region, g.country, ROUND(SUM(f.revenue_usd)::numeric, 2) AS revenue_usd, "
                "COUNT(DISTINCT f.order_id) AS orders, "
                "ROUND((100 * SUM(f.gross_profit_usd) / NULLIF(SUM(f.revenue_usd), 0))"
                "::numeric, 2) AS gross_margin_pct, "
                "COUNT(DISTINCT f.customer_id) AS customers "
                + realised(GEO_JOIN)
                + " GROUP BY g.region, g.country ORDER BY g.region, revenue_usd DESC",
                "table",
                (24, 8),
                {},
            ),
        ),
    ),
    Page(
        "Trends",
        (
            Card(
                "Revenue by year and YoY",
                "WITH yearly AS (SELECT d.year, SUM(f.revenue_usd) AS revenue "
                + realised(MONTH_JOIN)
                + " GROUP BY d.year) "
                "SELECT year, ROUND(revenue::numeric, 2) AS revenue_usd, "
                "ROUND((100 * (revenue - LAG(revenue) OVER (ORDER BY year)) "
                "/ NULLIF(LAG(revenue) OVER (ORDER BY year), 0))::numeric, 2) AS revenue_yoy_pct "
                "FROM yearly ORDER BY year",
                "table",
                (12, 6),
                {},
            ),
            Card(
                "Revenue by month of year",
                "WITH monthly AS (SELECT d.year, d.month, d.month_name, "
                "SUM(f.revenue_usd) AS revenue "
                + realised(MONTH_JOIN)
                + " GROUP BY d.year, d.month, d.month_name) "
                "SELECT month_name, ROUND(AVG(revenue)::numeric, 2) AS avg_revenue_usd "
                "FROM monthly GROUP BY month, month_name ORDER BY month",
                "bar",
                (12, 6),
                _graph("month_name", "avg_revenue_usd", _currency("avg_revenue_usd")),
            ),
            Card(
                "Revenue by season",
                "SELECT d.season, ROUND(SUM(f.revenue_usd)::numeric, 2) AS revenue_usd "
                + realised(MONTH_JOIN)
                + " GROUP BY d.season ORDER BY CASE d.season WHEN 'Winter' THEN 1 "
                "WHEN 'Spring' THEN 2 WHEN 'Summer' THEN 3 ELSE 4 END",
                "bar",
                (12, 6),
                _graph("season", "revenue_usd", _currency("revenue_usd")),
            ),
        ),
    ),
    Page(
        "Customers",
        # Segment and industry come from fact_sales (order level), never from dim_customer.
        (
            Card(
                "Revenue by segment",
                "SELECT f.customer_segment, ROUND(SUM(f.revenue_usd)::numeric, 2) AS revenue_usd "
                + realised()
                + " GROUP BY f.customer_segment ORDER BY revenue_usd DESC",
                "bar",
                (12, 6),
                _graph("customer_segment", "revenue_usd", _currency("revenue_usd")),
            ),
            Card(
                "Gross margin % by segment",
                "SELECT f.customer_segment, ROUND((100 * SUM(f.gross_profit_usd) "
                "/ NULLIF(SUM(f.revenue_usd), 0))::numeric, 2) AS gross_margin_pct "
                + realised()
                + " GROUP BY f.customer_segment ORDER BY gross_margin_pct DESC",
                "bar",
                (12, 6),
                _graph("customer_segment", "gross_margin_pct"),
            ),
            Card(
                "Revenue per customer by segment",
                "SELECT f.customer_segment, ROUND((SUM(f.revenue_usd) "
                "/ NULLIF(COUNT(DISTINCT f.customer_id), 0))::numeric, 2) "
                "AS revenue_per_customer_usd "
                + realised()
                + " GROUP BY f.customer_segment ORDER BY revenue_per_customer_usd DESC",
                "bar",
                (12, 6),
                _graph(
                    "customer_segment",
                    "revenue_per_customer_usd",
                    _currency("revenue_per_customer_usd"),
                ),
            ),
            Card(
                "Revenue by channel",
                "SELECT f.channel, ROUND(SUM(f.revenue_usd)::numeric, 2) AS revenue_usd "
                + realised()
                + " GROUP BY f.channel ORDER BY revenue_usd DESC",
                "bar",
                (12, 6),
                _graph("channel", "revenue_usd", _currency("revenue_usd")),
            ),
            Card(
                "NPS and churn by segment",
                # Pre-aggregated mart (averages over order rows, see docs/metabase.md).
                "SELECT customer_segment, customers, revenue_usd, avg_nps, avg_churn_probability "
                "FROM analytics.sales_by_segment ORDER BY revenue_usd DESC",
                "table",
                (24, 6),
                {},
            ),
        ),
    ),
    Page(
        "Data quality",
        (
            Card(
                "Cancellation rate %",
                "SELECT ROUND((100.0 * COUNT(*) FILTER (WHERE f.order_status = 'Cancelled') "
                "/ COUNT(*))::numeric, 2) AS cancellation_rate_pct FROM analytics.fact_sales f",
                "scalar",
                (4, 3),
                _scalar("cancellation_rate_pct"),
            ),
            Card(
                "Refund rate %",
                "SELECT ROUND((100.0 * COUNT(*) FILTER (WHERE f.order_status = 'Refunded') "
                "/ COUNT(*))::numeric, 2) AS refund_rate_pct FROM analytics.fact_sales f",
                "scalar",
                (4, 3),
                _scalar("refund_rate_pct"),
            ),
            Card(
                "Valid rows %",
                f"SELECT valid_pct FROM analytics.{QUALITY_TABLE}",
                "scalar",
                (4, 3),
                _scalar("valid_pct"),
            ),
            Card(
                "Order status breakdown",
                "SELECT f.order_status, COUNT(*) AS orders FROM analytics.fact_sales f "
                "GROUP BY f.order_status ORDER BY orders DESC",
                "pie",
                (12, 6),
                _pie("order_status", "orders"),
            ),
            Card(
                "Validation report",
                "SELECT total_rows, valid_rows, rejected_rows, valid_pct, duration_seconds, "
                f"errors_by_rule FROM analytics.{QUALITY_TABLE}",
                "table",
                (12, 6),
                {},
            ),
        ),
    ),
)


def flow_layout(sizes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Place (width, height) blocks left to right, starting a new row when one would overflow.

    Returns (col, row) per block. Blocks never overlap and each row is as tall as its tallest block.
    """
    positions: list[tuple[int, int]] = []
    col = row = row_height = 0
    for width, height in sizes:
        if width > GRID_WIDTH:
            raise ValueError(f"block wider than the grid: {width}")
        if col + width > GRID_WIDTH:
            row += row_height
            col = row_height = 0
        positions.append((col, row))
        col += width
        row_height = max(row_height, height)
    return positions


def ensure_cards(mb: Metabase, database_id: int, collection_id: int) -> dict[str, int]:
    """Delete the cards with our names in the collection, then create them again."""
    wanted = {card.name for page in PAGES for card in page.cards}
    for item in _collection_items(mb, collection_id, "card"):
        if item["name"] in wanted:
            mb.delete(f"/api/card/{item['id']}")

    card_ids: dict[str, int] = {}
    for page in PAGES:
        for card in page.cards:
            created = mb.post(
                "/api/card",
                json={
                    "name": card.name,
                    "collection_id": collection_id,
                    "display": card.display,
                    "visualization_settings": card.settings,
                    "dataset_query": {
                        "type": "native",
                        "database": database_id,
                        "native": {"query": card.sql.strip()},
                    },
                },
            )
            card_ids[card.name] = created["id"]
    logger.info("created %d cards", len(card_ids))
    return card_ids


def ensure_dashboard(mb: Metabase, collection_id: int, card_ids: dict[str, int]) -> int:
    """Create or reuse the dashboard and set its tabs and cards in one PUT."""
    dashboard_id = next(
        (
            item["id"]
            for item in _collection_items(mb, collection_id, "dashboard")
            if item["name"] == DASHBOARD_NAME
        ),
        None,
    )
    if dashboard_id is None:
        dashboard_id = mb.post(
            "/api/dashboard", json={"name": DASHBOARD_NAME, "collection_id": collection_id}
        )["id"]

    tabs: list[dict[str, Any]] = []
    dashcards: list[dict[str, Any]] = []
    temp_id = -1
    for tab_index, page in enumerate(PAGES, start=1):
        tabs.append({"id": -tab_index, "name": page.name, "position": tab_index - 1})
        positions = flow_layout([card.size for card in page.cards])
        for card, (col, row) in zip(page.cards, positions, strict=True):
            dashcards.append(
                {
                    "id": temp_id,
                    "card_id": card_ids[card.name],
                    "dashboard_tab_id": -tab_index,
                    "row": row,
                    "col": col,
                    "size_x": card.size[0],
                    "size_y": card.size[1],
                    "parameter_mappings": [],
                    "visualization_settings": {},
                }
            )
            temp_id -= 1

    mb.put(
        f"/api/dashboard/{dashboard_id}",
        json={
            "name": DASHBOARD_NAME,
            "description": "Retail sales KPIs built from the pipeline outputs (analytics schema).",
            "collection_id": collection_id,
            "parameters": [],
            "tabs": tabs,
            "dashcards": dashcards,
        },
    )
    logger.info("dashboard %s: %d tabs, %d cards", dashboard_id, len(tabs), len(dashcards))
    return dashboard_id


def verify_dashboard(mb: Metabase, dashboard_id: int) -> None:
    """Read the dashboard back and check that the tabs and cards were stored."""
    dashboard = mb.get(f"/api/dashboard/{dashboard_id}")
    tabs = dashboard.get("tabs", [])
    cards = [c for c in dashboard.get("dashcards", []) if c.get("card_id")]
    expected = sum(len(page.cards) for page in PAGES)
    if len(tabs) != len(PAGES) or len(cards) != expected:
        raise MetabaseError(
            f"dashboard stored {len(tabs)} tabs and {len(cards)} cards, "
            f"expected {len(PAGES)} and {expected}"
        )
    tab_ids = {tab["id"] for tab in tabs}
    orphans = [c for c in cards if c.get("dashboard_tab_id") not in tab_ids]
    if orphans:
        raise MetabaseError(f"{len(orphans)} cards are not attached to a stored tab")


def check_cards(mb: Metabase, card_ids: dict[str, int]) -> list[tuple[str, int, str]]:
    """Run every card and return (name, rows, status). Raises if one does not return rows."""
    results: list[tuple[str, int, str]] = []
    failures: list[str] = []
    for name, card_id in card_ids.items():
        response = mb.post(f"/api/card/{card_id}/query", json={})
        status = response.get("status", "unknown")
        rows = int(response.get("row_count") or 0)
        results.append((name, rows, status))
        if status != "completed" or rows <= 0:
            failures.append(f"{name}: status={status} rows={rows}")
    if failures:
        raise MetabaseError("cards failing the check: " + "; ".join(failures))
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base_url = _setting("MB_URL", "http://localhost:3000") or "http://localhost:3000"
    email = _required("MB_ADMIN_EMAIL")
    password = _required("MB_ADMIN_PASSWORD")

    mb = Metabase(base_url)
    wait_for_health(mb)
    ensure_admin(mb, email, password)
    database_id = ensure_database(mb)
    sync_and_wait(mb, database_id)
    collection_id = ensure_collection(mb)
    card_ids = ensure_cards(mb, database_id, collection_id)
    dashboard_id = ensure_dashboard(mb, collection_id, card_ids)
    verify_dashboard(mb, dashboard_id)
    results = check_cards(mb, card_ids)

    print("\nCard                                  rows  status")
    for name, rows, status in results:
        print(f"{name:<38}{rows:>5}  {status}")
    print(f"\nDashboard: {base_url}/dashboard/{dashboard_id}")
    print("Tabs: " + ", ".join(page.name for page in PAGES))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (MetabaseError, requests.RequestException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
