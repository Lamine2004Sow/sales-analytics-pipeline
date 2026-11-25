"""Audit what the sales data can and cannot support.

Each public function answers one question and returns plain numbers, dicts or DataFrames:

- ``forecastability``: is the monthly revenue series predictable at all? (CV, autocorrelation,
  Ljung-Box white-noise test, STL seasonal and trend strength.)
- ``backtest_forecasters``: do simple and classical forecasters beat a trivial mean forecast
  on rolling-origin folds? Every model is fitted on the training fold only.
- ``leakage_demo``: how much of the order amount do features explain when they are known only
  after the order (or are derived from revenue), compared with features known before it?
- ``synthetic_fingerprints``: signs that the dataset is generated (exact revenue identity,
  flat margins, customer attributes that change between orders, ...).
- ``derived_column_check``: which customer snapshot columns are computed from the orders?

Inputs are polars frames or numpy arrays. Nothing here plots. All randomness is seeded
(``SEED``), so results are deterministic.
"""

import warnings
from collections.abc import Callable
from datetime import date

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

from retail_sales_pipeline.config import REALISED_STATUS

SEED = 42
SPLIT_DATE = date(2023, 1, 1)
WHITE_NOISE_ALPHA = 0.05
IDENTITY_TOL = 0.15  # absolute USD tolerance for revenue == quantity * price_after_discount

# --------------------------------------------------------------------------- series


def monthly_series(clean_df: pl.DataFrame) -> pl.DataFrame:
    """Realised revenue per calendar month on a continuous month spine (gaps count as 0)."""
    realised = clean_df.filter(pl.col("is_realised"))
    monthly = realised.group_by(
        pl.col("transaction_date").dt.month_start().alias("month_start")
    ).agg(pl.col("revenue_usd").sum())
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
        .with_columns(pl.col("revenue_usd").fill_null(0.0))
        .sort("month_start")
    )


# ------------------------------------------------------------------ forecastability


def _strength(component: np.ndarray, resid: np.ndarray) -> float:
    """Hyndman's strength: max(0, 1 - Var(resid) / Var(component + resid))."""
    denom = np.var(component + resid)
    if denom == 0:
        return 0.0
    return float(max(0.0, 1.0 - np.var(resid) / denom))


def forecastability(
    series: np.ndarray, period: int = 12, lags: tuple[int, ...] = (1, 12, 24)
) -> dict:
    """Summarise how predictable a monthly series is. See module docstring."""
    y = np.asarray(series, dtype=float)
    if len(y) < 2 * period + 1:
        raise ValueError(f"need at least {2 * period + 1} points, got {len(y)}")

    mean = y.mean()
    rho = acf(y, nlags=max(lags), fft=False)
    lb = acorr_ljungbox(y, lags=[period, 2 * period], return_df=True)
    p_12 = float(lb.loc[period, "lb_pvalue"])
    p_24 = float(lb.loc[2 * period, "lb_pvalue"])

    stl = STL(y, period=period).fit()
    seasonal_strength = _strength(stl.seasonal, stl.resid)
    trend_strength = _strength(stl.trend, stl.resid)

    return {
        "n_months": len(y),
        "cv": float(y.std(ddof=1) / mean) if mean else float("nan"),
        "acf": {f"lag{k}": float(rho[k]) for k in lags},
        "ljung_box_p_12": p_12,
        "ljung_box_p_24": p_24,
        "seasonal_strength": seasonal_strength,
        "trend_strength": trend_strength,
        "is_white_noise": bool(p_12 > WHITE_NOISE_ALPHA and p_24 > WHITE_NOISE_ALPHA),
    }


# ----------------------------------------------------------------------- backtest


def _folds(n: int, n_splits: int, test_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds (train indices, test indices). Train always precedes test."""
    return list(TimeSeriesSplit(n_splits=n_splits, test_size=test_size).split(np.arange(n)))


def _f_mean(train: np.ndarray, horizon: int, _start_month: int) -> np.ndarray:
    return np.full(horizon, train.mean())


def _f_naive(train: np.ndarray, horizon: int, _start_month: int) -> np.ndarray:
    return np.full(horizon, train[-1])


def _f_seasonal_naive(
    train: np.ndarray, horizon: int, _start_month: int, period: int = 12
) -> np.ndarray:
    idx = len(train) - period + (np.arange(horizon) % period)
    return train[idx]


def _f_drift(train: np.ndarray, horizon: int, _start_month: int) -> np.ndarray:
    slope = (train[-1] - train[0]) / (len(train) - 1)
    return train[-1] + slope * np.arange(1, horizon + 1)


def _f_rolling_mean_3(train: np.ndarray, horizon: int, _start_month: int) -> np.ndarray:
    return np.full(horizon, train[-3:].mean())


def _f_linear_trend(train: np.ndarray, horizon: int, _start_month: int) -> np.ndarray:
    t = np.arange(len(train)).reshape(-1, 1)
    model = LinearRegression().fit(t, train)
    return model.predict(np.arange(len(train), len(train) + horizon).reshape(-1, 1))


def _f_holt_winters(train: np.ndarray, horizon: int, _start_month: int) -> np.ndarray:
    """Additive trend and seasonality. Returns NaN when the optimiser cannot fit the fold."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = ExponentialSmoothing(
                train,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated",
            ).fit()
            return np.asarray(fit.forecast(horizon), dtype=float)
        except (ValueError, np.linalg.LinAlgError):
            return np.full(horizon, np.nan)


GBR_LAGS = (1, 2, 3, 12)


def _gbr_features(history: list[float], start_month: int) -> list[float]:
    """Lags 1, 2, 3, 12 and the calendar month of the next (unseen) point."""
    t = len(history)
    return [history[t - k] for k in GBR_LAGS] + [(start_month - 1 + t) % 12 + 1]


def _f_gbr_lags(train: np.ndarray, horizon: int, start_month: int) -> np.ndarray:
    """Gradient boosting on lag features, forecast recursively one step at a time."""
    max_lag = max(GBR_LAGS)
    if len(train) <= max_lag:
        return np.full(horizon, np.nan)
    X = [_gbr_features(train[:t].tolist(), start_month) for t in range(max_lag, len(train))]
    y = train[max_lag:]
    model = HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.1, min_samples_leaf=5, random_state=SEED
    ).fit(np.array(X), y)
    history = train.tolist()
    preds: list[float] = []
    for _ in range(horizon):
        step = float(model.predict(np.array([_gbr_features(history, start_month)]))[0])
        preds.append(step)
        history.append(step)
    return np.array(preds)


FORECASTERS: dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {
    "mean": _f_mean,
    "naive": _f_naive,
    "seasonal_naive": _f_seasonal_naive,
    "drift": _f_drift,
    "rolling_mean_3": _f_rolling_mean_3,
    "linear_trend": _f_linear_trend,
    "holt_winters": _f_holt_winters,
    "gbr_lags": _f_gbr_lags,
}


def _errors(actual: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    if not np.all(np.isfinite(pred)):
        return {"mae": None, "rmse": None, "mape": None}
    err = actual - pred
    nonzero = actual != 0
    mape = float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": mape,
    }


def backtest_forecasters(
    series: np.ndarray,
    n_splits: int = 5,
    test_size: int = 12,
    start_month: int = 1,
) -> pl.DataFrame:
    """Rolling-origin backtest of every forecaster in FORECASTERS.

    ``start_month`` is the calendar month (1-12) of the first point, used by ``gbr_lags``.
    Returns one row per (model, fold) with mae, rmse and mape in percent. A model that
    cannot fit a fold gets null errors for that fold.
    """
    y = np.asarray(series, dtype=float)
    rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(_folds(len(y), n_splits, test_size)):
        train, actual = y[train_idx], y[test_idx]
        for name, forecast in FORECASTERS.items():
            pred = forecast(train, len(test_idx), start_month)
            rows.append({"model": name, "fold": fold, **_errors(actual, pred)})
    return pl.DataFrame(
        rows,
        schema={
            "model": pl.String,
            "fold": pl.Int64,
            "mae": pl.Float64,
            "rmse": pl.Float64,
            "mape": pl.Float64,
        },
    )


def summarise_backtest(df: pl.DataFrame) -> pl.DataFrame:
    """Mean and std of errors per model, sorted by mean MAE, with skill versus ``mean``.

    ``skill_vs_mean_pct = (1 - mae_mean / mae_mean_of_mean_model) * 100``: positive means the
    model beats the train-mean forecast.
    """
    summary = df.group_by("model").agg(
        pl.col("mae").mean().alias("mae_mean"),
        pl.col("mae").std().alias("mae_std"),
        pl.col("rmse").mean().alias("rmse_mean"),
        pl.col("mape").mean().alias("mape_mean"),
        pl.col("mae").count().alias("n_folds"),
    )
    base = summary.filter(pl.col("model") == "mean")["mae_mean"]
    base_mae = base.item() if base.len() and base.item() is not None else None
    if base_mae:
        skill = (1 - pl.col("mae_mean") / base_mae) * 100
    else:
        skill = pl.lit(None, dtype=pl.Float64)
    return summary.with_columns(skill_vs_mean_pct=skill).sort("mae_mean", nulls_last=True)


# ----------------------------------------------------------------------- leakage

LEAKY_NUMERIC = ["quantity", "unit_price_usd", "discount_pct"]
DERIVED_CUSTOMER_NUMERIC = [
    "engagement_score",
    "rfm_total_score",
    "rfm_m_score",
    "monetary_total_usd",
    "avg_order_value_usd",
    "clv_3yr_usd",
]
HONEST_CATEGORICAL = [
    "product",
    "country",
    "channel",
    "customer_segment",
    "industry",
    "payment_method",
    "month",
    "day_of_week",
]
HONEST_NUMERIC = ["is_weekend", "gdp_per_capita_usd"]

# feature_set -> (categorical columns, numeric columns)
FEATURE_SETS: dict[str, tuple[list[str], list[str]]] = {
    "leaky": ([], LEAKY_NUMERIC),
    "derived_customer": ([], DERIVED_CUSTOMER_NUMERIC),
    "honest": (HONEST_CATEGORICAL, HONEST_NUMERIC),
    "honest_no_product": ([c for c in HONEST_CATEGORICAL if c != "product"], HONEST_NUMERIC),
}


def _ridge_predict(
    cat_tr: np.ndarray, cat_te: np.ndarray, num_tr: np.ndarray, num_te: np.ndarray, y_tr: np.ndarray
) -> np.ndarray:
    blocks_tr: list[np.ndarray] = []
    blocks_te: list[np.ndarray] = []
    if cat_tr.shape[1]:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cat_tr)
        blocks_tr.append(ohe.transform(cat_tr))
        blocks_te.append(ohe.transform(cat_te))
    if num_tr.shape[1]:
        scaler = StandardScaler().fit(num_tr)
        blocks_tr.append(scaler.transform(num_tr))
        blocks_te.append(scaler.transform(num_te))
    model = Ridge(alpha=1.0).fit(np.hstack(blocks_tr), y_tr)
    return model.predict(np.hstack(blocks_te))


def _hgb_predict(
    cat_tr: np.ndarray, cat_te: np.ndarray, num_tr: np.ndarray, num_te: np.ndarray, y_tr: np.ndarray
) -> np.ndarray:
    blocks_tr: list[np.ndarray] = [num_tr]
    blocks_te: list[np.ndarray] = [num_te]
    categorical = None
    if cat_tr.shape[1]:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan).fit(cat_tr)
        blocks_tr.insert(0, enc.transform(cat_tr))
        blocks_te.insert(0, enc.transform(cat_te))
        categorical = np.array([True] * cat_tr.shape[1] + [False] * num_tr.shape[1])
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.1,
        categorical_features=categorical,
        random_state=SEED,
    ).fit(np.hstack(blocks_tr), y_tr)
    return model.predict(np.hstack(blocks_te))


def _metrics(feature_set: str, model: str, y_te: np.ndarray, pred: np.ndarray, n_tr: int) -> dict:
    return {
        "feature_set": feature_set,
        "model": model,
        "r2": float(r2_score(y_te, pred)),
        "mae": float(mean_absolute_error(y_te, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "n_train": n_tr,
        "n_test": len(y_te),
    }


def _matrix(df: pl.DataFrame, cat: list[str], num: list[str]) -> tuple[np.ndarray, np.ndarray]:
    n = df.height
    cat_arr = df.select(pl.col(cat).cast(pl.String)).to_numpy() if cat else np.empty((n, 0), object)
    num_arr = df.select(pl.col(num).cast(pl.Float64)).to_numpy() if num else np.empty((n, 0))
    return cat_arr, num_arr


def leakage_demo(
    clean_df: pl.DataFrame, sample: int | None = None, seed: int = SEED
) -> pl.DataFrame:
    """Predict realised revenue_usd per order with each feature set, chronological split.

    Train: transaction_date < 2023-01-01. Test: the rest. ``sample`` keeps that many random
    realised rows (seeded) to make big runs faster. The ``none``/``mean`` row is the baseline.
    """
    df = clean_df.filter(pl.col("is_realised")).with_columns(
        month=pl.col("transaction_date").dt.month(),
        day_of_week=pl.col("transaction_date").dt.weekday() - 1,
        is_weekend=pl.col("is_weekend").cast(pl.Float64),
    )
    if sample is not None:
        df = df.sample(n=min(sample, df.height), seed=seed)
    train = df.filter(pl.col("transaction_date") < SPLIT_DATE)
    test = df.filter(pl.col("transaction_date") >= SPLIT_DATE)
    y_tr = train["revenue_usd"].cast(pl.Float64).to_numpy()
    y_te = test["revenue_usd"].cast(pl.Float64).to_numpy()

    rows = [_metrics("none", "mean", y_te, np.full(len(y_te), y_tr.mean()), len(y_tr))]
    for name, (cat, num) in FEATURE_SETS.items():
        cat_tr, num_tr = _matrix(train, cat, num)
        cat_te, num_te = _matrix(test, cat, num)
        for model_name, predict in (("ridge", _ridge_predict), ("hist_gbr", _hgb_predict)):
            pred = predict(cat_tr, cat_te, num_tr, num_te, y_tr)
            rows.append(_metrics(name, model_name, y_te, pred, len(y_tr)))
    return pl.DataFrame(rows)


# ------------------------------------------------------------------- fingerprints


def _normalise(df: pl.DataFrame) -> pl.DataFrame:
    """Accept raw (string dates, no is_realised) or cleaned frames."""
    if df.schema["transaction_date"] == pl.String:
        df = df.with_columns(pl.col("transaction_date").str.to_date("%Y-%m-%d"))
    if "is_realised" not in df.columns:
        df = df.with_columns(is_realised=pl.col("order_status") == REALISED_STATUS)
    return df


def synthetic_fingerprints(df: pl.DataFrame) -> dict:
    """Measurable signs that the data is generated rather than recorded. Works on raw or clean."""
    df = _normalise(df)
    null_total = int(sum(df.null_count().row(0)))
    duplicates = df.height - df.unique().height

    identity_ok = (df["revenue_usd"] - df["quantity"] * df["unit_price_after_discount_usd"]).abs()
    identity_share = float((identity_ok <= IDENTITY_TOL).mean())

    margin_std_pp = float(
        df.group_by("product").agg(pl.col("profit_margin_pct").mean())["profit_margin_pct"].std()
    )

    by_month = (
        monthly_series(df)
        .group_by(pl.col("month_start").dt.month().alias("month"))
        .agg(pl.col("revenue_usd").mean())["revenue_usd"]
    )
    moy_cv = float(by_month.std() / by_month.mean())

    per_customer = df.group_by("customer_id").agg(
        pl.len().alias("n"),
        pl.col("customer_segment").n_unique().alias("n_segment"),
        pl.col("churn_probability").n_unique().alias("n_churn_prob"),
    )
    multi = per_customer.filter(pl.col("n") > 1)

    pair = df.select(
        pl.col("churn_probability").cast(pl.Float64), pl.col("is_churned").cast(pl.Float64)
    ).drop_nulls()
    churn_corr = float(np.corrcoef(pair["churn_probability"], pair["is_churned"])[0, 1])

    flags = []
    if identity_share > 0.99:
        flags.append("revenue_usd = quantity x unit_price_after_discount_usd on almost every row")
    if margin_std_pp < 1.0:
        flags.append(f"profit margin flat across products (sd {margin_std_pp:.2f} pp)")
    if int((multi["n_segment"] > 1).sum()) > 0:
        flags.append("customer_segment changes between orders of the same customer")
    if abs(churn_corr) < 0.1:
        flags.append(f"churn_probability unrelated to is_churned (r={churn_corr:.2f})")
    verdict = "synthetic-looking: " + "; ".join(flags) if flags else "no strong synthetic signature"

    return {
        "n_rows": df.height,
        "null_count_total": null_total,
        "duplicate_rows": duplicates,
        "revenue_identity_share": identity_share,
        "product_margin_sd_pp": margin_std_pp,
        "month_of_year_revenue_cv": moy_cv,
        "customers_multi_order": multi.height,
        "customers_segment_conflict": int((multi["n_segment"] > 1).sum()),
        "customers_churn_prob_conflict": int((multi["n_churn_prob"] > 1).sum()),
        "churn_prob_vs_is_churned_corr": churn_corr,
        "verdict": verdict,
    }


# ------------------------------------------------------------- derived columns

DERIVED_CHECK_COLUMNS = (
    "monetary_total_usd",
    "avg_order_value_usd",
    "clv_3yr_usd",
    "engagement_score",
    "rfm_total_score",
    "rfm_m_score",
    "frequency_orders",
)


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return None
    rx = pl.Series(x[keep]).rank().to_numpy().astype(float)
    ry = pl.Series(y[keep]).rank().to_numpy().astype(float)
    return _corr(rx, ry)


def derived_column_check(
    clean_df: pl.DataFrame,
    target: str = "revenue_usd",
    columns: tuple[str, ...] | list[str] = DERIVED_CHECK_COLUMNS,
) -> pl.DataFrame:
    """Correlate each column with the target at row level and with the customer's actual total.

    Row level: correlation of the column with ``target`` across all rows given.
    Customer level: the column's value on each customer's latest order, against the customer's
    actual sum of realised ``target`` (for ``frequency_orders``: the actual count of all orders).
    A customer-level correlation near 1 suggests the snapshot was computed from the orders.
    """
    df = _normalise(clean_df).sort(["transaction_date", "order_id"])
    target_all = df[target].cast(pl.Float64).to_numpy()
    actual_sum = (
        df.filter(pl.col("is_realised"))
        .group_by("customer_id")
        .agg(pl.col(target).cast(pl.Float64).sum().alias("actual_sum"))
    )
    actual_count = df.group_by("customer_id").agg(pl.len().alias("actual_count"))
    customers = (
        df.unique(subset="customer_id", keep="last", maintain_order=True)
        .join(actual_sum, on="customer_id", how="left")
        .join(actual_count, on="customer_id", how="left")
        .with_columns(pl.col("actual_sum").fill_null(0.0))
    )

    rows = []
    for col in columns:
        x_row = df[col].cast(pl.Float64).to_numpy()
        x_cust = customers[col].cast(pl.Float64).to_numpy()
        if col == "frequency_orders":
            actual, actual_name = (
                customers["actual_count"].cast(pl.Float64).to_numpy(),
                "order_count",
            )
        else:
            actual, actual_name = customers["actual_sum"].to_numpy(), f"sum_realised_{target}"
        rows.append(
            {
                "column": col,
                "pearson_row": _corr(x_row, target_all),
                "spearman_row": _spearman(x_row, target_all),
                "customer_actual": actual_name,
                "pearson_customer": _corr(x_cust, actual),
                "spearman_customer": _spearman(x_cust, actual),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "column": pl.String,
            "pearson_row": pl.Float64,
            "spearman_row": pl.Float64,
            "customer_actual": pl.String,
            "pearson_customer": pl.Float64,
            "spearman_customer": pl.Float64,
        },
    )
