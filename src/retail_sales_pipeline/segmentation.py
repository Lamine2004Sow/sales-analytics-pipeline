"""Customer segmentation on recomputed RFM (recency, frequency, monetary).

Choice of method: K-Means only, on three log/standardised RFM features, compared with the
classic rule-based RFM quintile segments (the non-ML baseline). No DBSCAN or hierarchical
clustering: the goal is one honest experiment, not a method zoo.

K-Means always returns k clusters, even on structure-free data, so a good silhouette alone
proves nothing. The guards are:
- `stability`: do the clusters survive subsampling and reseeding (adjusted Rand index)?
- `null_baseline`: is the silhouette clearly above what uniform or column-permuted data
  (same marginals, no joint structure) gives? If not, the clusters reflect marginal shapes,
  not groups.

Pre-computed RFM columns in the raw data are not used: customer attributes vary per order.
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

RFM_COLUMNS = ["recency_days", "frequency", "monetary"]


def compute_rfm(clean_df: pl.DataFrame, snapshot_date=None) -> pl.DataFrame:
    """One row per customer with RFM computed from realised orders only.

    snapshot_date defaults to the day after the latest transaction_date of `clean_df`.
    recency_days = snapshot - last realised order date; frequency = number of realised
    orders; monetary = sum of revenue_usd over realised orders.
    """
    realised = clean_df.filter(pl.col("is_realised"))
    if snapshot_date is None:
        snapshot_date = clean_df["transaction_date"].max() + dt.timedelta(days=1)
    return (
        realised.group_by("customer_id")
        .agg(
            first_order=pl.col("transaction_date").min(),
            last_order=pl.col("transaction_date").max(),
            frequency=pl.len(),
            monetary=pl.col("revenue_usd").sum(),
        )
        .with_columns(
            recency_days=(pl.lit(snapshot_date) - pl.col("last_order")).dt.total_days(),
        )
        .select("customer_id", "recency_days", "frequency", "monetary", "first_order", "last_order")
        .sort("customer_id")
    )


def rfm_features(rfm: pl.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Feature matrix: log1p(frequency), log1p(monetary), recency_days, then standardised.

    Returns (X, scaler) so the scaler can be reused on new data.
    """
    raw = np.column_stack(
        [
            np.log1p(rfm["frequency"].to_numpy().astype(float)),
            np.log1p(rfm["monetary"].to_numpy().astype(float)),
            rfm["recency_days"].to_numpy().astype(float),
        ]
    )
    scaler = StandardScaler()
    return scaler.fit_transform(raw), scaler


def evaluate_k(
    X: np.ndarray,
    k_range: Iterable[int] = range(2, 9),
    seed: int = 42,
    sample_size: int = 10_000,
) -> pl.DataFrame:
    """Fit K-Means for each k and report inertia and internal validity indices.

    Silhouette is estimated on a random sample of `sample_size` rows (exact silhouette is
    quadratic in n). Higher silhouette and Calinski-Harabasz are better; lower Davies-Bouldin
    is better.
    """
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        labels = model.labels_
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(
                    silhouette_score(
                        X, labels, sample_size=min(sample_size, len(X)), random_state=seed
                    )
                ),
                "davies_bouldin": float(davies_bouldin_score(X, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            }
        )
    return pl.DataFrame(rows)


def stability(X: np.ndarray, k: int, n_runs: int = 10, frac: float = 0.8, seed: int = 42) -> dict:
    """Agreement (adjusted Rand index) between K-Means runs.

    Subsample: fit on `n_runs` random `frac` subsamples, predict labels on the full X, and
    compare every pair of runs. Seed: fit on the full X with `n_runs` different random states.
    ARI is 1 for identical partitions and about 0 for random ones.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    sub_size = int(round(frac * n))
    sub_labels = []
    for run in range(n_runs):
        idx = rng.choice(n, size=sub_size, replace=False)
        model = KMeans(n_clusters=k, n_init=10, random_state=seed + run).fit(X[idx])
        sub_labels.append(model.predict(X))

    seed_labels = [
        KMeans(n_clusters=k, n_init=10, random_state=seed + 1000 + run).fit(X).labels_
        for run in range(n_runs)
    ]

    sub_ari = _pairwise_ari(sub_labels)
    seed_ari = _pairwise_ari(seed_labels)
    return {
        "k": k,
        "n_runs": n_runs,
        "frac": frac,
        "subsample_ari_mean": float(np.mean(sub_ari)),
        "subsample_ari_min": float(np.min(sub_ari)),
        "subsample_ari_std": float(np.std(sub_ari)),
        "seed_ari_mean": float(np.mean(seed_ari)),
        "seed_ari_min": float(np.min(seed_ari)),
        "seed_ari_std": float(np.std(seed_ari)),
    }


def _pairwise_ari(label_sets: Sequence[np.ndarray]) -> list[float]:
    return [adjusted_rand_score(a, b) for a, b in itertools.combinations(label_sets, 2)]


def null_baseline(X: np.ndarray, k: int, seed: int = 42, sample_size: int = 10_000) -> dict:
    """Silhouette of K-Means on the real data versus two structure-free references.

    - uniform: points drawn uniformly in the bounding box of X (same shape, no structure).
    - permuted: each column shuffled independently (same marginals, joint structure broken).
    If the real silhouette is close to the null values, the clusters reflect marginal shapes,
    not real groups.
    """
    rng = np.random.default_rng(seed)
    lo, hi = X.min(axis=0), X.max(axis=0)
    uniform = rng.uniform(lo, hi, size=X.shape)
    permuted = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])

    def _sil(data: np.ndarray) -> float:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(data)
        return float(
            silhouette_score(
                data, labels, sample_size=min(sample_size, len(data)), random_state=seed
            )
        )

    return {
        "k": k,
        "real_silhouette": _sil(X),
        "uniform_silhouette": _sil(uniform),
        "permuted_silhouette": _sil(permuted),
    }


def _tier(position: float) -> str:
    """Map a normalised rank in [0, 1] to low / mid / high."""
    if position >= 2 / 3:
        return "high"
    if position <= 1 / 3:
        return "low"
    return "mid"


def _cluster_names(medians: dict[int, tuple[float, float]]) -> dict[int, str]:
    """Readable names from the rank of each cluster's medians (deterministic rule).

    Each cluster gets a rank on median recency_days and on median monetary, normalised to a
    position in [0, 1] (k = 1 gives position 0, k = 2 gives 0 and 1, etc.). Position tiers:
    >= 2/3 high, <= 1/3 low, otherwise mid.
    - Recency: most recent cluster (lowest median recency_days) is position 1:
      high -> "Recent", mid -> "Mid-recency", low -> "Lapsed".
    - Monetary: highest median monetary is position 1: high -> "high-value",
      mid -> "mid-value", low -> "low-value".
    Name = "<recency> <monetary>"; duplicate names get " [cluster id]" appended.
    """
    ids = sorted(medians)
    k = len(ids)
    if k == 1:
        return {ids[0]: "All customers"}
    by_recency = sorted(ids, key=lambda c: medians[c][0], reverse=True)  # least recent first
    by_monetary = sorted(ids, key=lambda c: medians[c][1])  # lowest spend first
    rec_pos = {c: i / (k - 1) for i, c in enumerate(by_recency)}  # 1 = most recent
    mon_pos = {c: i / (k - 1) for i, c in enumerate(by_monetary)}  # 1 = highest spend
    rec_names = {"high": "Recent", "mid": "Mid-recency", "low": "Lapsed"}
    mon_names = {"high": "high-value", "mid": "mid-value", "low": "low-value"}
    names = {c: f"{rec_names[_tier(rec_pos[c])]} {mon_names[_tier(mon_pos[c])]}" for c in ids}
    counts: dict[str, int] = {}
    for name in names.values():
        counts[name] = counts.get(name, 0) + 1
    return {c: (f"{n} [{c}]" if counts[n] > 1 else n) for c, n in names.items()}


def profile_clusters(rfm: pl.DataFrame, labels: np.ndarray) -> pl.DataFrame:
    """Per-cluster size and medians of recency, frequency and monetary, plus an automatic name."""
    df = rfm.select(RFM_COLUMNS).with_columns(cluster=pl.Series(np.asarray(labels)).cast(pl.Int64))
    total = df.height
    profile = (
        df.group_by("cluster")
        .agg(
            size=pl.len(),
            median_recency_days=pl.col("recency_days").median(),
            median_frequency=pl.col("frequency").median(),
            median_monetary_usd=pl.col("monetary").median(),
            mean_monetary_usd=pl.col("monetary").mean(),
        )
        .sort("cluster")
    )
    medians = {
        int(r["cluster"]): (float(r["median_recency_days"]), float(r["median_monetary_usd"]))
        for r in profile.iter_rows(named=True)
    }
    names = _cluster_names(medians)
    return profile.with_columns(
        share_pct=pl.col("size") / total * 100,
        name=pl.col("cluster").replace_strict(names, return_dtype=pl.String),
    ).select(
        "cluster",
        "name",
        "size",
        "share_pct",
        "median_recency_days",
        "median_frequency",
        "median_monetary_usd",
        "mean_monetary_usd",
    )


def _quintile_score(col: str, ascending_good: bool = True) -> pl.Expr:
    """Quintile score 1..5 from a rank with method 'min'.

    Ties share the same rank, so columns with few distinct values (frequency) give fewer
    than five distinct scores. This is intentional: identical values must not be split.
    """
    n = pl.len()
    bucket = ((pl.col(col).rank(method="min") - 1) / n * 5).floor().cast(pl.Int32) + 1
    bucket = bucket.clip(1, 5)
    return bucket if ascending_good else 6 - bucket


def rfm_quantile_segments(rfm: pl.DataFrame) -> pl.DataFrame:
    """Classic rule-based RFM: R, F, M scores 1-5 and named segments (non-ML baseline).

    R = 5 for the most recent customers (lowest recency_days). F and M = 5 for the highest
    values. Segments are assigned by the first matching rule, in this order:
    Champions (R, F, M >= 4), Loyal (R, F >= 3), New (R >= 4, F <= 2),
    Potential loyalist (R >= 3), At risk (R <= 2, F >= 3), Hibernating (R, F <= 2).
    Only frequency and recency drive most rules; monetary only appears in Champions.
    """
    scored = rfm.with_columns(
        r_score=_quintile_score("recency_days", ascending_good=False),
        f_score=_quintile_score("frequency", ascending_good=True),
        m_score=_quintile_score("monetary", ascending_good=True),
    )
    r, f, m = pl.col("r_score"), pl.col("f_score"), pl.col("m_score")
    segment = (
        pl.when((r >= 4) & (f >= 4) & (m >= 4))
        .then(pl.lit("Champions"))
        .when((r >= 3) & (f >= 3))
        .then(pl.lit("Loyal"))
        .when((r >= 4) & (f <= 2))
        .then(pl.lit("New"))
        .when(r >= 3)
        .then(pl.lit("Potential loyalist"))
        .when((r <= 2) & (f >= 3))
        .then(pl.lit("At risk"))
        .otherwise(pl.lit("Hibernating"))
    )
    return scored.with_columns(
        rfm_code=pl.concat_str(r.cast(pl.String), f.cast(pl.String), m.cast(pl.String)),
        segment=segment,
    ).select(
        "customer_id",
        "recency_days",
        "frequency",
        "monetary",
        "r_score",
        "f_score",
        "m_score",
        "rfm_code",
        "segment",
    )


def compare_segmentations(labels_a: Sequence, labels_b: Sequence) -> dict:
    """Agreement (ARI) and a crosstab between two labelings of the same customers, in row order."""
    a = np.asarray(labels_a).astype(str)
    b = np.asarray(labels_b).astype(str)
    if len(a) != len(b):
        raise ValueError("labelings must cover the same customers")
    ari = float(adjusted_rand_score(a, b))
    a_vals = sorted(set(a))
    b_vals = sorted(set(b))
    table = {"labels_a": a_vals}
    for bv in b_vals:
        table[bv] = [int(np.sum((a == av) & (b == bv))) for av in a_vals]
    return {"ari": ari, "crosstab": pl.DataFrame(table)}
