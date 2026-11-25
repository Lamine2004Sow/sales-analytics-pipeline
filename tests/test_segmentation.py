"""Fast tests for the RFM segmentation helpers (synthetic data only)."""

from datetime import date

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_blobs

from retail_sales_pipeline.segmentation import (
    compare_segmentations,
    compute_rfm,
    evaluate_k,
    null_baseline,
    profile_clusters,
    rfm_features,
    rfm_quantile_segments,
    stability,
)


@pytest.fixture(scope="module")
def blobs():
    X, _ = make_blobs(n_samples=600, centers=3, cluster_std=0.5, random_state=0)
    return X


def test_compute_rfm_exact_values_and_excludes_non_realised():
    clean = pl.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "customer_id": ["A", "A", "A", "B", "C"],
            "transaction_date": [
                date(2024, 1, 10),
                date(2024, 3, 1),
                date(2024, 5, 1),  # not realised: must be ignored
                date(2024, 2, 1),
                date(2024, 4, 1),  # C has only a non-realised order: absent from RFM
            ],
            "revenue_usd": [100.0, 50.0, 999.0, 30.0, 500.0],
            "is_realised": [True, True, False, True, False],
        }
    )
    rfm = compute_rfm(clean).sort("customer_id")

    # Snapshot = max transaction_date (2024-05-01) + 1 day = 2024-05-02.
    assert rfm["customer_id"].to_list() == ["A", "B"]
    assert rfm["recency_days"].to_list() == [62, 91]  # A: last realised 2024-03-01
    assert rfm["frequency"].to_list() == [2, 1]
    assert rfm["monetary"].to_list() == [150.0, 30.0]
    assert rfm["first_order"].to_list() == [date(2024, 1, 10), date(2024, 2, 1)]
    assert rfm["last_order"].to_list() == [date(2024, 3, 1), date(2024, 2, 1)]


def test_rfm_features_standardised_and_returns_scaler():
    rfm = pl.DataFrame(
        {"recency_days": [1, 10, 100], "frequency": [1, 2, 5], "monetary": [10.0, 20.0, 300.0]}
    )
    X, scaler = rfm_features(rfm)
    assert X.shape == (3, 3)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(X.std(axis=0), 1)
    assert scaler.n_features_in_ == 3


def test_evaluate_k_best_silhouette_at_true_k(blobs):
    table = evaluate_k(blobs, k_range=range(2, 7), sample_size=600)
    assert table.columns == ["k", "inertia", "silhouette", "davies_bouldin", "calinski_harabasz"]
    best = table.sort("silhouette", descending=True)["k"][0]
    assert best == 3


def test_stability_high_on_blobs(blobs):
    result = stability(blobs, k=3, n_runs=5)
    assert result["subsample_ari_mean"] > 0.9
    assert result["seed_ari_mean"] > 0.9


def test_null_baseline_real_above_structure_free(blobs):
    result = null_baseline(blobs, k=3)
    assert result["real_silhouette"] > result["uniform_silhouette"] + 0.1
    assert result["real_silhouette"] > result["permuted_silhouette"] + 0.1


def test_profile_clusters_shares_sum_to_100():
    rng = np.random.default_rng(0)
    n = 300
    rfm = pl.DataFrame(
        {
            "customer_id": [str(i) for i in range(n)],
            "recency_days": rng.integers(1, 800, n),
            "frequency": rng.integers(1, 6, n),
            "monetary": rng.uniform(10, 1000, n),
        }
    )
    labels = rng.integers(0, 4, n)
    profile = profile_clusters(rfm, labels)
    assert profile["size"].sum() == n
    assert profile["share_pct"].sum() == pytest.approx(100)
    assert profile["name"].n_unique() == profile.height


def test_rfm_quantile_segments_one_per_customer():
    rng = np.random.default_rng(1)
    n = 200
    rfm = pl.DataFrame(
        {
            "customer_id": [str(i) for i in range(n)],
            "recency_days": rng.integers(1, 800, n),
            "frequency": rng.integers(1, 4, n),
            "monetary": rng.uniform(10, 1000, n),
        }
    )
    seg = rfm_quantile_segments(rfm)
    assert seg.height == n
    assert seg["segment"].null_count() == 0
    assert seg["r_score"].min() >= 1 and seg["r_score"].max() <= 5
    assert seg["customer_id"].n_unique() == n


def test_compare_segmentations_identical_and_crosstab():
    a = ["x", "x", "y", "y", "z"]
    result = compare_segmentations(a, [1, 1, 2, 2, 3])
    assert result["ari"] == pytest.approx(1.0)
    assert result["crosstab"].height == 3
