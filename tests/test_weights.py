"""Tests for sample weights in filter methods."""

import numpy as np
import pandas as pd
import pytest

from sift import select_mrmr, select_jmi, select_jmim, select_cefsplus, build_cache


@pytest.fixture
def sample_data():
    """Generate sample regression data."""
    rng = np.random.default_rng(42)
    n, p = 200, 10
    X = rng.standard_normal((n, p))
    # y is correlated with first 3 features
    y = X[:, 0] + 0.5 * X[:, 1] + 0.25 * X[:, 2] + 0.1 * rng.standard_normal(n)
    feature_names = [f"x{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=feature_names)
    return df, y


@pytest.fixture
def sample_data_classification():
    """Generate sample classification data."""
    rng = np.random.default_rng(42)
    n, p = 200, 10
    X = rng.standard_normal((n, p))
    # y is determined by first 2 features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    feature_names = [f"x{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=feature_names)
    return df, y


class TestWeightedMrmr:
    """Tests for weighted mRMR selection."""

    def test_mrmr_with_uniform_weights_matches_unweighted(self, sample_data):
        """Uniform weights should give same result as no weights."""
        X, y = sample_data
        n = len(y)

        # Without weights
        result1 = select_mrmr(X, y, k=3, task="regression", verbose=False)

        # With uniform weights
        weights = np.ones(n)
        result2 = select_mrmr(X, y, k=3, task="regression", sample_weight=weights, verbose=False)

        assert result1 == result2

    def test_mrmr_accepts_sample_weight(self, sample_data):
        """mRMR should accept sample_weight without error."""
        X, y = sample_data
        n = len(y)
        weights = np.random.default_rng(0).random(n)

        result = select_mrmr(X, y, k=3, task="regression", sample_weight=weights, verbose=False)
        assert len(result) == 3
        assert all(isinstance(name, str) for name in result)

    def test_mrmr_gaussian_with_weights(self, sample_data):
        """Gaussian mRMR should accept weights."""
        X, y = sample_data
        n = len(y)
        weights = np.random.default_rng(0).random(n)

        result = select_mrmr(
            X, y, k=3, task="regression",
            estimator="gaussian", sample_weight=weights, verbose=False
        )
        assert len(result) == 3

    def test_mrmr_classification_with_weights(self, sample_data_classification):
        """Classification mRMR should accept weights."""
        X, y = sample_data_classification
        n = len(y)
        weights = np.random.default_rng(0).random(n)

        result = select_mrmr(
            X, y, k=3, task="classification",
            sample_weight=weights, verbose=False
        )
        assert len(result) == 3


class TestWeightedJmi:
    """Tests for weighted JMI selection."""

    def test_jmi_accepts_sample_weight(self, sample_data):
        """JMI should accept sample_weight without error."""
        X, y = sample_data
        n = len(y)
        weights = np.random.default_rng(0).random(n)

        result = select_jmi(X, y, k=3, task="regression", sample_weight=weights, verbose=False)
        assert len(result) == 3

    def test_jmi_gaussian_with_weights(self, sample_data):
        """Gaussian JMI should accept weights."""
        X, y = sample_data
        n = len(y)
        weights = np.random.default_rng(0).random(n)

        result = select_jmi(
            X, y, k=3, task="regression",
            estimator="gaussian", sample_weight=weights, verbose=False
        )
        assert len(result) == 3


class TestWeightedJmim:
    """Tests for weighted JMIM selection."""

    def test_jmim_accepts_sample_weight(self, sample_data):
        """JMIM should accept sample_weight without error."""
        X, y = sample_data
        n = len(y)
        weights = np.random.default_rng(0).random(n)

        result = select_jmim(X, y, k=3, task="regression", sample_weight=weights, verbose=False)
        assert len(result) == 3


class TestWeightedCefsplus:
    """Tests for weighted CEFS+ selection."""

    def test_cefsplus_accepts_sample_weight(self, sample_data):
        """CEFS+ should accept sample_weight without error."""
        X, y = sample_data
        n = len(y)
        weights = np.random.default_rng(0).random(n)

        result = select_cefsplus(X, y, k=3, sample_weight=weights, verbose=False)
        assert len(result) == 3


class TestWeightedBuildCache:
    """Tests for weighted cache building."""

    def test_build_cache_with_weights(self, sample_data):
        """build_cache should accept sample_weight."""
        X, y = sample_data
        n = len(y)
        weights = np.random.default_rng(0).random(n)

        cache = build_cache(X, sample_weight=weights)
        assert cache.sample_weight is not None
        assert cache.Z is not None

    def test_build_cache_uniform_weights_stored(self, sample_data):
        """build_cache should store weights even when uniform."""
        X, y = sample_data

        cache = build_cache(X)
        assert cache.sample_weight is not None
        # Should be approximately uniform
        assert np.allclose(cache.sample_weight, 1.0)


class TestWeightedRelevance:
    """Tests for weighted relevance scoring."""

    def test_f_regression_weighted(self):
        """Weighted F-regression should work correctly."""
        from sift.estimators.relevance import f_regression_weighted, _ensure_weights

        rng = np.random.default_rng(42)
        n, p = 100, 5
        X = rng.standard_normal((n, p))
        y = X[:, 0] + 0.1 * rng.standard_normal(n)
        w = _ensure_weights(None, n)

        scores = f_regression_weighted(X, y.astype(np.float64), w)
        assert len(scores) == p
        # Feature 0 should have highest score
        assert np.argmax(scores) == 0

    def test_f_classif_weighted(self):
        """Weighted F-classif should work correctly."""
        from sift.estimators.relevance import f_classif_weighted, _ensure_weights

        rng = np.random.default_rng(42)
        n, p = 100, 5
        X = rng.standard_normal((n, p))
        y = (X[:, 0] > 0).astype(np.float64)
        w = _ensure_weights(None, n)

        scores = f_classif_weighted(X, y, w)
        assert len(scores) == p
        # Feature 0 should have highest score
        assert np.argmax(scores) == 0
