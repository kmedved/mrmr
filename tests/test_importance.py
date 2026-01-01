"""Tests for permutation importance."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge, LogisticRegression

from sift.importance import (
    permutation_importance,
    _build_group_info,
    _permute,
    _permute_within_group,
    _permute_block,
    _permute_circular_shift,
    _score,
)


@pytest.fixture
def sample_regression_data():
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
def panel_regression_data():
    """Generate panel data with groups and time."""
    rng = np.random.default_rng(42)
    n_groups = 20
    n_time = 30
    n = n_groups * n_time
    p = 10

    groups = np.repeat(np.arange(n_groups), n_time)
    time = np.tile(np.arange(n_time), n_groups)

    X = rng.standard_normal((n, p))
    y = X[:, 0] + 0.5 * X[:, 1] + 0.25 * X[:, 2] + 0.1 * rng.standard_normal(n)

    feature_names = [f"x{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=feature_names)
    return df, y, groups, time


class TestBuildGroupInfo:
    """Tests for _build_group_info."""

    def test_returns_none_without_groups(self):
        """Should return None when groups is None."""
        assert _build_group_info(None, None) is None

    def test_builds_group_info(self):
        """Should build dict of sorted indices per group."""
        groups = np.array([0, 0, 1, 1, 2, 2])
        time = np.array([1, 0, 1, 0, 1, 0])

        info = _build_group_info(groups, time)

        assert len(info) == 3
        # Check indices are sorted by time within each group
        np.testing.assert_array_equal(info[0], [1, 0])  # sorted by time
        np.testing.assert_array_equal(info[1], [3, 2])
        np.testing.assert_array_equal(info[2], [5, 4])


class TestPermuteFunctions:
    """Tests for permutation helper functions."""

    def test_permute_within_group(self):
        """Within-group permutation should shuffle within groups."""
        rng = np.random.default_rng(42)
        x = np.arange(6, dtype=float)
        group_info = {0: np.array([0, 1]), 1: np.array([2, 3]), 2: np.array([4, 5])}

        result = _permute_within_group(x, group_info, rng)

        # Values should be permuted within groups
        assert set(result[:2]) == {0, 1}
        assert set(result[2:4]) == {2, 3}
        assert set(result[4:6]) == {4, 5}

    def test_permute_block(self):
        """Block permutation should shuffle blocks."""
        rng = np.random.default_rng(42)
        x = np.arange(30, dtype=float)
        group_info = {0: np.arange(30)}

        result = _permute_block(x, group_info, block_size=5, rng=rng)

        # Result should contain same values
        assert set(result) == set(x)
        assert len(result) == len(x)

    def test_permute_circular_shift(self):
        """Circular shift should rotate values."""
        rng = np.random.default_rng(42)
        x = np.arange(10, dtype=float)
        group_info = {0: np.arange(10)}

        result = _permute_circular_shift(x, group_info, rng)

        # Result should be a rotation of x
        assert len(result) == len(x)
        assert set(result) == set(x)

    def test_permute_global(self):
        """Global permutation should shuffle all values."""
        rng = np.random.default_rng(42)
        x = np.arange(10, dtype=float)

        result = _permute(x, None, "global", "auto", 42)

        assert set(result) == set(x)
        assert not np.array_equal(result, x)  # Should be shuffled


class TestScoreFunction:
    """Tests for _score function."""

    def test_neg_mse_scoring(self, sample_regression_data):
        """neg_mse scoring should work."""
        X, y = sample_regression_data
        model = Ridge(alpha=1.0)
        model.fit(X.values, y)
        w = np.ones(len(y))

        score = _score(model, X.values, y, w, "neg_mse")
        assert score < 0  # MSE is positive, so neg_mse is negative
        assert np.isfinite(score)

    def test_r2_scoring(self, sample_regression_data):
        """r2 scoring should work."""
        X, y = sample_regression_data
        model = Ridge(alpha=1.0)
        model.fit(X.values, y)
        w = np.ones(len(y))

        score = _score(model, X.values, y, w, "r2")
        assert -1 <= score <= 1
        assert score > 0.5  # Should be reasonably good fit

    def test_custom_scoring(self, sample_regression_data):
        """Custom callable scoring should work."""
        X, y = sample_regression_data
        model = Ridge(alpha=1.0)
        model.fit(X.values, y)
        w = np.ones(len(y))

        def custom_score(y_true, y_pred, weights):
            return -np.mean((y_true - y_pred) ** 2)

        score = _score(model, X.values, y, w, custom_score)
        assert np.isfinite(score)


class TestPermutationImportance:
    """Tests for permutation_importance function."""

    def test_global_permutation(self, sample_regression_data):
        """Global permutation importance should work."""
        X, y = sample_regression_data
        model = Ridge(alpha=1.0)
        model.fit(X.values, y)

        result = permutation_importance(
            model, X, y,
            permute_method="global",
            n_repeats=3,
            n_jobs=1,
            random_state=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "importance_mean" in result.columns
        assert "importance_std" in result.columns
        assert "baseline_score" in result.columns
        assert len(result) == X.shape[1]

    def test_auto_method_without_groups(self, sample_regression_data):
        """Auto method should use global without groups."""
        X, y = sample_regression_data
        model = Ridge(alpha=1.0)
        model.fit(X.values, y)

        result = permutation_importance(
            model, X, y,
            permute_method="auto",  # Should become "global"
            n_repeats=3,
            n_jobs=1,
            random_state=42,
        )

        assert len(result) == X.shape[1]

    def test_within_group_permutation(self, panel_regression_data):
        """Within-group permutation should work with groups."""
        X, y, groups, _ = panel_regression_data
        model = Ridge(alpha=1.0)
        model.fit(X.values, y)

        result = permutation_importance(
            model, X, y,
            groups=groups,
            permute_method="within_group",
            n_repeats=3,
            n_jobs=1,
            random_state=42,
        )

        assert len(result) == X.shape[1]

    def test_block_permutation(self, panel_regression_data):
        """Block permutation should work with groups+time."""
        X, y, groups, time = panel_regression_data
        model = Ridge(alpha=1.0)
        model.fit(X.values, y)

        result = permutation_importance(
            model, X, y,
            groups=groups,
            time=time,
            permute_method="block",
            n_repeats=3,
            n_jobs=1,
            random_state=42,
        )

        assert len(result) == X.shape[1]

    def test_circular_shift_permutation(self, panel_regression_data):
        """Circular shift permutation should work with groups+time."""
        X, y, groups, time = panel_regression_data
        model = Ridge(alpha=1.0)
        model.fit(X.values, y)

        result = permutation_importance(
            model, X, y,
            groups=groups,
            time=time,
            permute_method="circular_shift",
            n_repeats=3,
            n_jobs=1,
            random_state=42,
        )

        assert len(result) == X.shape[1]

    def test_auto_method_with_groups_time(self, panel_regression_data):
        """Auto method should use circular_shift with groups+time."""
        X, y, groups, time = panel_regression_data
        model = Ridge(alpha=1.0)
        model.fit(X.values, y)

        result = permutation_importance(
            model, X, y,
            groups=groups,
            time=time,
            permute_method="auto",  # Should become "circular_shift"
            n_repeats=3,
            n_jobs=1,
            random_state=42,
        )

        assert len(result) == X.shape[1]

    def test_with_sample_weight(self, sample_regression_data):
        """Permutation importance should accept sample weights."""
        X, y = sample_regression_data
        n = len(y)
        model = Ridge(alpha=1.0)
        model.fit(X.values, y)
        weights = np.random.default_rng(0).random(n)

        result = permutation_importance(
            model, X, y,
            sample_weight=weights,
            n_repeats=3,
            n_jobs=1,
            random_state=42,
        )

        assert len(result) == X.shape[1]

    def test_important_features_ranked_higher(self, sample_regression_data):
        """True important features should be ranked higher."""
        X, y = sample_regression_data
        model = Ridge(alpha=1.0)
        model.fit(X.values, y)

        result = permutation_importance(
            model, X, y,
            n_repeats=10,
            n_jobs=1,
            random_state=42,
        )

        # x0, x1, x2 are the true important features
        top_3 = result.head(3)["feature"].tolist()
        assert "x0" in top_3
        assert "x1" in top_3

    def test_requires_groups_for_within_group(self, sample_regression_data):
        """within_group method should require groups."""
        X, y = sample_regression_data
        model = Ridge(alpha=1.0)
        model.fit(X.values, y)

        with pytest.raises(ValueError, match="requires groups"):
            permutation_importance(
                model, X, y,
                permute_method="within_group",
                n_repeats=3,
                random_state=42,
            )

    def test_requires_time_for_block(self, sample_regression_data):
        """block method should require time."""
        X, y = sample_regression_data
        n = len(y)
        groups = np.repeat(np.arange(10), n // 10)
        model = Ridge(alpha=1.0)
        model.fit(X.values, y)

        with pytest.raises(ValueError, match="requires time"):
            permutation_importance(
                model, X, y,
                groups=groups,
                permute_method="block",
                n_repeats=3,
                random_state=42,
            )
