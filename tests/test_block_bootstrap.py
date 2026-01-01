"""Tests for block bootstrap in StabilitySelector."""

import numpy as np
import pandas as pd
import pytest

from sift.stability import (
    StabilitySelector,
    stability_regression,
    stability_classif,
    _block_bootstrap_indices,
    _bootstrap_indices,
    _moving_block_sample,
    _circular_block_sample,
    _stationary_block_sample,
)


@pytest.fixture
def panel_data():
    """Generate panel data with groups and time."""
    rng = np.random.default_rng(42)
    n_groups = 20
    n_time = 30
    n = n_groups * n_time
    p = 15

    # Create groups and time
    groups = np.repeat(np.arange(n_groups), n_time)
    time = np.tile(np.arange(n_time), n_groups)

    # Generate features
    X = rng.standard_normal((n, p))
    # y is correlated with first 3 features
    y = X[:, 0] + 0.5 * X[:, 1] + 0.25 * X[:, 2] + 0.1 * rng.standard_normal(n)

    feature_names = [f"x{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=feature_names)
    return df, y, groups, time


@pytest.fixture
def panel_data_classification():
    """Generate panel classification data with groups and time."""
    rng = np.random.default_rng(42)
    n_groups = 20
    n_time = 30
    n = n_groups * n_time
    p = 15

    groups = np.repeat(np.arange(n_groups), n_time)
    time = np.tile(np.arange(n_time), n_groups)

    X = rng.standard_normal((n, p))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    feature_names = [f"x{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=feature_names)
    return df, y, groups, time


class TestBlockBootstrapHelpers:
    """Tests for block bootstrap helper functions."""

    def test_moving_block_sample(self):
        """Moving block sample should return indices."""
        rng = np.random.default_rng(42)
        sorted_idx = np.arange(30)
        result = _moving_block_sample(sorted_idx, block_size=5, n=30, rng=rng)
        assert len(result) >= 30
        assert all(idx in sorted_idx for idx in set(result))

    def test_circular_block_sample(self):
        """Circular block sample should wrap around."""
        rng = np.random.default_rng(42)
        sorted_idx = np.arange(30)
        result = _circular_block_sample(sorted_idx, block_size=5, n=30, rng=rng)
        assert len(result) >= 30
        assert all(idx in sorted_idx for idx in set(result))

    def test_stationary_block_sample(self):
        """Stationary block sample with geometric block length."""
        rng = np.random.default_rng(42)
        sorted_idx = np.arange(30)
        result = _stationary_block_sample(sorted_idx, mean_block_size=5, n=30, rng=rng)
        assert len(result) == 30
        assert all(idx in sorted_idx for idx in set(result))


class TestBlockBootstrapIndices:
    """Tests for block bootstrap index generation."""

    def test_block_bootstrap_generates_splits(self, panel_data):
        """Block bootstrap should generate train/val splits."""
        _, y, groups, time = panel_data
        n = len(y)

        splits = list(_block_bootstrap_indices(
            n=n,
            n_bootstrap=10,
            groups=groups,
            time=time,
            block_size="auto",
            block_method="moving",
            random_state=42,
        ))

        assert len(splits) == 10
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            # No overlap
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_block_bootstrap_circular_method(self, panel_data):
        """Circular block bootstrap should work."""
        _, y, groups, time = panel_data
        n = len(y)

        splits = list(_block_bootstrap_indices(
            n=n,
            n_bootstrap=5,
            groups=groups,
            time=time,
            block_method="circular",
            random_state=42,
        ))

        assert len(splits) == 5

    def test_block_bootstrap_stationary_method(self, panel_data):
        """Stationary block bootstrap should work."""
        _, y, groups, time = panel_data
        n = len(y)

        splits = list(_block_bootstrap_indices(
            n=n,
            n_bootstrap=5,
            groups=groups,
            time=time,
            block_method="stationary",
            random_state=42,
        ))

        assert len(splits) == 5

    def test_block_bootstrap_with_classification(self, panel_data_classification):
        """Block bootstrap should handle classification targets."""
        _, y, groups, time = panel_data_classification
        n = len(y)

        splits = list(_block_bootstrap_indices(
            n=n,
            n_bootstrap=5,
            groups=groups,
            time=time,
            y=y,
            task="classification",
            random_state=42,
        ))

        assert len(splits) == 5
        # Check all classes present in training
        for train_idx, val_idx in splits:
            assert len(np.unique(y[train_idx])) >= 2


class TestIidBootstrapIndices:
    """Tests for i.i.d. bootstrap index generation."""

    def test_iid_bootstrap_generates_splits(self):
        """I.i.d. bootstrap should generate train/val splits."""
        n = 100
        splits = list(_bootstrap_indices(
            n=n,
            n_bootstrap=10,
            sample_frac=0.5,
            random_state=42,
        ))

        assert len(splits) == 10
        for train_idx, val_idx in splits:
            assert len(train_idx) == 50
            assert len(val_idx) == 50
            # No overlap
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_iid_bootstrap_stratified_classification(self):
        """I.i.d. bootstrap should be stratified for classification."""
        n = 100
        y = np.array([0] * 70 + [1] * 30)

        splits = list(_bootstrap_indices(
            n=n,
            n_bootstrap=5,
            y=y,
            task="classification",
            sample_frac=0.5,
            random_state=42,
        ))

        assert len(splits) == 5
        for train_idx, _ in splits:
            # Both classes should be present
            assert len(np.unique(y[train_idx])) == 2


class TestStabilitySelectorBlockBootstrap:
    """Tests for StabilitySelector with block bootstrap."""

    def test_selector_uses_block_bootstrap_when_groups_time_provided(self, panel_data):
        """StabilitySelector should use block bootstrap with groups+time."""
        X, y, groups, time = panel_data

        selector = StabilitySelector(
            task='regression',
            n_bootstrap=5,
            block_method='moving',
            verbose=False,
        )
        selector.fit(X, y, groups=groups, time=time)

        assert hasattr(selector, 'selected_features_')
        assert len(selector.selected_features_) > 0

    def test_selector_uses_iid_without_groups_time(self, panel_data):
        """StabilitySelector should use i.i.d. bootstrap without groups."""
        X, y, _, _ = panel_data

        selector = StabilitySelector(
            task='regression',
            n_bootstrap=5,
            verbose=False,
        )
        selector.fit(X, y)  # No groups/time

        assert hasattr(selector, 'selected_features_')

    def test_selector_with_classification_block_bootstrap(self, panel_data_classification):
        """Classification StabilitySelector should work with block bootstrap."""
        X, y, groups, time = panel_data_classification

        selector = StabilitySelector(
            task='classification',
            n_bootstrap=5,
            block_method='circular',
            verbose=False,
        )
        selector.fit(X, y, groups=groups, time=time)

        assert hasattr(selector, 'selected_features_')


class TestConvenienceFunctionsBlockBootstrap:
    """Tests for convenience functions with block bootstrap."""

    def test_stability_regression_with_block_bootstrap(self, panel_data):
        """stability_regression should support block bootstrap."""
        X, y, groups, time = panel_data

        result = stability_regression(
            X, y, k=5,
            groups=groups,
            time=time,
            block_method='moving',
            n_bootstrap=5,
            verbose=False,
        )

        assert len(result) <= 5
        assert all(isinstance(name, str) for name in result)

    def test_stability_regression_without_block_bootstrap(self, panel_data):
        """stability_regression should work without groups/time."""
        X, y, _, _ = panel_data

        result = stability_regression(
            X, y, k=5,
            n_bootstrap=5,
            verbose=False,
        )

        assert len(result) <= 5

    def test_stability_classif_with_block_bootstrap(self, panel_data_classification):
        """stability_classif should support block bootstrap."""
        X, y, groups, time = panel_data_classification

        result = stability_classif(
            X, y, k=5,
            groups=groups,
            time=time,
            block_method='circular',
            n_bootstrap=5,
            verbose=False,
        )

        assert len(result) <= 5

    def test_stability_regression_with_sample_weight(self, panel_data):
        """stability_regression should accept sample_weight."""
        X, y, groups, time = panel_data
        n = len(y)
        weights = np.random.default_rng(0).random(n)

        result = stability_regression(
            X, y, k=5,
            sample_weight=weights,
            groups=groups,
            time=time,
            n_bootstrap=5,
            verbose=False,
        )

        assert len(result) <= 5
