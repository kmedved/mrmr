"""Relevance scoring: feature-target association."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def f_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    F-statistic for regression (vectorized).

    Computed from Pearson correlation: F = r² / (1 - r²) * (n - 2)
    """
    n, p = X.shape
    y_mean = 0.0
    for i in range(n):
        y_mean += y[i]
    y_mean /= n

    y_ss = 0.0
    for i in range(n):
        y_ss += (y[i] - y_mean) ** 2

    scores = np.empty(p, dtype=np.float64)
    for j in prange(p):
        x_mean = 0.0
        for i in range(n):
            x_mean += X[i, j]
        x_mean /= n

        x_ss = 0.0
        xy = 0.0
        for i in range(n):
            xc = X[i, j] - x_mean
            x_ss += xc * xc
            xy += xc * (y[i] - y_mean)

        if x_ss < 1e-12 or y_ss < 1e-12:
            scores[j] = 0.0
        else:
            r = xy / np.sqrt(x_ss * y_ss)
            r2 = r * r
            if r2 > 0.99999:
                r2 = 0.99999
            scores[j] = r2 / (1.0 - r2) * (n - 2)

    return scores


@njit(cache=True, parallel=True)
def f_classif(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    F-statistic for classification (one-way ANOVA).

    Computes between-group vs within-group variance ratio.
    """
    n, p = X.shape
    n_classes = int(y.max()) + 1

    class_counts = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        class_counts[int(y[i])] += 1

    scores = np.empty(p, dtype=np.float64)

    for j in prange(p):
        x_mean = 0.0
        for i in range(n):
            x_mean += X[i, j]
        x_mean /= n

        class_sums = np.zeros(n_classes, dtype=np.float64)
        class_sq_sums = np.zeros(n_classes, dtype=np.float64)

        for i in range(n):
            val = X[i, j]
            c_idx = int(y[i])
            class_sums[c_idx] += val
            class_sq_sums[c_idx] += val * val

        ss_between = 0.0
        ss_within = 0.0

        for c_idx in range(n_classes):
            n_c = class_counts[c_idx]
            if n_c == 0:
                continue
            mean_c = class_sums[c_idx] / n_c
            ss_between += n_c * (mean_c - x_mean) ** 2
            ss_within += class_sq_sums[c_idx] - n_c * mean_c * mean_c

        df_between = n_classes - 1
        df_within = n - n_classes

        if df_within <= 0 or df_between <= 0:
            scores[j] = 0.0
        elif ss_within < 1e-12:
            if ss_between > 1e-12:
                scores[j] = 1.0e15
            else:
                scores[j] = 0.0
        else:
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within
            scores[j] = ms_between / ms_within

    return scores


def ks_classif(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kolmogorov-Smirnov statistic for classification."""
    from scipy.stats import ks_2samp

    classes = np.unique(y)
    n, p = X.shape
    scores = np.zeros(p, dtype=np.float64)

    for j in range(p):
        x = X[:, j]
        ks_sum = 0.0
        count = 0
        for c in classes:
            mask = y == c
            if mask.sum() < 2:
                continue
            stat, _ = ks_2samp(x[mask], x[~mask])
            ks_sum += stat
            count += 1
        scores[j] = ks_sum / max(count, 1)

    return scores


def rf_regression(X: np.ndarray, y: np.ndarray, max_depth: int = 5) -> np.ndarray:
    """Random forest importance for regression."""
    from sklearn.ensemble import RandomForestRegressor

    X_filled = np.nan_to_num(X, nan=np.nanmin(X) - 1)
    rf = RandomForestRegressor(max_depth=max_depth, n_estimators=100, random_state=0)
    rf.fit(X_filled, y)
    return rf.feature_importances_


def rf_classif(X: np.ndarray, y: np.ndarray, max_depth: int = 5) -> np.ndarray:
    """Random forest importance for classification."""
    from sklearn.ensemble import RandomForestClassifier

    X_filled = np.nan_to_num(X, nan=np.nanmin(X) - 1)
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=100, random_state=0)
    rf.fit(X_filled, y)
    return rf.feature_importances_


# =============================================================================
# Weighted relevance scoring
# =============================================================================


@njit(cache=True, parallel=True)
def f_regression_weighted(
    X: np.ndarray, y: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """
    Weighted F-statistic for regression.

    Weighted correlation: r = Σw*(x-μx)*(y-μy) / sqrt(Σw*(x-μx)² * Σw*(y-μy)²)
    where μx = Σw*x / Σw (weighted mean).
    F-stat from weighted r² as before.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Feature matrix.
    y : ndarray of shape (n,)
        Target variable.
    w : ndarray of shape (n,)
        Sample weights (must be positive).

    Returns
    -------
    ndarray of shape (p,)
        F-statistics for each feature.
    """
    n, p = X.shape

    # Compute sum of weights
    w_sum = 0.0
    for i in range(n):
        w_sum += w[i]

    # Weighted mean and variance of y
    y_wmean = 0.0
    for i in range(n):
        y_wmean += w[i] * y[i]
    y_wmean /= w_sum

    y_wss = 0.0
    for i in range(n):
        y_wss += w[i] * (y[i] - y_wmean) ** 2

    scores = np.empty(p, dtype=np.float64)

    for j in prange(p):
        # Weighted mean of x_j
        x_wmean = 0.0
        for i in range(n):
            x_wmean += w[i] * X[i, j]
        x_wmean /= w_sum

        # Weighted variance and covariance
        x_wss = 0.0
        xy_wcov = 0.0
        for i in range(n):
            xc = X[i, j] - x_wmean
            x_wss += w[i] * xc * xc
            xy_wcov += w[i] * xc * (y[i] - y_wmean)

        if x_wss < 1e-12 or y_wss < 1e-12:
            scores[j] = 0.0
        else:
            r = xy_wcov / np.sqrt(x_wss * y_wss)
            r2 = r * r
            if r2 > 0.99999:
                r2 = 0.99999
            # Effective degrees of freedom approximation
            # Using sum of weights as effective n
            n_eff = w_sum * w_sum
            w_sq_sum = 0.0
            for i in range(n):
                w_sq_sum += w[i] * w[i]
            n_eff = n_eff / w_sq_sum if w_sq_sum > 0 else n
            scores[j] = r2 / (1.0 - r2) * max(n_eff - 2, 1)

    return scores


@njit(cache=True, parallel=True)
def f_classif_weighted(
    X: np.ndarray, y: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """
    Weighted F-statistic for classification (one-way ANOVA).

    Computes weighted between-group vs within-group variance ratio.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Feature matrix.
    y : ndarray of shape (n,)
        Class labels (integers 0, 1, ..., n_classes-1).
    w : ndarray of shape (n,)
        Sample weights (must be positive).

    Returns
    -------
    ndarray of shape (p,)
        F-statistics for each feature.
    """
    n, p = X.shape
    n_classes = int(y.max()) + 1

    # Weighted class counts (sum of weights per class)
    class_w_sum = np.zeros(n_classes, dtype=np.float64)
    for i in range(n):
        class_w_sum[int(y[i])] += w[i]

    total_w = 0.0
    for c in range(n_classes):
        total_w += class_w_sum[c]

    scores = np.empty(p, dtype=np.float64)

    for j in prange(p):
        # Overall weighted mean
        x_wmean = 0.0
        for i in range(n):
            x_wmean += w[i] * X[i, j]
        x_wmean /= total_w

        # Per-class weighted sums and squared sums
        class_w_sums = np.zeros(n_classes, dtype=np.float64)
        class_w_sq_sums = np.zeros(n_classes, dtype=np.float64)

        for i in range(n):
            val = X[i, j]
            c_idx = int(y[i])
            class_w_sums[c_idx] += w[i] * val
            class_w_sq_sums[c_idx] += w[i] * val * val

        ss_between = 0.0
        ss_within = 0.0

        for c_idx in range(n_classes):
            w_c = class_w_sum[c_idx]
            if w_c < 1e-12:
                continue
            mean_c = class_w_sums[c_idx] / w_c
            ss_between += w_c * (mean_c - x_wmean) ** 2
            ss_within += class_w_sq_sums[c_idx] - w_c * mean_c * mean_c

        # Count active classes
        n_active = 0
        for c_idx in range(n_classes):
            if class_w_sum[c_idx] > 1e-12:
                n_active += 1

        df_between = n_active - 1
        # Effective df_within using Satterthwaite approximation
        df_within = total_w - n_active  # Simplified approximation

        if df_within <= 0 or df_between <= 0:
            scores[j] = 0.0
        elif ss_within < 1e-12:
            if ss_between > 1e-12:
                scores[j] = 1.0e15
            else:
                scores[j] = 0.0
        else:
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within
            scores[j] = ms_between / ms_within

    return scores
