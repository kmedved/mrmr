"""Gaussian copula transforms and caching for fast selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numba import njit
from scipy.special import ndtri
from scipy.stats import rankdata


@dataclass
class FeatureCache:
    """Cached feature data for multi-target selection."""

    Z: np.ndarray
    Rxx: Optional[np.ndarray]
    valid_cols: np.ndarray
    row_idx: np.ndarray
    feature_names: Optional[List[str]] = None
    sample_weight: Optional[np.ndarray] = None


def build_cache(
    X,
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    compute_Rxx: bool = False,
    min_std: float = 1e-12,
    sample_weight: Optional[np.ndarray] = None,
) -> FeatureCache:
    """
    Build feature cache for multi-target selection.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix.
    subsample : int, optional
        Maximum number of rows to use. Default 50_000.
    random_state : int
        Random seed for subsampling.
    compute_Rxx : bool
        Whether to compute full correlation matrix.
    min_std : float
        Minimum std to keep a feature.
    sample_weight : ndarray, optional
        Sample weights. If provided, uses weighted rank-Gaussian transform
        and weighted correlation matrix.

    Returns
    -------
    FeatureCache
        Cached transformed features.
    """
    from sift._preprocess import extract_feature_names, to_numpy

    feature_names = extract_feature_names(X)
    if hasattr(X, "select_dtypes"):
        non_numeric = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if non_numeric:
            sample = non_numeric[:5]
            suffix = "..." if len(non_numeric) > 5 else ""
            raise ValueError(
                f"Non-numeric columns found: {sample}{suffix}. "
                "Encode categorical columns before using gaussian estimator."
            )
    X_arr = to_numpy(X, dtype=np.float64)
    n, p = X_arr.shape
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(p)]

    # Handle sample weights
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64)
        if len(w) != n:
            raise ValueError(f"sample_weight has {len(w)} elements but X has {n} rows")
    else:
        w = None

    rng = np.random.default_rng(random_state)
    if subsample is not None and n > subsample:
        row_idx = rng.choice(n, size=subsample, replace=False)
    else:
        row_idx = np.arange(n)

    Xs = X_arr[row_idx]
    ws = w[row_idx] if w is not None else None

    Xs = np.where(np.isfinite(Xs), Xs, np.nan)
    col_means = np.nanmean(Xs, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(Xs)
    Xs[nan_mask] = col_means[np.where(nan_mask)[1]]

    stds = np.std(Xs, axis=0)
    valid_mask = stds > min_std
    valid_cols = np.where(valid_mask)[0]
    Xs = Xs[:, valid_mask]
    if Xs.shape[1] == 0:
        raise ValueError("All features were filtered out (constant or invalid). Cannot build cache.")

    # Use weighted or unweighted rank-Gaussian transform
    if ws is not None:
        Z = rank_gauss_2d_weighted(Xs, ws)
        Rxx = correlation_matrix_weighted(Z, ws) if compute_Rxx else None
    else:
        Z = rank_gauss_2d(Xs)
        Rxx = correlation_matrix_fast(Z) if compute_Rxx else None

    return FeatureCache(
        Z=Z.astype(np.float32),
        Rxx=Rxx.astype(np.float32) if Rxx is not None else None,
        valid_cols=valid_cols,
        row_idx=row_idx,
        feature_names=feature_names,
        sample_weight=ws.astype(np.float32) if ws is not None else None,
    )


def rank_gauss_1d(x: np.ndarray) -> np.ndarray:
    """Rank-based Gaussian transform for 1D array."""
    mask = np.isfinite(x)
    m = mask.sum()
    if m <= 1:
        return np.zeros_like(x, dtype=np.float32)

    ranks = rankdata(x[mask], method="average")
    u = ranks / (m + 1.0)
    z = ndtri(u).astype(np.float64)
    z -= z.mean()
    std = z.std(ddof=1)
    if std < 1e-12:
        z[:] = 0.0
    else:
        z /= std

    out = np.zeros_like(x, dtype=np.float32)
    out[mask] = z.astype(np.float32)
    return out


def rank_gauss_2d(X: np.ndarray) -> np.ndarray:
    """Apply rank-Gaussian transform to each column."""
    n, p = X.shape
    Z = np.empty((n, p), dtype=np.float32)
    for j in range(p):
        Z[:, j] = rank_gauss_1d(X[:, j])
    return Z


def rank_gauss_1d_weighted(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted rank-based Gaussian transform for 1D array.

    Weighted ranks: rank_i = Σ_{j: x_j < x_i} w_j + 0.5 * w_i
    Normalize: u_i = rank_i / Σw
    Apply inverse normal CDF.

    Parameters
    ----------
    x : ndarray of shape (n,)
        Input values.
    w : ndarray of shape (n,)
        Sample weights.

    Returns
    -------
    ndarray of shape (n,)
        Transformed values.
    """
    mask = np.isfinite(x)
    m = mask.sum()
    if m <= 1:
        return np.zeros_like(x, dtype=np.float32)

    x_valid = x[mask]
    w_valid = w[mask]

    # Sort by x values
    order = np.argsort(x_valid)
    x_sorted = x_valid[order]
    w_sorted = w_valid[order]

    # Compute weighted ranks
    # For ties, use average weighted rank
    w_sum = w_sorted.sum()

    # Cumulative weights
    cumw = np.cumsum(w_sorted)

    # Weighted rank for each position (midpoint formula)
    # rank_i = cumsum(w) - 0.5 * w_i  (for sorted data)
    w_ranks = cumw - 0.5 * w_sorted

    # Handle ties: average weighted ranks for tied values
    unique_vals, inverse_idx, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if len(unique_vals) < len(x_sorted):
        # There are ties - compute average weighted rank for each group
        avg_ranks = np.zeros(len(unique_vals))
        for i, uv in enumerate(unique_vals):
            tie_mask = x_sorted == uv
            avg_ranks[i] = w_ranks[tie_mask].mean()
        w_ranks = avg_ranks[inverse_idx]

    # Normalize to (0, 1) - use (rank) / (sum of weights) with small buffer
    u = np.clip(w_ranks / w_sum, 1e-6, 1 - 1e-6)

    # Apply inverse normal CDF
    z = ndtri(u).astype(np.float64)

    # Weighted standardization
    z_wmean = np.average(z, weights=w_sorted)
    z_centered = z - z_wmean
    z_wvar = np.average(z_centered ** 2, weights=w_sorted)
    z_wstd = np.sqrt(z_wvar)

    if z_wstd < 1e-12:
        z_standardized = np.zeros_like(z)
    else:
        z_standardized = z_centered / z_wstd

    # Restore original order
    result_valid = np.empty(m, dtype=np.float64)
    result_valid[order] = z_standardized

    out = np.zeros_like(x, dtype=np.float32)
    out[mask] = result_valid.astype(np.float32)
    return out


def rank_gauss_2d_weighted(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply weighted rank-Gaussian transform to each column."""
    n, p = X.shape
    Z = np.empty((n, p), dtype=np.float32)
    for j in range(p):
        Z[:, j] = rank_gauss_1d_weighted(X[:, j], w)
    return Z


@njit(cache=True)
def correlation_matrix_weighted(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted correlation matrix from standardized data.

    R_jk = Σ w_i * z_ij * z_ik / Σw

    Parameters
    ----------
    Z : ndarray of shape (n, p)
        Standardized feature matrix.
    w : ndarray of shape (n,)
        Sample weights.

    Returns
    -------
    ndarray of shape (p, p)
        Weighted correlation matrix.
    """
    n, p = Z.shape

    # Sum of weights
    w_sum = 0.0
    for i in range(n):
        w_sum += w[i]

    R = np.empty((p, p), dtype=np.float64)

    for j in range(p):
        for k in range(j, p):
            cov = 0.0
            for i in range(n):
                cov += w[i] * Z[i, j] * Z[i, k]
            cov /= w_sum

            if cov > 0.999999:
                cov = 0.999999
            elif cov < -0.999999:
                cov = -0.999999

            R[j, k] = cov
            R[k, j] = cov

    # Ensure diagonal is 1
    for i in range(p):
        R[i, i] = 1.0

    return R


@njit(cache=True)
def corr_with_vector_weighted(
    Z: np.ndarray, zy: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """
    Weighted correlation of each column of Z with vector zy.

    Parameters
    ----------
    Z : ndarray of shape (n, p)
        Feature matrix.
    zy : ndarray of shape (n,)
        Target vector.
    w : ndarray of shape (n,)
        Sample weights.

    Returns
    -------
    ndarray of shape (p,)
        Weighted correlations.
    """
    n, p = Z.shape

    w_sum = 0.0
    for i in range(n):
        w_sum += w[i]

    r = np.empty(p, dtype=np.float32)
    for j in range(p):
        cov = 0.0
        for i in range(n):
            cov += w[i] * Z[i, j] * zy[i]
        r[j] = cov / w_sum

    return np.clip(r, -0.999999, 0.999999)


@njit(cache=True)
def correlation_matrix_fast(Z: np.ndarray) -> np.ndarray:
    """Correlation matrix from standardized data."""
    n, p = Z.shape
    R = (Z.T @ Z) / max(n - 1, 1)
    for i in range(p):
        for j in range(p):
            if R[i, j] > 0.999999:
                R[i, j] = 0.999999
            elif R[i, j] < -0.999999:
                R[i, j] = -0.999999
        R[i, i] = 1.0
    return R


@njit(cache=True)
def corr_with_vector(Z: np.ndarray, zy: np.ndarray) -> np.ndarray:
    """Correlation of each column of Z with vector zy."""
    n, p = Z.shape
    r = np.empty(p, dtype=np.float32)
    for j in range(p):
        r[j] = np.sum(Z[:, j] * zy) / max(n - 1, 1)
    return np.clip(r, -0.999999, 0.999999)


@njit(cache=True)
def gaussian_mi_from_corr(r: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Gaussian MI approximation: I(X;Y) = -0.5 * log(1 - r²)."""
    r2 = np.clip(r * r, 0.0, 1.0 - eps)
    return -0.5 * np.log(1.0 - r2)


def greedy_corr_prune(
    candidates: np.ndarray,
    Rxx: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.95,
) -> np.ndarray:
    """Prune candidates with high correlation to higher-scoring features."""
    if len(candidates) == 0:
        return candidates

    order = candidates[np.argsort(-scores[candidates])]
    keep = []
    active = np.ones(len(order), dtype=bool)

    for i, fi in enumerate(order):
        if not active[i]:
            continue
        keep.append(fi)

        for j in range(i + 1, len(order)):
            if active[j]:
                fj = order[j]
                if np.abs(Rxx[fi, fj]) >= threshold:
                    active[j] = False

    return np.array(keep, dtype=np.int64)
