"""Time-series-aware permutation importance."""

from __future__ import annotations

from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# =============================================================================
# Permutation Strategies
# =============================================================================


def _permute_global(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Standard shuffle."""
    return rng.permutation(x)


def _permute_within_group(
    x: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle within each group independently."""
    result = x.copy()
    for g in np.unique(groups):
        mask = groups == g
        result[mask] = rng.permutation(x[mask])
    return result


def _permute_block(
    x: np.ndarray,
    groups: np.ndarray,
    time: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Permute contiguous blocks within each group.

    Preserves local autocorrelation while breaking feature-target relationship.
    """
    result = x.copy()
    for g in np.unique(groups):
        mask = groups == g
        idx = np.where(mask)[0]
        order = np.argsort(time[idx])
        sorted_idx = idx[order]

        n = len(sorted_idx)
        n_blocks = max(1, n // block_size)
        blocks = np.array_split(sorted_idx, n_blocks)

        # Shuffle blocks
        rng.shuffle(blocks)
        new_order = np.concatenate(blocks)
        result[sorted_idx] = x[new_order]

    return result


def _permute_circular_shift(
    x: np.ndarray,
    groups: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Circular shift within each group by random offset.

    Best preserves autocorrelation structure.
    """
    result = x.copy()
    for g in np.unique(groups):
        mask = groups == g
        idx = np.where(mask)[0]
        order = np.argsort(time[idx])
        sorted_idx = idx[order]

        n = len(sorted_idx)
        shift = rng.integers(1, max(2, n))
        result[sorted_idx] = np.roll(x[sorted_idx], shift)

    return result


# =============================================================================
# Weighted Scoring
# =============================================================================


def _score(
    model,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    scoring: Optional[Union[str, Callable]],
) -> float:
    """Compute score, optionally weighted."""
    y_pred = model.predict(X)

    if callable(scoring):
        return scoring(y, y_pred, sample_weight=sample_weight)

    if scoring in (None, "neg_mean_squared_error"):
        if sample_weight is not None:
            return -np.average((y - y_pred) ** 2, weights=sample_weight)
        return -np.mean((y - y_pred) ** 2)

    if scoring == "neg_mean_absolute_error":
        if sample_weight is not None:
            return -np.average(np.abs(y - y_pred), weights=sample_weight)
        return -np.mean(np.abs(y - y_pred))

    if scoring == "r2":
        if sample_weight is not None:
            ss_res = np.average((y - y_pred) ** 2, weights=sample_weight)
            y_mean = np.average(y, weights=sample_weight)
            ss_tot = np.average((y - y_mean) ** 2, weights=sample_weight)
        else:
            ss_res = np.mean((y - y_pred) ** 2)
            ss_tot = np.var(y)
        return 1 - ss_res / (ss_tot + 1e-10)

    if scoring == "accuracy":
        # For classification
        correct = (y_pred.round() == y).astype(float)
        if sample_weight is not None:
            return np.average(correct, weights=sample_weight)
        return np.mean(correct)

    raise ValueError(f"Unknown scoring: {scoring}")


# =============================================================================
# Permutation Importance
# =============================================================================


def permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: Optional[Union[str, Callable]] = None,
    n_repeats: int = 5,
    sample_weight: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    permute_method: str = "global",
    block_size: Union[int, str] = "auto",
    n_jobs: int = -1,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Permutation importance with time-series-aware permutation strategies.

    Parameters
    ----------
    model : fitted estimator
        Must have .predict() method.
    X : DataFrame
        Features.
    y : Series
        Target.
    scoring : str or callable, optional
        Scoring metric. Default: neg_mean_squared_error for regression.
        If callable, signature: scoring(y_true, y_pred, sample_weight=None) -> float.
    n_repeats : int
        Number of permutation repeats per feature.
    sample_weight : array, optional
        Weights for scoring.
    groups : array, optional
        Group labels (e.g., player_id). Required for within_group/block/circular_shift.
    time : array, optional
        Time ordering. Required for block method.
    permute_method : str
        - "global": standard permutation (shuffle all rows)
        - "within_group": permute only within each group
        - "block": permute contiguous blocks within each group
        - "circular_shift": shift each group's values by random offset
    block_size : int or "auto"
        Block size for block method. "auto" uses sqrt(n_per_group).
    n_jobs : int
        Number of parallel jobs (-1 = all cores).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DataFrame with columns:
        feature: feature name
        importance_mean: mean decrease in score across repeats
        importance_std: std across repeats
        baseline_score: score with unpermuted data
    """
    # Validate inputs
    if permute_method not in ("global", "within_group", "block", "circular_shift"):
        raise ValueError(
            f"Unknown permute_method: {permute_method}. "
            "Valid options: 'global', 'within_group', 'block', 'circular_shift'"
        )

    if permute_method in ("within_group", "block", "circular_shift") and groups is None:
        raise ValueError(
            f"permute_method='{permute_method}' requires groups parameter."
        )

    if permute_method == "block" and time is None:
        raise ValueError("permute_method='block' requires time parameter.")

    # Convert to arrays
    feature_names = list(X.columns)
    X_arr = X.values.astype(np.float64)
    y_arr = np.asarray(y).ravel()
    n, p = X_arr.shape

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)

    if groups is not None:
        groups = np.asarray(groups)
        # Encode to integers if needed
        if not np.issubdtype(groups.dtype, np.integer):
            _, groups = np.unique(groups, return_inverse=True)

    if time is not None:
        time = np.asarray(time, dtype=np.float64)

    # Determine block size
    if isinstance(block_size, str) and block_size == "auto":
        if groups is not None:
            # Use median group size
            _, counts = np.unique(groups, return_counts=True)
            median_size = int(np.median(counts))
            b_size = max(1, int(np.sqrt(median_size)))
        else:
            b_size = max(1, int(np.sqrt(n)))
    else:
        b_size = int(block_size)

    # Calculate baseline score
    baseline_score = _score(model, X_arr, y_arr, sample_weight, scoring)

    # Create RNG
    rng = np.random.default_rng(random_state)

    def _compute_importance_for_feature(j: int, seeds: List[int]) -> List[float]:
        """Compute importance for a single feature across repeats."""
        scores = []
        for seed in seeds:
            rng_local = np.random.default_rng(seed)
            X_permuted = X_arr.copy()

            # Permute feature j
            if permute_method == "global":
                X_permuted[:, j] = _permute_global(X_arr[:, j], rng_local)
            elif permute_method == "within_group":
                X_permuted[:, j] = _permute_within_group(
                    X_arr[:, j], groups, rng_local
                )
            elif permute_method == "block":
                X_permuted[:, j] = _permute_block(
                    X_arr[:, j], groups, time, b_size, rng_local
                )
            elif permute_method == "circular_shift":
                X_permuted[:, j] = _permute_circular_shift(
                    X_arr[:, j], groups, time, rng_local
                )

            permuted_score = _score(model, X_permuted, y_arr, sample_weight, scoring)
            scores.append(baseline_score - permuted_score)

        return scores

    # Generate seeds for reproducibility
    all_seeds = rng.integers(0, 2**31, size=(p, n_repeats)).tolist()

    # Compute importance in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_importance_for_feature)(j, all_seeds[j])
        for j in range(p)
    )

    # Aggregate results
    importance_mean = []
    importance_std = []
    for scores in results:
        importance_mean.append(np.mean(scores))
        importance_std.append(np.std(scores))

    return pd.DataFrame({
        "feature": feature_names,
        "importance_mean": importance_mean,
        "importance_std": importance_std,
        "baseline_score": baseline_score,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


__all__ = ["permutation_importance"]
