"""User-facing API for feature selection."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from sift._preprocess import (
    CatEncoding,
    EstimatorJMI,
    EstimatorMRMR,
    Formula,
    RelevanceMethod,
    Task,
    check_regression_only,
    encode_categoricals,
    resolve_jmi_estimator,
    subsample_xy,
    validate_inputs,
)
from sift.estimators import relevance as rel_est
from sift.estimators.copula import FeatureCache, build_cache
from sift.selection.cefsplus import select_cached
from sift.selection.loops import jmi_select, mrmr_select


def _default_top_m(top_m: Optional[int], k: int) -> int:
    tm = max(5 * k, 250) if top_m is None else int(top_m)
    # Ensure we can still return k features when a user passes top_m < k.
    return max(tm, int(k))


def _prepare_xy_classic(
    X,
    y,
    *,
    task: Task,
    cat_features: Optional[List[str]],
    cat_encoding: CatEncoding,
    subsample: Optional[int],
    random_state: int,
    sample_weight: Optional[np.ndarray] = None,
):
    """
    Shared preparation for 'classic' selectors:
    - infer cat_features for DataFrames
    - optional categorical encoding
    - validate_inputs + subsample_xy
    Returns: (X_arr, y_arr, feature_names, sample_weight_arr)
    """
    if cat_features and cat_encoding != "none" and not isinstance(X, pd.DataFrame):
        raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")

    if isinstance(X, pd.DataFrame) and cat_features is None:
        cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if cat_features and cat_encoding != "none":
        X = encode_categoricals(X, y, cat_features, cat_encoding)

    X_arr, y_arr, feature_names = validate_inputs(X, y, task)

    # Handle sample weights
    if sample_weight is not None:
        w_arr = np.asarray(sample_weight, dtype=np.float64)
        if len(w_arr) != len(X_arr):
            raise ValueError(f"sample_weight has {len(w_arr)} elements but X has {len(X_arr)} rows")
    else:
        w_arr = None

    # Subsample together
    if subsample is not None and len(X_arr) > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_arr), size=subsample, replace=False)
        X_arr = X_arr[idx]
        y_arr = y_arr[idx]
        if w_arr is not None:
            w_arr = w_arr[idx]

    return X_arr, y_arr, feature_names, w_arr


def select_mrmr(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: int,
    *,
    task: Task,
    relevance: RelevanceMethod = "f",
    estimator: EstimatorMRMR = "classic",
    formula: Formula = "quotient",
    top_m: Optional[int] = None,
    cat_features: Optional[List[str]] = None,
    cat_encoding: CatEncoding = "loo",
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    sample_weight: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Minimum Redundancy Maximum Relevance feature selection.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix.
    y : Series or ndarray
        Target variable.
    k : int
        Number of features to select.
    task : {"regression", "classification"}
        Task type.
    relevance : {"f", "ks", "rf"}
        Relevance scoring (only for estimator="classic").
    estimator : {"classic", "gaussian"}
        - "classic": F-stat relevance, Pearson correlation redundancy
        - "gaussian": Gaussian MI proxy (fast, regression only)
    formula : {"quotient", "difference"}
        - "quotient": rel / mean(red)
        - "difference": rel - mean(red)
    top_m : int, optional
        Prefilter to top_m features by relevance. Default: max(5*k, 250).
    sample_weight : array-like, optional
        Sample weights. If provided, uses weighted relevance and redundancy.

    Returns
    -------
    List[str]
        Selected feature names.
    """
    if estimator == "gaussian":
        check_regression_only(task, estimator)
        return _mrmr_gaussian(
            X,
            y,
            k,
            formula,
            top_m,
            cat_features,
            cat_encoding,
            subsample,
            random_state,
            sample_weight,
            verbose,
        )

    return _mrmr_classic(
        X,
        y,
        k,
        task,
        relevance,
        formula,
        top_m,
        cat_features,
        cat_encoding,
        subsample,
        random_state,
        sample_weight,
        verbose,
    )


def _mrmr_classic(
    X,
    y,
    k,
    task,
    relevance_method,
    formula,
    top_m,
    cat_features,
    cat_encoding,
    subsample,
    random_state,
    sample_weight,
    verbose,
):
    """Classic mRMR implementation."""
    X_arr, y_arr, feature_names, w_arr = _prepare_xy_classic(
        X,
        y,
        task=task,
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        subsample=subsample,
        random_state=random_state,
        sample_weight=sample_weight,
    )

    # Use weighted or unweighted relevance functions
    if w_arr is not None:
        if task == "regression":
            rel_funcs = {
                "f": lambda X, y: rel_est.f_regression_weighted(X, y, w_arr),
                "rf": rel_est.rf_regression,  # RF doesn't support weights here
            }
        else:
            rel_funcs = {
                "f": lambda X, y: rel_est.f_classif_weighted(X, y, w_arr),
                "ks": rel_est.ks_classif,  # KS doesn't support weights
                "rf": rel_est.rf_classif,  # RF doesn't support weights here
            }
    else:
        if task == "regression":
            rel_funcs = {"f": rel_est.f_regression, "rf": rel_est.rf_regression}
        else:
            rel_funcs = {
                "f": rel_est.f_classif,
                "ks": rel_est.ks_classif,
                "rf": rel_est.rf_classif,
            }

    if relevance_method not in rel_funcs:
        raise ValueError(
            f"relevance='{relevance_method}' not valid for task='{task}'. "
            f"Valid options: {sorted(rel_funcs.keys())}"
        )

    rel = rel_funcs[relevance_method](X_arr, y_arr)

    top_m = _default_top_m(top_m, k)

    if verbose:
        weighted_str = " (weighted)" if w_arr is not None else ""
        print(f"mRMR classic{weighted_str}: selecting {k} features from {X_arr.shape[1]} (top_m={top_m})")

    selected_idx = mrmr_select(X_arr, rel, k, formula=formula, top_m=top_m, sample_weight=w_arr)

    return [feature_names[i] for i in selected_idx]


def _mrmr_gaussian(
    X,
    y,
    k,
    formula,
    top_m,
    cat_features,
    cat_encoding,
    subsample,
    random_state,
    sample_weight,
    verbose,
):
    """Gaussian mRMR via cached selection."""
    if isinstance(X, pd.DataFrame) and cat_features is None:
        cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if cat_features and cat_encoding != "none" and not isinstance(X, pd.DataFrame):
        raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")

    if cat_features and cat_encoding != "none":
        X = encode_categoricals(X, y, cat_features, cat_encoding)
    if verbose:
        weighted_str = " (weighted)" if sample_weight is not None else ""
        print(f"mRMR gaussian{weighted_str}: selecting {k} features (top_m={top_m})")
    cache = build_cache(X, subsample=subsample, random_state=random_state, sample_weight=sample_weight)
    method = "mrmr_quot" if formula == "quotient" else "mrmr_diff"
    return select_cached(cache, y, k, method=method, top_m=top_m)


def select_jmi(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: int,
    *,
    task: Task,
    estimator: EstimatorJMI = "auto",
    relevance: RelevanceMethod = "f",
    top_m: Optional[int] = None,
    cat_features: Optional[List[str]] = None,
    cat_encoding: CatEncoding = "loo",
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    sample_weight: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Joint Mutual Information feature selection.

    score(f) = Σ_{s ∈ S} I(f, s; y)

    Parameters
    ----------
    sample_weight : array-like, optional
        Sample weights. If provided, uses weighted relevance and cache.
    """
    estimator = resolve_jmi_estimator(estimator, task)

    check_regression_only(task, estimator)

    if estimator == "gaussian":
        if isinstance(X, pd.DataFrame) and cat_features is None:
            cat_features = X.select_dtypes(
                include=["object", "category", "string"]
            ).columns.tolist()
        if cat_features and cat_encoding != "none" and not isinstance(X, pd.DataFrame):
            raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")
        if cat_features and cat_encoding != "none":
            X = encode_categoricals(X, y, cat_features, cat_encoding)
        if verbose:
            weighted_str = " (weighted)" if sample_weight is not None else ""
            print(f"JMI gaussian{weighted_str}: selecting {k} features (top_m={top_m})")
        cache = build_cache(X, subsample=subsample, random_state=random_state, sample_weight=sample_weight)
        return select_cached(cache, y, k, method="jmi", top_m=top_m)

    return _jmi_classic(
        X,
        y,
        k,
        task,
        estimator,
        relevance,
        False,
        top_m,
        cat_features,
        cat_encoding,
        subsample,
        random_state,
        sample_weight,
        verbose,
    )


def select_jmim(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: int,
    *,
    task: Task,
    estimator: EstimatorJMI = "auto",
    relevance: RelevanceMethod = "f",
    top_m: Optional[int] = None,
    cat_features: Optional[List[str]] = None,
    cat_encoding: CatEncoding = "loo",
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    sample_weight: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> List[str]:
    """
    JMI Maximization — conservative variant.

    score(f) = min_{s ∈ S} I(f, s; y)

    Parameters
    ----------
    sample_weight : array-like, optional
        Sample weights. If provided, uses weighted relevance and cache.
    """
    estimator = resolve_jmi_estimator(estimator, task)

    check_regression_only(task, estimator)

    if estimator == "gaussian":
        if isinstance(X, pd.DataFrame) and cat_features is None:
            cat_features = X.select_dtypes(
                include=["object", "category", "string"]
            ).columns.tolist()
        if cat_features and cat_encoding != "none" and not isinstance(X, pd.DataFrame):
            raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")
        if cat_features and cat_encoding != "none":
            X = encode_categoricals(X, y, cat_features, cat_encoding)
        if verbose:
            weighted_str = " (weighted)" if sample_weight is not None else ""
            print(f"JMIM gaussian{weighted_str}: selecting {k} features (top_m={top_m})")
        cache = build_cache(X, subsample=subsample, random_state=random_state, sample_weight=sample_weight)
        return select_cached(cache, y, k, method="jmim", top_m=top_m)

    return _jmi_classic(
        X,
        y,
        k,
        task,
        estimator,
        relevance,
        True,
        top_m,
        cat_features,
        cat_encoding,
        subsample,
        random_state,
        sample_weight,
        verbose,
    )


def _jmi_classic(
    X,
    y,
    k,
    task,
    mi_estimator,
    relevance_method,
    use_min,
    top_m,
    cat_features,
    cat_encoding,
    subsample,
    random_state,
    sample_weight,
    verbose,
):
    """Classic JMI/JMIM implementation."""
    X_arr, y_arr, feature_names, w_arr = _prepare_xy_classic(
        X,
        y,
        task=task,
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        subsample=subsample,
        random_state=random_state,
        sample_weight=sample_weight,
    )

    # Use weighted or unweighted relevance functions
    if w_arr is not None:
        if task == "regression":
            rel_funcs = {
                "f": lambda X, y: rel_est.f_regression_weighted(X, y, w_arr),
                "rf": rel_est.rf_regression,
            }
        else:
            rel_funcs = {
                "f": lambda X, y: rel_est.f_classif_weighted(X, y, w_arr),
                "ks": rel_est.ks_classif,
                "rf": rel_est.rf_classif,
            }
    else:
        if task == "regression":
            rel_funcs = {"f": rel_est.f_regression, "rf": rel_est.rf_regression}
        else:
            rel_funcs = {
                "f": rel_est.f_classif,
                "ks": rel_est.ks_classif,
                "rf": rel_est.rf_classif,
            }

    if relevance_method not in rel_funcs:
        raise ValueError(
            f"relevance='{relevance_method}' not valid for task='{task}'. "
            f"Valid options: {sorted(rel_funcs.keys())}"
        )

    rel = rel_funcs[relevance_method](X_arr, y_arr)

    y_kind = "discrete" if task == "classification" else "continuous"
    aggregation = "min" if use_min else "sum"

    top_m = _default_top_m(top_m, k)

    if verbose:
        method = "JMIM" if use_min else "JMI"
        weighted_str = " (weighted)" if w_arr is not None else ""
        print(f"{method} classic{weighted_str}: selecting {k} features from {X_arr.shape[1]} (top_m={top_m})")

    selected_idx = jmi_select(
        X_arr,
        y_arr,
        k,
        rel,
        mi_estimator=mi_estimator,
        aggregation=aggregation,
        top_m=top_m,
        y_kind=y_kind,
    )

    return [feature_names[i] for i in selected_idx]


def select_cefsplus(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: int,
    *,
    top_m: Optional[int] = None,
    corr_prune: float = 0.95,
    cat_features: Optional[List[str]] = None,
    cat_encoding: CatEncoding = "loo",
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    sample_weight: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> List[str]:
    """
    CEFS+ feature selection using log-det Gaussian MI proxy.

    REGRESSION ONLY.

    Parameters
    ----------
    sample_weight : array-like, optional
        Sample weights. If provided, uses weighted cache building.
    """
    if cat_features and cat_encoding != "none" and not isinstance(X, pd.DataFrame):
        raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")
    if isinstance(X, pd.DataFrame) and cat_features is None:
        cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if cat_features and cat_encoding != "none":
        X = encode_categoricals(X, y, cat_features, cat_encoding)
    from sift._preprocess import to_numpy

    y_arr = to_numpy(y, dtype=np.float32).ravel()
    n_rows = X.shape[0] if hasattr(X, "shape") else len(X)
    if len(y_arr) != n_rows:
        raise ValueError(f"X has {n_rows} rows but y has {len(y_arr)}")
    if not np.isfinite(y_arr).all():
        raise ValueError("Non-finite values in y are not allowed for regression.")
    top_m = _default_top_m(top_m, k)
    if verbose:
        weighted_str = " (weighted)" if sample_weight is not None else ""
        print(f"CEFS+{weighted_str}: selecting {k} features (top_m={top_m}, corr_prune={corr_prune})")
    cache = build_cache(X, subsample=subsample, random_state=random_state, sample_weight=sample_weight)
    return select_cached(
        cache,
        y,
        k,
        method="cefsplus",
        top_m=top_m,
        corr_prune=corr_prune,
    )


__all__ = [
    "FeatureCache",
    "build_cache",
    "select_cached",
    "select_cefsplus",
    "select_jmi",
    "select_jmim",
    "select_mrmr",
]
