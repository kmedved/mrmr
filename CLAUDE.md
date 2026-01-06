# CLAUDE.md - AI Assistant Guide for sift

This document provides guidance for AI assistants working with the `sift` codebase.

## Project Overview

**sift** is a feature selection toolbox for machine learning that implements:
- **mRMR / JMI / JMIM** - Minimum Redundancy Maximum Relevance and Joint Mutual Information variants
- **CEFS+** - Correlation-based feature selection with Gaussian copula MI proxy
- **Stability Selection** - Bootstrap-based robust feature selection with Lasso/ElasticNet
- **Boruta / Boruta-SHAP** - All-relevant feature selection using shadow features
- **CatBoost-based selection** - SHAP/permutation/forward selection (optional dependency)

The library supports both pandas DataFrames and numpy arrays, with a Polars path for mRMR.

## Repository Structure

```
sift/
├── sift/                    # Main library code
│   ├── __init__.py         # Package exports and version
│   ├── api.py              # User-facing API (select_mrmr, select_jmi, select_cefsplus, etc.)
│   ├── stability.py        # StabilitySelector class and convenience functions
│   ├── boruta.py           # BorutaSelector and Boruta-SHAP
│   ├── catboost.py         # CatBoost-based selection (optional)
│   ├── importance.py       # Permutation importance utilities
│   ├── _preprocess.py      # Input validation, encoding, subsampling
│   ├── _permute.py         # Shadow feature permutation strategies
│   ├── _impute.py          # Missing value imputation
│   ├── estimators/         # MI estimators (copula, relevance, joint_mi)
│   ├── selection/          # Core selection algorithms (loops, cefsplus, auto_k)
│   └── sampling/           # Smart sampling for large datasets (anchors, leverage-based)
├── tests/                  # pytest test suite
├── setup.py               # Package metadata and dependencies
└── README.md              # User documentation
```

## Key Modules

### `sift/api.py`
Main user-facing API with functions:
- `select_mrmr(X, y, k, task, ...)` - mRMR feature selection
- `select_jmi(X, y, k, task, ...)` - Joint Mutual Information selection
- `select_jmim(X, y, k, task, ...)` - JMI Maximization (conservative variant)
- `select_cefsplus(X, y, k, ...)` - CEFS+ (regression only)

Parameters follow consistent patterns:
- `k`: Number of features or `"auto"` for automatic selection
- `task`: `"regression"` or `"classification"`
- `estimator`: `"classic"`, `"gaussian"`, or `"auto"`
- `subsample`: Optional subsampling for large datasets (default 50,000)
- `verbose`: Print progress (default True)

### `sift/stability.py`
`StabilitySelector` class implementing bootstrap stability selection:
- Fits Lasso/ElasticNet (regression) or L1-LogisticRegression (classification)
- Supports block bootstrap for time series (`groups`, `time` parameters)
- Smart sampling integration (`use_smart_sampler`)
- Convenience functions: `stability_regression()`, `stability_classif()`

### `sift/boruta.py`
`BorutaSelector` class for all-relevant feature selection:
- Pluggable importance backend (`"native"` or `"shap"`)
- Time-series aware shadow permutations
- Returns `BorutaResult` with feature rankings

### `sift/catboost.py` (optional)
CatBoost-based feature selection:
- `catboost_select()` - Full API with k search, stability selection
- `catboost_regression()`, `catboost_classif()` - Simple wrappers
- Supports custom CV splitters (TimeSeriesSplit, GroupKFold, etc.)

### `sift/estimators/`
Mutual information and relevance estimators:
- `copula.py` - Gaussian copula MI proxy, `FeatureCache`, `build_cache()`
- `relevance.py` - F-stat, KS, Random Forest relevance scores
- `joint_mi.py` - Joint MI estimation (binned, KSG variants)

### `sift/selection/`
Core selection algorithms:
- `loops.py` - `mrmr_select()`, `jmi_select()` greedy loops
- `cefsplus.py` - `select_cached()` for CEFS+ with cached covariances
- `auto_k.py` - Automatic k selection via elbow or cross-validation

### `sift/sampling/`
Smart sampling for large datasets:
- `smart.py` - Leverage-based subsampling with inverse-probability weights
- `anchors.py` - Anchor point strategies for panel/time series data

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/kmedved/sift.git
cd sift
python -m pip install -e ".[test]"

# Optional extras
pip install -e ".[all]"         # All optional dependencies
pip install -e ".[polars]"      # Polars support
pip install -e ".[categorical]" # category_encoders
pip install -e ".[catboost]"    # CatBoost support
pip install -e ".[numba]"       # Numba JIT (included by default)
```

## Running Tests

```bash
# Run full test suite
pytest tests/

# Run specific test file
pytest tests/test_main.py

# Run with verbose output
pytest tests/ -v

# Run CatBoost tests (requires catboost extra)
pytest tests/test_catboost.py -q
```

Tests are run on Python 3.10, 3.11, and 3.12 via GitHub Actions.

## Code Conventions

### Function Signatures
- Use type hints for all public functions
- Accept both DataFrame and ndarray inputs where practical
- Return feature names (list[str]) for DataFrame inputs, indices for arrays
- Use `return_indices` parameter to override default behavior

### Parameter Naming
- `k`: Number of features to select
- `task`: `"regression"` or `"classification"`
- `groups`: Group labels for grouped/panel data
- `time`: Time values for temporal ordering
- `sample_weight`: Sample weights (normalized internally)
- `random_state`: Random seed for reproducibility
- `verbose`: Print progress information
- `n_jobs`: Parallelism (-1 for all cores)

### Internal Conventions
- Prefix private functions with `_` (e.g., `_prepare_xy_classic`)
- Use `@dataclass` for result objects (e.g., `BorutaResult`, `CatBoostSelectionResult`)
- Validate inputs early with clear error messages
- Support both explicit parameters and auto-detection

### Testing Patterns
- Use `np.random.default_rng(seed)` for reproducibility
- Test with small synthetic datasets (200-1000 samples, 4-10 features)
- Verify output length matches expected `k`
- Check that informative features are selected

## Common Patterns

### Creating a New Selector
```python
from sklearn.base import BaseEstimator, TransformerMixin

class MySelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=10, random_state=None, verbose=True):
        self.k = k
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        # Store feature names
        self.feature_names_in_ = list(X.columns) if hasattr(X, 'columns') else None
        # Perform selection logic
        self.selected_features_ = [...]
        return self

    def transform(self, X):
        return X[self.selected_features_] if hasattr(X, 'columns') else X[:, self.selected_indices_]

    def get_support(self, indices=False):
        # Return mask or indices
        pass
```

### Adding Auto-K Support
Functions that support `k="auto"` should:
1. Accept `groups` and `time` parameters for evaluation strategy
2. Use `AutoKConfig` for configuration
3. Implement elbow detection or cross-validation selection

### Working with Groups/Time
```python
# Block bootstrap for panel data
selector.fit(X, y, groups=df['entity_id'], time=df['date'])

# Shadow permutations respecting temporal structure
select_boruta(X, y, groups=groups, time=time, shadow_method='block')
```

## Dependencies

### Required
- numpy >= 1.18.1
- pandas >= 1.0.3
- scikit-learn
- scipy
- joblib
- numba

### Optional
- `polars` - Polars DataFrame support
- `category_encoders` - Advanced categorical encoding
- `catboost` - CatBoost-based selection
- `shap` - SHAP values (for non-CatBoost models)

## API Design Principles

1. **Consistency**: All selectors follow similar parameter patterns
2. **Flexibility**: Support both simple and advanced use cases
3. **Safety**: Validate inputs, handle edge cases, avoid data leakage
4. **Performance**: Subsample large datasets, use parallelism, cache computations
5. **Transparency**: Verbose output, return intermediate results when useful

## When Making Changes

1. **Read existing code first** - Understand patterns before modifying
2. **Run tests** - `pytest tests/` should pass
3. **Follow conventions** - Match existing parameter names and patterns
4. **Update `__all__`** - When adding public functions/classes
5. **Add tests** - For new functionality or bug fixes
6. **Keep it simple** - Avoid over-engineering; focus on the task at hand

## Common Tasks

### Adding a new selection method
1. Implement in appropriate module (`api.py` for user-facing, `selection/` for core algorithm)
2. Add to `__init__.py` exports
3. Write tests in `tests/`
4. Update README.md examples if significant

### Debugging selection issues
1. Check input shapes and types
2. Verify `task` parameter matches target type
3. Use `verbose=True` to see progress
4. Check for missing/infinite values in data

### Performance optimization
1. Use `subsample` parameter for large datasets
2. Enable `n_jobs=-1` for parallel MI estimation
3. Consider `estimator="gaussian"` for faster MI proxy
4. Use `prefilter_k` in CatBoost selection
