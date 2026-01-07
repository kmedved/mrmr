# Feature Selection Algorithms

This document provides detailed explanations of the feature selection algorithms implemented in SIFT, including their mathematical foundations, strengths, and when to use each one.

## Table of Contents

- [Overview](#overview)
- [Information-Theoretic Methods](#information-theoretic-methods)
  - [mRMR (Minimum Redundancy Maximum Relevance)](#mrmr-minimum-redundancy-maximum-relevance)
  - [JMI (Joint Mutual Information)](#jmi-joint-mutual-information)
  - [JMIM (JMI Maximization)](#jmim-jmi-maximization)
  - [CEFS+ (Conditional Entropy Feature Selection)](#cefs-conditional-entropy-feature-selection)
- [Stability-based Methods](#stability-based-methods)
  - [Stability Selection](#stability-selection)
- [Wrapper Methods](#wrapper-methods)
  - [Boruta](#boruta)
  - [CatBoost-based Selection](#catboost-based-selection)
- [Mutual Information Estimators](#mutual-information-estimators)
- [Algorithm Comparison](#algorithm-comparison)

---

## Overview

Feature selection methods can be categorized into three main types:

1. **Filter Methods**: Evaluate features independently of any learning algorithm using statistical measures. Fast but may miss feature interactions.

2. **Wrapper Methods**: Use a learning algorithm to evaluate feature subsets. More accurate but computationally expensive.

3. **Embedded Methods**: Feature selection occurs as part of the model training process.

SIFT implements algorithms from all categories:

| Method | Category | Complexity | Best For |
|--------|----------|------------|----------|
| mRMR | Filter | O(k × p) | Quick baseline |
| JMI/JMIM | Filter | O(k × p²) | Capturing interactions |
| CEFS+ | Filter | O(k³) | Minimal-optimal subset |
| Stability Selection | Embedded | O(B × n × p) | Robust selection |
| Boruta | Wrapper | O(T × n × p) | All-relevant features |
| CatBoost | Wrapper | O(k × n × p) | Production pipelines |

Where: k = features selected, p = total features, n = samples, B = bootstrap iterations, T = trees

---

## Information-Theoretic Methods

These methods use information theory concepts, primarily mutual information, to measure feature relevance and redundancy.

### Mutual Information Background

Mutual information I(X; Y) measures the amount of information obtained about one random variable through observing another:

```
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
```

Where H is entropy. For continuous variables:

```
I(X; Y) = ∫∫ p(x,y) log(p(x,y) / (p(x)p(y))) dx dy
```

Key properties:
- I(X; Y) ≥ 0 (non-negative)
- I(X; Y) = 0 iff X and Y are independent
- I(X; Y) = I(Y; X) (symmetric)

---

### mRMR (Minimum Redundancy Maximum Relevance)

**Reference:** Peng et al. (2005) "Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy"

#### Algorithm

mRMR iteratively selects features that maximize relevance to the target while minimizing redundancy with already-selected features.

At each step, select feature f that maximizes:

**Quotient formula:**
```
score(f) = I(f; y) / (1/|S|) Σ_{s∈S} ρ(f, s)
```

**Difference formula:**
```
score(f) = I(f; y) - (1/|S|) Σ_{s∈S} ρ(f, s)
```

Where:
- I(f; y) = relevance (mutual information with target)
- ρ(f, s) = redundancy (correlation with selected features)
- S = set of already selected features

#### Implementations in SIFT

**Classic Estimator:**
- Relevance: F-statistic, KS-statistic, or Random Forest importance
- Redundancy: Pearson correlation

**Gaussian Estimator:**
- Uses Gaussian copula to approximate MI via correlation
- Much faster but assumes Gaussian relationships
- Regression only

#### When to Use

- Fast baseline selection
- Large datasets where speed matters
- Features with approximately linear relationships

#### Limitations

- Greedy algorithm may not find global optimum
- Pairwise redundancy may miss higher-order interactions
- Assumes features can be evaluated independently

---

### JMI (Joint Mutual Information)

**Reference:** Yang & Moody (1999) "Data Visualization and Feature Selection: New Algorithms for Nongaussian Data"

#### Algorithm

JMI considers the joint information between candidate features, already-selected features, and the target:

```
score(f) = Σ_{s∈S} I(f, s; y)
```

Where I(f, s; y) is the joint mutual information, which can be decomposed as:

```
I(f, s; y) = I(f; y) + I(s; y|f)
```

This captures both:
1. The individual relevance of f
2. The complementary information that f provides about y given s

#### Advantages Over mRMR

- Considers feature interactions through joint information
- Better at selecting complementary features
- More robust to redundant features

#### When to Use

- When feature interactions are important
- When you want complementary rather than just non-redundant features
- Classification and regression

---

### JMIM (JMI Maximization)

**Reference:** Bennasar et al. (2015) "Feature selection using Joint Mutual Information Maximisation"

#### Algorithm

JMIM is a conservative variant of JMI that uses the minimum instead of sum:

```
score(f) = min_{s∈S} I(f, s; y)
```

#### Rationale

- Using minimum ensures the selected feature provides value with respect to ALL previously selected features
- Prevents selecting features that are highly redundant with any single selected feature
- More conservative than JMI but may miss some good features

#### When to Use

- When you want to ensure every feature adds value
- When the feature set should be maximally diverse
- When you prefer false negatives over false positives

---

### CEFS+ (Conditional Entropy Feature Selection)

**Reference:** Brown et al. (2012) "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection"

#### Algorithm

CEFS+ uses a Gaussian approximation to compute conditional mutual information efficiently:

```
I(y; f | S) = -0.5 log(1 - ρ²(y, f | S))
```

Where ρ(y, f | S) is the partial correlation between y and f given S.

The algorithm:
1. Transform features to Gaussian via probability integral transform
2. Compute correlation matrix
3. Use Schur complement for efficient incremental updates

#### Incremental Updates

Instead of recomputing the full conditional correlation at each step, CEFS+ uses matrix updates:

```
Σ_{new} = Σ_{old} - σ_f σ_f^T / σ_ff
```

This reduces complexity from O(k³) per step to O(k²) per step.

#### When to Use

- Regression problems
- When you want minimal-optimal feature subsets
- When features have complex conditional relationships

#### Limitations

- Regression only (requires continuous target)
- Assumes Gaussian relationships (after transformation)

---

## Stability-based Methods

### Stability Selection

**Reference:** Meinshausen & Bühlmann (2010) "Stability selection"

#### Algorithm

Stability selection identifies features that are consistently selected across multiple bootstrap resamples:

1. For b = 1 to B bootstrap iterations:
   - Sample subset of data (typically 50%)
   - Fit sparse model (Lasso/ElasticNet/LogisticRegression)
   - Record which features have non-zero coefficients

2. Compute selection frequency for each feature:
   ```
   freq(f) = (1/B) Σ_b 1[f selected in bootstrap b]
   ```

3. Select features with freq(f) ≥ threshold

#### Theoretical Guarantees

Under certain conditions, stability selection provides:
- Finite-sample control of false discovery rate
- Consistency in high-dimensional settings

The expected number of false positives is bounded by:

```
E[V] ≤ q² / (2θ - 1) × p
```

Where q = threshold, θ = expected fraction of selected features.

#### Block Bootstrap for Time Series

For time-series data, standard bootstrap breaks temporal dependencies. SIFT implements block bootstrap:

**Moving Block Bootstrap:**
- Sample consecutive blocks of observations
- Preserves within-block dependencies

**Circular Block Bootstrap:**
- Wraps around at boundaries
- Reduces edge effects

**Stationary Block Bootstrap:**
- Random block lengths (geometric distribution)
- Ensures stationarity of bootstrap samples

#### When to Use

- When stability/robustness is important
- Noisy data with many spurious correlations
- When you want confidence in feature selection
- Time-series or grouped data (with block bootstrap)

---

## Wrapper Methods

### Boruta

**Reference:** Kursa & Rudnicki (2010) "Feature Selection with the Boruta Package"

#### Algorithm

Boruta is an "all-relevant" feature selection method that finds all features carrying information about the target:

1. Create shadow features by shuffling each original feature
2. Train Random Forest on original + shadow features
3. Compute importance for all features
4. Compare each feature's importance to max shadow importance
5. Mark features as:
   - **Confirmed**: Significantly higher than max shadow (binomial test)
   - **Rejected**: Significantly lower than max shadow
   - **Tentative**: Cannot decide
6. Remove rejected features, reset shadow features
7. Repeat until all features are confirmed/rejected or max iterations

#### Rationale

- Shadow features represent "random" importance
- Features must beat this baseline to be considered relevant
- Multiple iterations reduce false positives from random fluctuations

#### When to Use

- When you want ALL relevant features (not minimal subset)
- When false negatives are costly
- Exploratory analysis
- Understanding feature importance

#### Limitations

- Slow (many Random Forest fits)
- May select redundant features (all-relevant, not minimal)

---

### CatBoost-based Selection

SIFT implements several CatBoost-based selection algorithms:

#### Forward Selection

```
For k = 1 to K:
    1. Train model on selected features
    2. Compute importance for remaining features
    3. Add most important feature to selected set
```

Complexity: O(K) model fits

#### Forward Greedy Selection

```
For k = 1 to K:
    For each remaining feature f:
        1. Train model on selected + {f}
        2. Record validation score
    Add feature with best score
```

Complexity: O(K × p) model fits - most accurate but slowest

#### RFE with SHAP/Permutation Importance

```
While |features| > K:
    1. Train model on current features
    2. Compute SHAP or permutation importance
    3. Remove least important feature
```

Complexity: O((p - K) × n_splits) model fits

#### Two-Stage Pipeline

To avoid information leakage, prefiltering happens inside CV:

```
For each CV fold:
    1. On training data: prefilter to top-M features
    2. Run main selection on prefiltered features
    3. Evaluate on validation fold
```

This prevents information from the test set influencing feature selection.

---

## Mutual Information Estimators

SIFT implements several MI estimators:

### Gaussian Estimator

Assumes Gaussian copula relationship:

```
I(X; Y) ≈ -0.5 log(1 - ρ²)
```

- **Pros:** Very fast, handles high dimensions
- **Cons:** Only captures linear relationships

### Binned (Histogram) Estimator

Discretize continuous variables into bins:

```
I(X; Y) = Σ_x Σ_y p(x,y) log(p(x,y) / (p(x)p(y)))
```

- **Pros:** No distributional assumptions
- **Cons:** Sensitive to bin count, curse of dimensionality

### KSG (Kraskov-Stögbauer-Grassberger)

Uses k-nearest neighbors:

```
I(X; Y) ≈ ψ(k) - <ψ(n_x) + ψ(n_y)> + ψ(N)
```

Where ψ is the digamma function and n_x, n_y are neighbor counts.

- **Pros:** Captures nonlinear relationships, no binning required
- **Cons:** Computationally expensive, sensitive to k

### R² Proxy

Quick proxy based on coefficient of determination:

```
I(X; Y) ≈ -0.5 log(1 - R²)
```

- **Pros:** Very fast
- **Cons:** Only captures linear relationships

---

## Algorithm Comparison

### Speed vs Accuracy

```
Speed:    mRMR(gaussian) > mRMR(classic) > JMI > CEFS+ > Stability > Boruta > CatBoost(SHAP)
Accuracy: CatBoost(SHAP) ≈ Boruta > Stability > CEFS+ > JMI > mRMR
```

### Feature Set Size

```
Minimal:     CEFS+ → mRMR → Stability → JMI
All-relevant: Boruta
Flexible:    CatBoost (adjustable)
```

### Handling Interactions

```
High-order interactions: Boruta > CatBoost(SHAP) > JMI > JMIM > mRMR
```

### Robustness to Noise

```
Most robust: Stability > Boruta > CEFS+ > JMI > mRMR
```

### Decision Guide

| Scenario | Recommended |
|----------|-------------|
| Quick baseline | mRMR (gaussian) |
| Production pipeline | CatBoost (forward) + Stability |
| Exploratory analysis | Boruta |
| Time series | Stability (block bootstrap) + CatBoost (TimeSeriesSplit) |
| High noise | Stability Selection |
| Feature interactions matter | JMI or CatBoost (SHAP) |
| Minimal feature set | CEFS+ or mRMR |
| Interpretability needed | CatBoost (SHAP) |
| Very large dataset | mRMR (gaussian) + smart sampling |

---

## References

1. Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy. IEEE TPAMI.

2. Yang, H. H., & Moody, J. (1999). Data Visualization and Feature Selection: New Algorithms for Nongaussian Data. NIPS.

3. Bennasar, M., Hicks, Y., & Setchi, R. (2015). Feature selection using Joint Mutual Information Maximisation. Expert Systems with Applications.

4. Brown, G., Pocock, A., Zhao, M. J., & Luján, M. (2012). Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection. JMLR.

5. Meinshausen, N., & Bühlmann, P. (2010). Stability selection. Journal of the Royal Statistical Society: Series B.

6. Kursa, M. B., & Rudnicki, W. R. (2010). Feature Selection with the Boruta Package. Journal of Statistical Software.

7. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical Review E.
