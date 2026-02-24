# Contextual Scoring System - Implementation Summary

## Overview

Contextual scoring evaluates model performance separately within each market regime (defined by conditional data), revealing where the model degrades rather than averaging performance across conditions.

**Core insight**: Instead of aggregating metrics across spread regimes, contextual scoring returns `dict[regime_id: (point_est, ci, bootstrapped)]` to expose regime-specific failures.

## What Was Added

1. **`score_data_context()`** - Defines regimes from conditional data, evaluates metrics within each regime
2. **`compute_metrics_context()`** - Calculates metrics per-regime without aggregation
3. **`run_benchmark(..., contextual=True)`** - Routes to contextual path
4. **`DEFAULT_SCORING_CONFIG_CONTEXT`** in run_bench.py - Configuration format

## Return Structure

```python
metric = {
    'wasserstein': {
        0: (0.35, array([0.30, 0.40]), bootstrapped),  # Low spread
        1: (0.42, array([0.38, 0.46]), bootstrapped),  # Medium
        2: (0.52, array([0.48, 0.56]), bootstrapped),  # High spread
    }
}
# Each regime: (point_est, confidence_interval, bootstrap_samples)
```

## Usage Example

```python
# Run contextual scoring
scores, score_dfs = scoring.run_benchmark(
    loader, 
    DEFAULT_SCORING_CONFIG_CONTEXT, 
    contextual=True
)

# Extract per-regime performance
metrics = scores['ask_volume | spread']['wasserstein']
for regime_id, (point_est, ci, boot) in metrics.items():
    print(f"Regime {regime_id}: {point_est:.4f}")
```

## Key Difference from Conditional Scoring

| | Conditional | Contextual |
|---|---|---|
| Aggregation | Weighted sum across regimes | Per-regime (no aggregation) |
| Returns | Single metric value | Dict of metrics by regime |
| Purpose | Overall performance | Identify which regimes fail |

## Backward Compatible

✅ All existing paths unchanged  
✅ Contextual is opt-in via `contextual=True`  
✅ Results saved separately  
