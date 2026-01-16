# Contextual Scoring System Implementation Guide

## Overview

The contextual scoring system reveals **per-regime performance degradation** by evaluating model performance within specific market conditions (e.g., spread regimes) separately, without aggregating results.

**Key difference from conditional scoring:**
- **Conditional**: Bins by spread, then does secondary binning → weighted average across regimes → hides regime-specific failures
- **Contextual**: Bins by spread → evaluates metrics directly → returns per-regime metrics → exposes which regimes have worse performance

## Architecture

### New Functions Added to `scoring.py`

#### 1. `score_data_context()`
Bins data by contextual regime (e.g., spread levels) without secondary binning.

```python
score_df = score_data_context(
    loader, 
    scoring_fn=lambda m, b: eval.l1_volume(m, b).ask_vol.values,
    scoring_fn_context=lambda m, b: eval.spread(m, b).values,
    score_context_kwargs={"discrete": True}
)
```

**Returns**: DataFrame with columns `[score, group, type, score_context]`
- `score`: The evaluation metric (e.g., ask_volume)
- `group`: Regime ID (e.g., 0=low spread, 1=medium, 2=high)
- `type`: 'real' or 'generated'
- `score_context`: The contextual variable value (spread)

#### 2. `compute_metrics_context()`
Calculates metrics separately per regime without aggregation.

```python
metric, score_df, plot_fn = compute_metrics_context(
    loader,
    scoring_fn=lambda m, b: eval.l1_volume(m, b).ask_vol.values,
    metric_fn={'wasserstein': metrics.wasserstein, 'l1': metrics.l1_by_group},
    scoring_fn_context=lambda m, b: eval.spread(m, b).values,
    score_context_kwargs={"discrete": True}
)
```

**Returns**: 
- `metric`: `dict[metric_name: dict[regime_id: (point_est, ci, bootstrapped)]]`
- `score_df`: Per-regime scores
- `plot_fn`: Plotting function (currently dummy)

### Updated `run_benchmark()` Function

Added `contextual: bool = False` parameter to route to contextual path.

```python
scores, score_dfs, plot_fns = scoring.run_benchmark(
    loader,
    scoring_config=DEFAULT_SCORING_CONFIG_CONTEXT,
    default_metric=DEFAULT_METRICS,
    contextual=True  # NEW PARAMETER
)
```

## Configuration Format

### Contextual Config Structure

```python
DEFAULT_SCORING_CONFIG_CONTEXT = {
    "ask_volume | spread": {
        "fn": lambda m, b: eval.l1_volume(m, b).ask_vol.values,
        "context_fn": lambda m, b: eval.spread(m, b).values,
        "context_config": {
            "discrete": True,
            # Optional: quantiles, n_bins, thresholds, bin_method
        }
    },
}
```

**Keys:**
- `fn`: Evaluation metric function
- `context_fn`: Contextual variable for regime definition (e.g., spread)
- `context_config`: Binning config for regimes (see `get_kwargs()`)

### Available Binning Options

From `get_kwargs(..., conditional=True)`:
- `discrete`: Use unique values as bins
- `quantiles`: List of quantiles, e.g., `[0.25, 0.5, 0.75]` (default: deciles)
- `n_bins`: Number of bins
- `thresholds`: Explicit threshold array
- `bin_method`: Method like 'fd' (Freedman-Diaconis)

## Return Structure & Analysis

### Return Format

```python
metric = {
    'wasserstein': {
        0: (0.35, np.array([0.30, 0.40]), bootstrapped_array),  # Low spread regime
        1: (0.42, np.array([0.38, 0.46]), bootstrapped_array),  # Medium spread
        2: (0.52, np.array([0.48, 0.56]), bootstrapped_array),  # High spread
    },
    'l1': {
        0: (0.28, ..., ...),
        1: (0.35, ..., ...),
        2: (0.45, ..., ...),
    }
}
```

### Example: Extracting Per-Regime Performance

```python
# Get Wasserstein metric by regime
metrics = scores['ask_volume | spread']['wasserstein']

for regime_id, (point_est, ci, bootstrapped) in metrics.items():
    print(f"Regime {regime_id}:")
    print(f"  Point estimate: {point_est:.4f}")
    print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  Worse performance in this regime? {point_est > threshold}")
```

### Comparative Analysis Pattern

```python
# Compare regimes for a single metric
wasserstein_by_regime = scores['ask_volume | spread']['wasserstein']

# Find worst-performing regime
worst_regime = max(
    wasserstein_by_regime.items(),
    key=lambda x: x[1][0]  # x[1][0] is point estimate
)
print(f"Worst regime: {worst_regime[0]} with loss {worst_regime[1][0]:.4f}")

# Identify degradation pattern
degradation = {
    regime: est for regime, (est, _, _) in wasserstein_by_regime.items()
}
print("Performance degradation pattern:", degradation)
```

## Key Differences from Conditional Scoring

| Aspect | Conditional | Contextual |
|--------|-------------|-----------|
| **Grouping** | Two-level: regime → secondary bins | Single-level: regime only |
| **Aggregation** | Weighted sum across regimes | Per-regime (no aggregation) |
| **Returns** | Single tuple `(est, ci, boot)` per metric | Dict of regimes, each with tuple |
| **Visibility** | Hides per-regime failures | Exposes per-regime degradation |
| **Use Case** | Overall performance in conditions | Which conditions break the model? |

## Usage in run_bench.py

The contextual scoring is automatically called if enabled:

```python
python run_bench.py --model_name s5_main --stock GOOG --time_period 2023
```

Results saved to:
```
results/scores/scores_context_GOOG_s5_main_YYYYMMDD_HHMMSS.pkl
```

Load results:
```python
from run_bench import load_results
metric, score_dfs = load_results('results/scores/scores_context_GOOG_s5_main_20240115_120000.pkl')
```

## Analysis Examples

### Example 1: Identify Worst Market Condition

```python
metric, score_dfs = load_results('results/scores/scores_context_GOOG_s5_main.pkl')

# For ask_volume performance by spread regime
ask_vol_metrics = metric['ask_volume | spread']['wasserstein']

print("Model performance degradation by spread regime:")
for regime_id in sorted(ask_vol_metrics.keys()):
    point_est, ci, _ = ask_vol_metrics[regime_id]
    print(f"  Spread Regime {regime_id}: {point_est:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
```

### Example 2: Statistical Comparison Between Regimes

```python
regime_0_boot = ask_vol_metrics[0][2]  # bootstrapped losses for regime 0
regime_2_boot = ask_vol_metrics[2][2]  # bootstrapped losses for regime 2

# Compare regimes using bootstrap samples
difference = regime_2_boot - regime_0_boot
prob_worse = (difference > 0).mean()
print(f"Probability regime 2 is worse than regime 0: {prob_worse:.2%}")
```

### Example 3: Find Consistent Issues

```python
# Compare across multiple metrics in a specific regime
high_spread_regime = 2
bad_metrics = {}

for metric_name in ['wasserstein', 'l1']:
    metric_by_regime = metric[f'ask_volume | spread'][metric_name]
    point_est = metric_by_regime[high_spread_regime][0]
    bad_metrics[metric_name] = point_est

print(f"High-spread regime ({high_spread_regime}) performance:")
for metric_name, value in bad_metrics.items():
    print(f"  {metric_name}: {value:.4f}")
```

## Backward Compatibility

✅ **Fully backward compatible**
- Existing conditional/unconditional paths unchanged
- Contextual is opt-in via `contextual=True` parameter
- Separate results (different pickle files)
- Can run both conditional and contextual simultaneously

## Next Steps

1. Run contextual scoring: `python run_bench.py --model_name s5_main --stock GOOG`
2. Load results with `load_results()` 
3. Extract per-regime metrics using the return structure
4. Analyze which market conditions cause degradation
5. Use insights to improve model robustness
