# Contextual Scoring System - Implementation Summary

## ✅ Implementation Complete

The contextual scoring system has been successfully implemented to expose context-dependent performance degradation across different market regimes (e.g., spread conditions).

## What Was Implemented

### 1. **`score_data_context()` function** in [scoring.py](scoring.py#L144)
- Bins data by contextual regime (e.g., spread levels) 
- No secondary binning (unlike conditional scoring)
- Returns DataFrame with columns: `[score, group, type, score_context]`
- Uses `partitioning.score_real_gen()` and `partitioning.group_by_score()` following existing patterns

### 2. **`compute_metrics_context()` function** in [scoring.py](scoring.py#L191)
- Calculates metrics **separately per regime** without aggregation
- Iterates through each regime via `score_df.groupby('group')`
- Returns: `dict[metric_name: dict[regime_id: (point_est, ci, bootstrapped_losses)]]`
- Preserves full bootstrap information per regime for statistical testing

### 3. **Updated `run_benchmark()` function** in [scoring.py](scoring.py#L434)
- Added `contextual: bool = False` parameter
- Routes to `compute_metrics_context()` when `contextual=True`
- Existing unconditional/conditional/divergence paths remain unchanged
- Clean separation prevents interference with existing code

### 4. **Configuration Setup** in [run_bench.py](run_bench.py#L148)
- `DEFAULT_SCORING_CONFIG_CONTEXT` with proper format
- Required keys: `fn`, `context_fn`, `context_config`
- Optional binning: `discrete`, `quantiles`, `n_bins`, `thresholds`, `bin_method`
- Example configs for `ask_volume | spread` and `bid_volume_touch | spread`

### 5. **Result Handling** in [run_bench.py](run_bench.py#L196)
- Added `scoring_config_context` parameter to `run_benchmark()`
- Contextual scoring results saved with `_context` suffix
- Separate pickle files prevent mixing with conditional results
- Maintains backward compatibility with existing workflows

## Return Structure

```python
metric = {
    'wasserstein': {
        0: (0.35, np.array([0.30, 0.40]), bootstrapped_array),  # Low spread regime
        1: (0.42, np.array([0.38, 0.46]), bootstrapped_array),  # Medium spread regime
        2: (0.52, np.array([0.48, 0.56]), bootstrapped_array),  # High spread regime
    },
    'l1': {
        0: (0.28, np.array([0.25, 0.32]), bootstrapped_array),
        1: (0.35, np.array([0.32, 0.39]), bootstrapped_array),
        2: (0.45, np.array([0.42, 0.49]), bootstrapped_array),
    }
}
```

Each regime has independent `(point_estimate, confidence_interval, bootstrapped_losses)` tuple.

## Key Differences from Conditional Scoring

| Aspect | Conditional | Contextual |
|--------|-------------|-----------|
| Grouping | Two-level (regime → sub-bins) | Single-level (regime only) |
| Aggregation | Weighted sum across regimes | **Per-regime (no aggregation)** |
| Return type | `dict[metric: (est, ci, boot)]` | **`dict[metric: dict[regime: (est, ci, boot)]]`** |
| Purpose | Overall performance given conditions | **Which conditions break the model?** |
| Visibility | Hides regime-specific failures | **Exposes regime-specific degradation** |

## Example Usage

```python
from scoring import run_benchmark
from run_bench import DEFAULT_SCORING_CONFIG_CONTEXT, DEFAULT_METRICS

# Run contextual scoring
scores, score_dfs, plot_fns = run_benchmark(
    loader,
    scoring_config=DEFAULT_SCORING_CONFIG_CONTEXT,
    default_metric=DEFAULT_METRICS,
    contextual=True  # Enable contextual mode
)

# Extract per-regime metrics
metric_by_regime = scores['ask_volume | spread']['wasserstein']

# Analyze: Which regime is worst?
for regime_id, (point_est, ci, bootstrap) in metric_by_regime.items():
    print(f"Regime {regime_id}: {point_est:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")

# Statistical comparison: Is high-spread worse than low-spread?
regime_0_boot = metric_by_regime[0][2]
regime_2_boot = metric_by_regime[2][2]
prob_worse = (regime_2_boot > regime_0_boot.mean()).mean()
print(f"P(high-spread worse) = {prob_worse:.2%}")
```

## File Changes

### Modified Files

1. **[scoring.py](scoring.py)**
   - Lines 144-189: Added `score_data_context()`
   - Lines 191-242: Added `compute_metrics_context()`
   - Line 439: Added `contextual: bool = False` parameter to `run_benchmark()`
   - Lines 454-474: Added contextual routing logic

2. **[run_bench.py](run_bench.py)**
   - Lines 148-171: Added `DEFAULT_SCORING_CONFIG_CONTEXT` configuration
   - Line 195: Added `scoring_config_context` parameter
   - Line 210: Default `scoring_config_context = DEFAULT_SCORING_CONFIG_CONTEXT`
   - Lines 261-278: Added contextual scoring call with result saving

### New Files

1. **[CONTEXTUAL_SCORING_GUIDE.md](CONTEXTUAL_SCORING_GUIDE.md)** - Complete usage guide with examples
2. **[test_contextual_scoring.py](test_contextual_scoring.py)** - Demonstration script showing return structure and analysis patterns

## Backward Compatibility

✅ **Fully backward compatible**
- All existing code paths unchanged
- Unconditional scoring: unchanged
- Conditional scoring: unchanged  
- Divergence analysis: unchanged
- Contextual is **opt-in** via `contextual=True`
- Results stored separately (different pickle files)

## Analysis Capabilities

The per-regime structure enables:

1. **Identify problematic regimes**: Find which spread conditions cause model failure
2. **Statistical comparison**: Bootstrap-based regime comparisons
3. **Degradation patterns**: Understand how performance changes across market conditions
4. **Confidence intervals**: Full uncertainty quantification per regime
5. **Downstream analysis**: Load results and conduct custom regime analysis

## Next Steps for Users

1. **Run contextual scoring**:
   ```bash
   python run_bench.py --model_name s5_main --stock GOOG --time_period 2023
   ```

2. **Load and analyze results**:
   ```python
   from run_bench import load_results
   metric, score_dfs = load_results('results/scores/scores_context_GOOG_s5_main_*.pkl')
   ```

3. **Extract per-regime insights**:
   - See [CONTEXTUAL_SCORING_GUIDE.md](CONTEXTUAL_SCORING_GUIDE.md) for analysis patterns

4. **Compare with conditional results**:
   - Conditional: What is average performance given conditions?
   - Contextual: Which conditions cause performance degradation?

## Technical Notes

- **Binning**: Uses `partitioning.group_by_score()` with configurable methods
- **Metrics**: Works with both conditional (tuple) and unconditional (scalar) metric functions
- **Bootstrap**: Preserves full bootstrap distributions for statistical testing
- **Confidence intervals**: Computed per-regime via percentiles (default: 99% CI)

## Validation

✅ Python compilation: No syntax errors  
✅ Function signatures: Match expected patterns  
✅ Return structures: Nested dicts with regime-level granularity  
✅ Configuration: Proper key structure and defaults  
✅ Integration: Clean routing in run_benchmark()  
✅ Backward compatibility: Existing paths unchanged  
