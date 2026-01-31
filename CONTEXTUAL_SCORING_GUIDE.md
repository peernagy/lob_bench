# Contextual Scoring Guide

## Quick Start

Contextual scoring evaluates performance within each market regime (e.g., spread conditions) separately, revealing where the model degrades.

```python
from scoring import run_benchmark
from run_bench import DEFAULT_SCORING_CONFIG_CONTEXT, DEFAULT_METRICS

# Run contextual scoring
scores, score_dfs = scoring.run_benchmark(
    loader, 
    DEFAULT_SCORING_CONFIG_CONTEXT, 
    contextual=True
)

# Access per-regime metrics
metrics = scores['ask_volume | spread']['wasserstein']
for regime_id, (point_est, ci, bootstrap) in metrics.items():
    print(f"Regime {regime_id}: {point_est:.4f}")
```

## Configuration Format

```python
DEFAULT_SCORING_CONFIG_CONTEXT = {
    "ask_volume | spread": {
        "fn": lambda m, b: eval.l1_volume(m, b).ask_vol.values,
        "context_fn": lambda m, b: eval.spread(m, b).values,
        "context_config": {
            "discrete": True,  # Or: quantiles, n_bins, thresholds, bin_method
        }
    },
}
```

**Keys:**
- `fn`: Metric to evaluate
- `context_fn`: Variable defining regimes (e.g., spread)
- `context_config`: Binning config for regime boundaries

## Return Structure

```python
# Returns: metric[metric_name][regime_id] = (point_est, ci, bootstrapped)
metric = {
    'wasserstein': {
        0: (0.35, array([0.30, 0.40]), bootstrap_samples),
        1: (0.42, array([0.38, 0.46]), bootstrap_samples),
        2: (0.52, array([0.48, 0.56]), bootstrap_samples),
    }
}
```

## Analysis Examples

### Find Worst Regime
```python
metrics = scores['ask_volume | spread']['wasserstein']
worst = max(metrics.items(), key=lambda x: x[1][0])
print(f"Worst regime: {worst[0]} with loss {worst[1][0]:.4f}")
```

### Compare Regimes with Bootstrap
```python
regime_0_boot = metrics[0][2]  # Bootstrapped samples for regime 0
regime_2_boot = metrics[2][2]  # Bootstrapped samples for regime 2
prob_worse = (regime_2_boot > regime_0_boot.mean()).mean()
print(f"P(regime 2 worse): {prob_worse:.2%}")
```

### View Confidence Intervals
```python
for regime_id in sorted(metrics.keys()):
    est, ci, _ = metrics[regime_id]
    print(f"Regime {regime_id}: {est:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
```

## vs Conditional Scoring

| | Conditional | Contextual |
|---|---|---|
| Output | Single aggregate metric | Per-regime metrics |
| Purpose | Average performance across conditions | Identify failing regimes |
| Aggregation | Weighted sum | None (individual regimes) |

Conditional: "What's average performance?" → Hides regime failures  
Contextual: "Which regimes fail?" → Exposes performance degradation
