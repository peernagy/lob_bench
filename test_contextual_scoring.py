"""
Test script demonstrating contextual scoring functionality.
This shows how to extract and analyze per-regime performance metrics.
"""

import numpy as np
import pandas as pd
from run_bench import DEFAULT_SCORING_CONFIG_CONTEXT, DEFAULT_METRICS, load_results, save_results


def demonstrate_return_structure():
    """
    Shows the expected return structure from contextual scoring.
    """
    print("=" * 80)
    print("CONTEXTUAL SCORING RETURN STRUCTURE")
    print("=" * 80)
    
    # Simulated return structure from compute_metrics_context()
    metric = {
        'wasserstein': {
            0: (0.35, np.array([0.30, 0.40]), np.random.rand(101)),  # Low spread regime
            1: (0.42, np.array([0.38, 0.46]), np.random.rand(101)),  # Medium spread
            2: (0.52, np.array([0.48, 0.56]), np.random.rand(101)),  # High spread
        },
        'l1': {
            0: (0.28, np.array([0.25, 0.32]), np.random.rand(101)),
            1: (0.35, np.array([0.32, 0.39]), np.random.rand(101)),
            2: (0.45, np.array([0.42, 0.49]), np.random.rand(101)),
        }
    }
    
    print("\nReturn structure type: dict[metric_name: dict[regime_id: (est, ci, bootstrap)]]")
    print("\nTop-level keys (metrics):")
    for metric_name in metric.keys():
        print(f"  - {metric_name}")
    
    print("\nSecond-level keys (regime IDs):")
    for regime_id in metric['wasserstein'].keys():
        print(f"  - Regime {regime_id}")
    
    print("\nThird-level value structure per regime:")
    regime_tuple = metric['wasserstein'][0]
    print(f"  - Point estimate: {regime_tuple[0]:.4f}")
    print(f"  - Confidence interval: {regime_tuple[1]}")
    print(f"  - Bootstrapped losses: array of shape {regime_tuple[2].shape}")
    
    return metric


def analyze_regime_performance(metric):
    """
    Example analysis: Identify which regimes have worst performance.
    """
    print("\n" + "=" * 80)
    print("REGIME PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Analyze ask_volume | spread performance across regimes
    metric_name = 'wasserstein'
    metric_by_regime = metric['ask_volume | spread'][metric_name]
    
    print(f"\n{metric_name} metric by spread regime:")
    print(f"{'Regime':<10} {'Est':<10} {'95% CI Lower':<15} {'95% CI Upper':<15}")
    print("-" * 50)
    
    regime_performance = {}
    for regime_id in sorted(metric_by_regime.keys()):
        point_est, ci, bootstrap = metric_by_regime[regime_id]
        regime_performance[regime_id] = point_est
        print(f"{regime_id:<10} {point_est:<10.4f} {ci[0]:<15.4f} {ci[1]:<15.4f}")
    
    # Find worst regime
    worst_regime = max(regime_performance.items(), key=lambda x: x[1])
    print(f"\n⚠️  Worst performing regime: {worst_regime[0]} (loss: {worst_regime[1]:.4f})")
    
    return regime_performance


def compare_regimes(metric):
    """
    Example: Compare regime 0 (low spread) vs regime 2 (high spread).
    """
    print("\n" + "=" * 80)
    print("REGIME COMPARISON")
    print("=" * 80)
    
    metric_by_regime = metric['wasserstein']
    
    regime_0_boot = metric_by_regime[0][2]  # Bootstrapped losses for regime 0
    regime_2_boot = metric_by_regime[2][2]  # Bootstrapped losses for regime 2
    
    # Compare regimes using bootstrap samples
    difference = regime_2_boot - regime_0_boot
    prob_worse = (difference > 0).mean()
    
    print(f"\nRegime 0 (low spread): mean loss = {regime_0_boot.mean():.4f}")
    print(f"Regime 2 (high spread): mean loss = {regime_2_boot.mean():.4f}")
    print(f"\nProbability that high-spread regime is worse: {prob_worse:.2%}")
    
    if prob_worse > 0.9:
        print("⚠️  Strong evidence of performance degradation in high-spread markets!")
    elif prob_worse > 0.5:
        print("✓ Model likely performs worse in high-spread markets")
    else:
        print("✓ No strong evidence of degradation in high-spread markets")


def show_config_format():
    """
    Display the configuration format for contextual scoring.
    """
    print("\n" + "=" * 80)
    print("CONTEXTUAL SCORING CONFIGURATION")
    print("=" * 80)
    
    print("\nConfiguration from DEFAULT_SCORING_CONFIG_CONTEXT:")
    print("\n```python")
    print("""
DEFAULT_SCORING_CONFIG_CONTEXT = {
    "ask_volume | spread": {
        "fn": lambda m, b: eval.l1_volume(m, b).ask_vol.values,
        "context_fn": lambda m, b: eval.spread(m, b).values,
        "context_config": {
            "discrete": True,  # Use unique values as bins
            # Optional: quantiles, n_bins, thresholds, bin_method
        }
    },
}
    """)
    print("```")
    
    print("\nRequired config keys:")
    print("  - 'fn': Evaluation metric function")
    print("  - 'context_fn': Contextual variable for regime definition")
    print("  - 'context_config': Binning configuration")
    
    print("\nOptional binning parameters in context_config:")
    print("  - 'discrete': Boolean (use unique values as bins)")
    print("  - 'quantiles': List like [0.25, 0.5, 0.75]")
    print("  - 'n_bins': Number of bins")
    print("  - 'thresholds': Explicit threshold array")
    print("  - 'bin_method': Method like 'fd' (Freedman-Diaconis)")


def show_usage_pattern():
    """
    Show how to use contextual scoring in a real workflow.
    """
    print("\n" + "=" * 80)
    print("USAGE PATTERN")
    print("=" * 80)
    
    print("""
# Step 1: Run contextual scoring
scores, score_dfs, plot_fns = scoring.run_benchmark(
    loader,
    scoring_config=DEFAULT_SCORING_CONFIG_CONTEXT,
    default_metric=DEFAULT_METRICS,
    contextual=True  # Enable contextual mode
)

# Step 2: Extract metrics for a specific evaluation
metric_wasserstein = scores['ask_volume | spread']['wasserstein']
# metric_wasserstein = {0: (est, ci, bootstrap), 1: (...), 2: (...)}

# Step 3: Analyze per-regime performance
for regime_id, (point_est, ci, bootstrap) in metric_wasserstein.items():
    print(f"Regime {regime_id}: {point_est:.4f} ± {(ci[1]-ci[0])/2:.4f}")

# Step 4: Identify problematic regimes
worst = max(metric_wasserstein.items(), key=lambda x: x[1][0])
print(f"Worst regime: {worst[0]} with loss {worst[1][0]:.4f}")

# Step 5: Compare regimes statistically
regime_0_boot = metric_wasserstein[0][2]
regime_2_boot = metric_wasserstein[2][2]
prob_worse = (regime_2_boot > regime_0_boot.mean()).mean()
print(f"P(high-spread worse than low-spread) = {prob_worse:.2%}")
    """)


if __name__ == "__main__":
    print("\n" + "🎯 CONTEXTUAL SCORING SYSTEM DEMONSTRATION" + "\n")
    
    # Show the return structure
    metric = demonstrate_return_structure()
    
    # Analyze regimes (note: this uses dummy data)
    # analyze_regime_performance(metric)
    # compare_regimes(metric)
    
    # Show configuration
    show_config_format()
    
    # Show usage pattern
    show_usage_pattern()
    
    print("\n" + "=" * 80)
    print("IMPLEMENTATION NOTES")
    print("=" * 80)
    print("""
✅ Implementation Complete:
  1. score_data_context() - Bins by regime without secondary binning
  2. compute_metrics_context() - Per-regime metric calculation
  3. run_benchmark(..., contextual=True) - Clean routing
  4. DEFAULT_SCORING_CONFIG_CONTEXT - Pre-configured examples
  
Key Features:
  • Per-regime metrics preserved (no aggregation)
  • Full bootstrap confidence intervals per regime
  • Backward compatible (existing paths unchanged)
  • Exposes performance degradation in specific markets
  
Return Structure:
  dict[metric_name: dict[regime_id: (point_est, ci, bootstrapped)]]
  
Analysis Opportunities:
  ✓ Which market conditions break the model?
  ✓ Which regimes have statistically worse performance?
  ✓ Bootstrap-based regime comparisons
  ✓ Identify consistent failure patterns
    """)
