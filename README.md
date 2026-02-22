# LOB-Bench

Benchmarking library for evaluating generative **Limit Order Book** models against real LOBSTER-format data. Compares 20+ distributional metrics with bootstrap confidence intervals.

**Paper:** [LOB-Bench: Benchmarking Generative AI for Finance](https://arxiv.org/abs/2502.09172)

## Setup

```bash
conda create -n lob_bench python=3.11
conda activate lob_bench
pip install -r requirements-fixed.txt
```

## Data Format

Expects LOBSTER-format CSV directories:
```
{DATA_DIR}/{MODEL}/{STOCK}/{TIME_PERIOD}/
    data_real/    *message*.csv + *orderbook*.csv
    data_gen/     *message*.csv (with real_id_X_gen_id_Y naming)
    data_cond/    *message*.csv + *orderbook*.csv (conditioning sequences)
```

Messages: 6 columns (time, event_type, order_id, size, price, direction).
Orderbook: 40 columns (10 levels x 2 sides x {price, volume}).

## Running Benchmarks

```bash
# All scoring modes
python run_bench.py \
    --data_dir /path/to/eval_data \
    --model_name s5_main \
    --stock GOOG \
    --time_period 2023_Jan \
    --save_dir ./results \
    --all

# Specific modes
python run_bench.py ... --unconditional --divergence

# SLURM array (parallel across metrics)
sbatch run_bench_metrics_array.sh

# Merge shards after array job
python merge_shards.py "results/scores/scores_uncond_*_shard*_<JOB_ID>.pkl" \
    -o results/scores/scores_uncond_GOOG_merged.pkl
```

### Scoring Modes

| Flag | Mode | Description |
|------|------|-------------|
| `--unconditional` | Marginal | 21 score functions (spread, volumes, depths, OFI, etc.) |
| `--conditional` | Conditional | Metric A given metric B (e.g. volume \| spread) |
| `--context` | Contextual | Regime-aware: bins by conditioning data, evaluates per-regime |
| `--time_lagged` | Time-lagged | Metric at _t_ conditioned on metric at _t - lag_ |
| `--divergence` | Divergence | Distribution drift across prediction horizons |

## Plotting

```bash
# Summary plots from saved scores
python run_plotting.py --score_dir ./results/scores --plot_dir ./results/plots --summary_only

# Full plots including per-metric histograms
python run_plotting.py --score_dir ./results/scores --histograms
```

## Impact Analysis

```bash
python impact.py \
    --stock GOOG --data_dir /path/to/data --model_name s5_main \
    --save_dir ./results \
    --micro_calculate --micro_plot
```

Computes Bouchaud-style price-response curves for 6 event types (MO_0/1, LO_0/1, CA_0/1).

## Metrics

All scoring modes produce distances measured by:
- **L1** (histogram bin divergence)
- **Wasserstein** (earth mover's, normalized)
- **KS** (Kolmogorov-Smirnov statistic)

Each includes bootstrap CIs (default n=100).

## File Structure

```
data_loading.py        # LOBSTER CSV loader (Simple_Loader, Lobster_Sequence)
eval.py                # Score functions (spread, volumes, depths, OFI, interarrival, etc.)
scoring.py             # Scoring engine (runs all modes, bootstrap, summary stats)
partitioning.py        # Binning, grouping, score table construction
metrics.py             # Distance metrics (L1, Wasserstein, KS) with bootstrap
plotting.py            # Visualization (histograms, spider plots, divergence curves)
impact.py              # Market impact response functions
run_bench.py           # CLI: run scoring
run_plotting.py        # CLI: generate plots from saved scores
merge_shards.py        # Merge SLURM array shard outputs
tests/                 # Test and experiment scripts
```

## Citing

```bibtex
@misc{nagy2025lobbenchbenchmarkinggenerativeai,
  title   = {LOB-Bench: Benchmarking Generative AI for Finance -- an Application to Limit Order Book Data},
  author  = {Peer Nagy and Sascha Frey and Kang Li and Bidipta Sarkar and Svitlana Vyetrenko and Stefan Zohren and Ani Calinescu and Jakob Foerster},
  year    = {2025},
  eprint  = {2502.09172},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url     = {https://arxiv.org/abs/2502.09172}
}
```
