# Data Pipeline

End-to-end flow from raw LOBSTER data to benchmark scores and plots.

## Pipeline Overview

```
Raw LOBSTER CSVs
       |
       v
[1] Preprocessing (LOBS5/lob/preproc_lob.py)
       |  Converts raw exchange data to .npy arrays
       |  Normalizes book to 40-dim (10 levels x 2 sides x {price, vol})
       v
Preprocessed .npy data
       |
       v
[2] Training (LOBS5/run_train.py)
       |  Tokenizes messages via Vocab/Message_Tokenizer (22 or 24 tokens)
       |  Trains S5 autoregressive model on token sequences
       v
Trained checkpoint (Orbax format)
       |
       v
[3] Inference (pipeline/_infer.batch → LOBS5/run_inference.py)
       |  Loads checkpoint, selects test sequences
       |  For each sequence:
       |    - Feeds conditioning prefix (N_COND_MSGS messages)
       |    - Autoregressively generates N_GEN_MSGS messages
       |    - JaxLOB simulator reconstructs orderbook state per message
       |  Outputs: data_real/, data_gen/, data_cond/ CSV directories
       v
Inference results (LOBSTER-format CSVs)
       |
       +---> [4a] Distributional Scoring
       |          (lob_bench/run_bench.py)
       |
       +---> [4b] Impact Analysis
                   (lob_bench/impact.py)
```

## Stage Details

### [3] Inference Output Structure

The inference step produces three directories per stock/checkpoint:

```
inference_results/{RUN_NAME}_{STOCK}_{JOB_ID}/
    data_real/
        {DATE}_message_real_id_{N}.csv
        {DATE}_orderbook_real_id_{N}.csv
    data_gen/
        {DATE}_message_real_id_{N}_gen_id_{M}.csv
        {DATE}_orderbook_real_id_{N}_gen_id_{M}.csv
    data_cond/
        {DATE}_message_real_id_{N}.csv
        {DATE}_orderbook_real_id_{N}.csv
```

- `real_id` identifies the test sequence
- `gen_id` identifies the sample (multiple samples per sequence)
- `data_cond/` holds the conditioning prefix used for generation

### [4a] Distributional Scoring

```
data_real/ + data_gen/ + data_cond/
       |
       v
Simple_Loader (data_loading.py)
  Groups real/gen/cond CSVs by (date, real_id)
  Returns Lobster_Sequence objects with lazy-loaded DataFrames
       |
       v
Score Functions (eval.py)
  For each sequence, compute 1-D statistics:
    spread, volumes, depths, levels, OFI, interarrival times, etc.
  Returns arrays of score values per sequence
       |
       v
Partitioning (partitioning.py)
  Bins score values into groups (Freedman-Diaconis, quantiles, discrete)
  Creates score_df: DataFrame[score, group, type={real|generated}]
       |
       v
Metrics (metrics.py)
  For each score_df, compute:
    L1 distance, Wasserstein distance, KS statistic
  Bootstrap CI: resample real+gen, recompute metric (n=100)
       |
       v
Scoring Engine (scoring.py)
  Orchestrates unconditional/conditional/contextual/time-lagged/divergence modes
  Returns: {score_name: {metric_name: (point_est, CI, bootstrap_samples)}}
       |
       v
Saved as gzipped pickle: results/scores/scores_{mode}_{stock}_{model}_{timestamp}.pkl
       |
       v
Plotting (run_plotting.py → plotting.py)
  Loads .pkl files
  Generates: summary bar plots, spider plots, per-metric histograms,
             conditional facet grids, divergence curves
```

### [4b] Impact Analysis

```
Simple_Loader
       |
       v
Filter touch events (impact.py)
  Keep messages where book top-of-book changes
  Classify into 6 Bouchaud event types:
    MO_0/1 (market orders), LO_0/1 (limit orders), CA_0/1 (cancellations)
    _0 = no price change, _1 = price change
       |
       v
Response Functions
  For each event type and lag l = 1..1000:
    R(l) = mean midprice change from t to t+l, signed by trade direction
  Computed separately for real and generated sequences
       |
       v
Output: response curves + sum of absolute differences (real vs gen)
```

## Full Automation (pipeline/)

The pipeline orchestrator (`run_lobbench_pipeline.sh`) automates stages [3] and [4]:

```bash
./pipeline/run_lobbench_pipeline.sh <CKPT_PATH> --stocks "GOOG INTC" --name my_run
```

1. Extracts HF-matched sample indices (or uses random sampling with `--no_hf_compare`)
2. Submits GPU inference job (`_infer.batch`) via SLURM
3. Chains CPU scoring job (`_bench.batch`) as `--dependency=afterok`
4. Results land in `{RESULTS_BASE_DIR}/results_{NAME}/`

Configuration lives in `pipeline/config.sh` (copy from `config.sh.template`).
