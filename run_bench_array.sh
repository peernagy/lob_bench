#!/bin/bash
#SBATCH --job-name=lob_bench
#SBATCH --account=brics.s5e
#SBATCH --array=0-9%8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/lob_bench_%A_%a.out
#SBATCH --error=logs/lob_bench_%A_%a.err
#SBATCH --exclude=nid[010696-010718],nid010152,nid010110,nid[011112-011115],nid011294,nid[010083-010086],nid[010561-010564],nid010052,nid010412,nid011247

# === CONFIG: Edit these arrays ===
STOCKS=(GOOG AAPL NVDA MSFT AMZN META AMD INTC GOOGL TSLA)
MODELS=(pious-snowball-264_matched)
PERIODS=(2023)
DATA_ROOT=/lus/lfs1aip2/projects/s5e/lob_bench/bench_data

# === Decompose array task ID into (stock, model, period) ===
TASK_ID=${SLURM_ARRAY_TASK_ID}
N_PERIODS=${#PERIODS[@]}
N_MODELS=${#MODELS[@]}
PERIOD_IDX=$((TASK_ID % N_PERIODS))
MODEL_IDX=$(((TASK_ID / N_PERIODS) % N_MODELS))
STOCK_IDX=$((TASK_ID / (N_PERIODS * N_MODELS)))

STOCK=${STOCKS[$STOCK_IDX]}
MODEL=${MODELS[$MODEL_IDX]}
PERIOD=${PERIODS[$PERIOD_IDX]}

echo "========================================"
echo "[array] task=$TASK_ID stock=$STOCK model=$MODEL period=$PERIOD"
echo "[array] node=$(hostname) started=$(date)"
echo "========================================"

# Navigate to the working directory
cd /home/s5e/aramis.s5e/lob_bench

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate our lob_bench env (JAX 0.4.35 + CUDA 12)
source /home/s5e/aramis.s5e/miniforge3/bin/activate
conda activate lob_bench
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Run the benchmark for this (stock, model, period) combo
PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 python run_bench.py \
    --data_dir ${DATA_ROOT}/${MODEL} \
    --model_name . \
    --stock ${STOCK} \
    --time_period ${PERIOD} \
    --save_dir ./results \
    --all \
    --n_workers 64 \
    --progress_interval 60

echo "[array] task=$TASK_ID completed=$(date)"
