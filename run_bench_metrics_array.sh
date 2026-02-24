#!/bin/bash
#SBATCH --job-name=lob_metrics
#SBATCH --account=brics.s5e
#SBATCH --array=0-20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/lob_metrics_%A_%a.out
#SBATCH --error=logs/lob_metrics_%A_%a.err
#SBATCH --exclude=nid[010696-010718],nid010152,nid010110,nid[011112-011115],nid011294,nid[010083-010086],nid[010561-010564]

# LOBbench Phase 4: metric partitioning across SLURM array tasks.
# Each task scores a disjoint subset of unconditional metrics.
# After all tasks complete, merge with:
#   python merge_shards.py "results/scores/scores_uncond_*_shard*_${SLURM_ARRAY_JOB_ID}.pkl" \
#       -o results/scores/scores_uncond_GOOG_._merged.pkl
#   python merge_shards.py "results/scores/scores_div_*_shard*_${SLURM_ARRAY_JOB_ID}.pkl" \
#       -o results/scores/scores_div_GOOG_._merged.pkl

# === CONFIG (edit these) ===
STOCK=GOOG
MODEL=pious-snowball-264_matched
PERIOD=2023
DATA_ROOT=/lus/lfs1aip2/projects/s5e/lob_bench/bench_data
N_SHARDS=21    # must match --array range (0 to N_SHARDS-1)

# All 21 unconditional metrics from DEFAULT_SCORING_CONFIG
ALL_METRICS="spread,orderbook_imbalance,log_inter_arrival_time,log_time_to_cancel,\
ask_volume_touch,bid_volume_touch,ask_volume,bid_volume,\
limit_ask_order_depth,limit_bid_order_depth,ask_cancellation_depth,bid_cancellation_depth,\
limit_ask_order_levels,limit_bid_order_levels,ask_cancellation_levels,bid_cancellation_levels,\
vol_per_min,ofi,ofi_up,ofi_stay,ofi_down"

# === Compute metric subset for this array task ===
METRICS=$(python3 -c "
metrics = '${ALL_METRICS}'.replace('\\\\', '').split(',')
metrics = [m.strip() for m in metrics if m.strip()]
n, i = ${N_SHARDS}, ${SLURM_ARRAY_TASK_ID}
shard = [m for j, m in enumerate(metrics) if j % n == i]
print(','.join(shard))
")

echo "========================================"
echo "[shard] task=${SLURM_ARRAY_TASK_ID}/${N_SHARDS} metrics=${METRICS}"
echo "[shard] node=$(hostname) started=$(date)"
echo "========================================"

cd /home/s5e/satyamaga.s5e/lob_bench
mkdir -p logs

source /home/s5e/satyamaga.s5e/miniforge3/bin/activate
conda activate lob_bench

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 python run_bench.py \
    --data_dir "${DATA_ROOT}/${MODEL}" \
    --model_name . \
    --stock "${STOCK}" \
    --time_period "${PERIOD}" \
    --save_dir ./results \
    --unconditional --divergence \
    --metrics "${METRICS}" \
    --run_id "${SLURM_ARRAY_JOB_ID}" \
    --shard_id "${SLURM_ARRAY_TASK_ID}" \
    --progress_interval 60

echo "[shard] task=${SLURM_ARRAY_TASK_ID} completed=$(date)"
