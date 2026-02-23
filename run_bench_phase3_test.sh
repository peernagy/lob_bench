#!/bin/bash
#SBATCH --job-name=bench_p3test
#SBATCH --account=brics.s5e
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/s5e/aramis.s5e/lob_bench/logs/bench_p3test_%j.out
#SBATCH --error=/home/s5e/aramis.s5e/lob_bench/logs/bench_p3test_%j.err
#SBATCH --exclude=nid[010696-010718],nid010152,nid010110,nid[011112-011115],nid011294,nid[010083-010086],nid[010561-010564],nid010052,nid010412,nid011247

# Phase 3 validation test:
#   - Subsample: expect "[metrics] Bootstrap subsample: 1881600 → 50000"
#   - JAX vmap: expect JAX GPU activation (not "skipped")
#   - ThreadPoolExecutor: expect "[scoring] Parallelizing N metrics across 4 threads"
#   - n_workers: expect loky parallelism for sequence scoring

cd /home/s5e/aramis.s5e/lob_bench
mkdir -p logs

source /home/s5e/aramis.s5e/miniforge3/bin/activate
conda activate lob_bench
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo "=== Phase 3 Test: unconditional + divergence on GOOG ==="
echo "Started: $(date)"

# Run unconditional (20 metrics → tests ThreadPoolExecutor) + divergence
PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 python run_bench.py \
    --data_dir /lus/lfs1aip2/projects/s5e/lob_bench/bench_data/pious-snowball-264_matched \
    --model_name . \
    --stock GOOG \
    --time_period 2023 \
    --save_dir ./results_phase3_test \
    --unconditional \
    --divergence \
    --n_workers 64 \
    --progress_interval 30

echo "Completed: $(date)"
