#!/bin/bash
#SBATCH --job-name=lob_bench_test
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/lob_bench_test_%j.out
#SBATCH --error=logs/lob_bench_test_%j.err

# Parallel speedup test: run --all with n_workers=64 on pious-snowball-264_matched
cd /home/s5e/satyamaga.s5e/lob_bench
mkdir -p logs

source /home/s5e/satyamaga.s5e/miniforge3/bin/activate
conda activate lob_bench

DATA_DIR="/lus/lfs1aip2/projects/s5e/lob_bench/bench_data/pious-snowball-264_matched"

echo "=== Parallel run (n_workers=64) on $(hostname) at $(date) ==="
echo "CPUs available: $(nproc)"

PYTHONUNBUFFERED=1 python run_bench.py \
    --data_dir "$DATA_DIR" \
    --model_name . \
    --stock GOOG \
    --time_period 2023 \
    --save_dir ./results_test_parallel \
    --all \
    --progress_interval 30 \
    --n_workers 64

echo "=== Completed at $(date) ==="
