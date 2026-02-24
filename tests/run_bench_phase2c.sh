#!/bin/bash
#SBATCH --job-name=lob_bench_p2c
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/lob_bench_phase2c_%j.out
#SBATCH --error=logs/lob_bench_phase2c_%j.err

# Phase 2c: GIL fix + JAX vmap GPU bootstrap (full VRAM, no prealloc limit)
cd /home/s5e/satyamaga.s5e/lob_bench
mkdir -p logs

source /home/s5e/satyamaga.s5e/miniforge3/bin/activate
conda activate lob_bench

# Let JAX allocate GPU memory on-demand instead of pre-reserving 75%
export XLA_PYTHON_CLIENT_PREALLOCATE=false

DATA_DIR="/lus/lfs1aip2/projects/s5e/lob_bench/bench_data/pious-snowball-264_matched"

echo "=== Phase 2c run (n_workers=64, JAX GPU, no prealloc) on $(hostname) at $(date) ==="
echo "CPUs available: $(nproc)"
python -c "import jax; print('JAX devices:', jax.devices())"

PYTHONUNBUFFERED=1 python run_bench.py \
    --data_dir "$DATA_DIR" \
    --model_name . \
    --stock GOOG \
    --time_period 2023 \
    --save_dir ./results_phase2c \
    --all \
    --progress_interval 30 \
    --n_workers 64

echo "=== Completed at $(date) ==="
