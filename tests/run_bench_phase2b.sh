#!/bin/bash
#SBATCH --job-name=lob_bench_p2b
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/lob_bench_phase2b_%j.out
#SBATCH --error=logs/lob_bench_phase2b_%j.err

# Phase 2b: GIL fix + JAX vmap GPU bootstrap + OOM guard (2M sample limit)
cd /home/s5e/satyamaga.s5e/lob_bench
mkdir -p logs

source /home/s5e/satyamaga.s5e/miniforge3/bin/activate
conda activate lob_bench

DATA_DIR="/lus/lfs1aip2/projects/s5e/lob_bench/bench_data/pious-snowball-264_matched"

echo "=== Phase 2b run (n_workers=64, JAX GPU + OOM guard) on $(hostname) at $(date) ==="
echo "CPUs available: $(nproc)"
python -c "import jax; print('JAX devices:', jax.devices())"

PYTHONUNBUFFERED=1 python run_bench.py \
    --data_dir "$DATA_DIR" \
    --model_name . \
    --stock GOOG \
    --time_period 2023 \
    --save_dir ./results_phase2b \
    --all \
    --progress_interval 30 \
    --n_workers 64

echo "=== Completed at $(date) ==="
