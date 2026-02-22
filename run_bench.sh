#!/bin/bash
#SBATCH --job-name=lob_bench
#SBATCH --account=brics.s5e
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/s5e/aramis.s5e/lob_bench/logs/lob_bench_%j.out
#SBATCH --error=/home/s5e/aramis.s5e/lob_bench/logs/lob_bench_%j.err
#SBATCH --exclude=nid[010696-010718],nid010152,nid010110,nid[011112-011115],nid011294,nid[010083-010086],nid[010561-010564]

# Navigate to the working directory
cd /home/s5e/aramis.s5e/lob_bench

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate our lob_bench env (JAX 0.4.35 + CUDA 12)
source /home/s5e/aramis.s5e/miniforge3/bin/activate
conda activate lob_bench
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Run the benchmark
echo "Starting benchmark run at $(date)"
PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 python run_bench.py \
    --data_dir /lus/lfs1aip2/projects/s5e/lob_bench/bench_data/pious-snowball-264 \
    --model_name . \
    --stock GOOG \
    --time_period 2023 \
    --save_dir ./results \
    --divergence \
    --progress_interval 60 \
    --n_workers 64

echo "Benchmark completed at $(date)"
