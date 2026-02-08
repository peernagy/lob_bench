#!/bin/bash
#SBATCH --job-name=lob_bench
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/lob_bench_%j.out
#SBATCH --error=logs/lob_bench_%j.err

# Navigate to the working directory
cd /home/s5e/satyamaga.s5e/lob_bench

# Create logs directory if it doesn't exist
mkdir -p logs

# Load Miniforge/Conda
source ~/miniforge3/bin/activate

# Create or use existing conda environment
CONDA_ENV="lob_bench"
if conda info --envs | grep -q "^$CONDA_ENV "; then
    echo "Using existing conda environment: $CONDA_ENV"
    conda activate $CONDA_ENV
else
    echo "Creating new conda environment: $CONDA_ENV"
    conda create -n $CONDA_ENV -y python=3.11
    conda activate $CONDA_ENV
fi

# Install dependencies from requirements-fixed.txt
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements-fixed.txt

# Run the benchmark
echo "Starting benchmark run at $(date)"
PYTHONUNBUFFERED=1 python run_bench.py \
    --data_dir /lus/lfs1aip2/projects/s5e/public/quant_team/LOBS5/inference/logical-serenity-19_2168595 \
    --model_name . \
    --stock . \
    --time_period . \
    --save_dir ./results \
    --all \
    --progress_interval 60

echo "Benchmark completed at $(date)"
