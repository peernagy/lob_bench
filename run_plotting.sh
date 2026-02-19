#!/bin/bash
#SBATCH --job-name=lob_bench_plotting
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/lob_bench_plotting_%j.out
#SBATCH --error=logs/lob_bench_plotting_%j.err

set -euo pipefail

# 1. Setup environment
source ~/miniforge3/bin/activate
CONDA_ENV="lob_bench"

if ! conda info --envs | grep -q "^$CONDA_ENV "; then
    conda create -n "$CONDA_ENV" -y python=3.11 > /dev/null
fi
conda activate "$CONDA_ENV"

# 2. Install dependencies (Quietly)
echo "Installing dependencies..."
pip install -r requirements-fixed.txt -qq

# 3. The "HPC Fix" for Segmentation Faults
# This forces Kaleido to run in a pure headless state
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
export KALEIDO_DISABLE_SANDBOX=1
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

# 4. Run
echo "Starting plot run..."
python run_plotting.py --histograms