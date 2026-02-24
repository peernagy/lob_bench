#!/bin/bash
#SBATCH --job-name=lob_bench_plotting
#SBATCH --time=00:25:00
#SBATCH --nodes=1
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
pip install --upgrade "numpy==2.2.4" "matplotlib>=3.10.0" "scikit_learn>=1.5.2" -qq
# Force clean reinstall to avoid mixed statsmodels files that can cause
# ImportError: cannot import name 'SP_LT_116' from statsmodels.compat.scipy
pip install --upgrade --force-reinstall --no-cache-dir "scipy==1.15.2" "statsmodels==0.14.6" -qq

# 3. The "HPC Fix" for Segmentation Faults
# This forces Kaleido to run in a pure headless state
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
export KALEIDO_DISABLE_SANDBOX=1
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

# 4. Run
echo "Starting plot run..."
python -u run_plotting.py --score_dir /projects/s5e/lob_pipeline/Symlinked-Scores --summary_only