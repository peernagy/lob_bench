#!/bin/bash
#SBATCH --job-name=lob_bench_plotting
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/lob_bench_plotting_%j.out
#SBATCH --error=logs/lob_bench_plotting_%j.err

set -euo pipefail

# Navigate to the working directory
cd /home/s5e/satyamaga.s5e/lob_bench

# Create logs / plots directories if they do not exist
mkdir -p logs
mkdir -p results/plots

# Load Miniforge/Conda
source ~/miniforge3/bin/activate

# Create or use existing conda environment
CONDA_ENV="lob_bench"
if conda info --envs | awk '{print $1}' | grep -q "^$CONDA_ENV$"; then
    echo "Using existing conda environment: $CONDA_ENV"
    conda activate "$CONDA_ENV"
else
    echo "Creating new conda environment: $CONDA_ENV"
    conda create -n "$CONDA_ENV" -y python=3.11
    conda activate "$CONDA_ENV"
fi

# Install python deps
echo "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements-fixed.txt

# Optional: enable Plotly PNG export by ensuring Chrome/Chromium is installed.
# By default we skip this on clusters where downloads or system packages are restricted.
if [[ "${PLOTLY_IMAGE_EXPORT:-}" == "1" ]]; then
    if command -v google-chrome >/dev/null 2>&1 || command -v chromium >/dev/null 2>&1 || command -v chromium-browser >/dev/null 2>&1; then
        echo "Found system Chrome/Chromium; Plotly PNG export should work."
    else
        echo "PLOTLY_IMAGE_EXPORT=1 set, but no Chrome/Chromium found. PNG export may fail."
    fi
else
    echo "Skipping Chrome setup; Plotly PNG export will fall back to HTML if unavailable."
fi

# Run the benchmark / plotting
echo "Starting plotting run at $(date)"
PYTHONUNBUFFERED=1 python run_plotting.py --histograms

echo "Plotting completed at $(date)"
