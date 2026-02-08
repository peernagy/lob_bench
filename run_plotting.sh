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

# Install python deps (only if requirements exist)
echo "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements-fixed.txt

# ---- Ensure Chrome/Chromium is available for Kaleido (non-interactive) ----

# 1) If a system chrome/chromium is already on PATH, use it.
if command -v google-chrome >/dev/null 2>&1 || command -v chromium >/dev/null 2>&1 || command -v chromium-browser >/dev/null 2>&1; then
    echo "Found system Chrome/Chromium; Kaleido should detect it automatically."
else
    # 2) Prefer installing chromium from conda-forge into current env (non-interactive).
    echo "No Chrome detected. Attempting non-interactive conda install of chromium (conda-forge)..."
    # Use -y to avoid prompts. If conda fails (no network / no permission) we fall back.
    if conda install -y -c conda-forge chromium; then
        echo "Conda-installed chromium into environment."
    else
        echo "Conda install failed or is not permitted. Attempting Kaleido's installer non-interactively..."
        # 3) Fallback: run plotly_get_chrome non-interactively by piping 'y'
        #    This will download Chrome to the local environment directory if network permits.
        #    If your cluster forbids outbound downloads, this may fail.
        if command -v plotly_get_chrome >/dev/null 2>&1; then
            printf 'y\n' | plotly_get_chrome || {
                echo "plotly_get_chrome failed. If downloads are blocked, please pre-install Chrome/Chromium or ask sysadmin."
                exit 1
            }
        else
            # try kaleido_get_chrome name as well
            if command -v kaleido_get_chrome >/dev/null 2>&1; then
                printf 'y\n' | kaleido_get_chrome || {
                    echo "kaleido_get_chrome failed. If downloads are blocked, please pre-install Chrome/Chromium or ask sysadmin."
                    exit 1
                }
            else
                echo "No plotly_get_chrome/kaleido_get_chrome command found. Please ensure kaleido/plotly is installed."
                exit 1
            fi
        fi
    fi
fi

# Optional: If you know where the binary ended up (example for conda), point Kaleido at it explicitly:
# export BROWSER_PATH="$CONDA_PREFIX/bin/chromium"
# export PRE_LOAD_CHROME=True

# Run the benchmark / plotting
echo "Starting plotting run at $(date)"
PYTHONUNBUFFERED=1 python run_plotting.py --histograms

echo "Plotting completed at $(date)"
