#!/usr/bin/env bash
set -euo pipefail

# Minimal bootstrap: create or update the Conda environment from environment.yml
ENV_NAME="policy-analytics"

# Detect conda
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Please install Miniconda or Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

# Load conda in this shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create or update env
if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "[INFO] Creating environment ${ENV_NAME} from environment.yml..."
  conda env create -f environment.yml
else
  echo "[INFO] Updating environment ${ENV_NAME} from environment.yml..."
  conda env update -f environment.yml --prune
fi

# Optional: show how to activate after creation
echo
echo "[OK] Environment is ready."
echo "To activate:  conda activate ${ENV_NAME}"
echo "Optional Jupyter kernel:  python -m ipykernel install --user --name ${ENV_NAME} --display-name 'policy-analytics (py312)'"
