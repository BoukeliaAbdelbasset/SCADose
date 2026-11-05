#!/usr/bin/env bash
# ==============================================================================
# SCADose: Environment Setup Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Setting up SCADose Python environment..."

if command -v conda &> /dev/null; then
    echo "Using conda to create/update environment 'scadose'..."
    conda env create -f "${REPO_ROOT}/environment.yml" || conda env update -f "${REPO_ROOT}/environment.yml"
    echo "To activate: conda activate scadose"
else
    echo "Conda not found. Installing via pip..."
    pip install -r "${REPO_ROOT}/requirements.txt"
fi

echo "Environment setup complete!"

