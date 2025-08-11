#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Navigate to the project directory
PROJECT_DIR="/Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas"
cd "$PROJECT_DIR" || { echo "Failed to navigate to project directory: $PROJECT_DIR"; exit 1; }

# Activate the conda environment
CONDA_ENV="pynas"
if conda activate "$CONDA_ENV"; then
    echo "Activated conda environment: $CONDA_ENV"
else
    echo "Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

# Install the package in editable mode
if pip install -e .; then
    echo "Package installed successfully in editable mode."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Failed to install the package."
    exit 1
fi torchmetrics pytorch-lightning 