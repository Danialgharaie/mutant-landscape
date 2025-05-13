#!/bin/bash

#!/usr/bin/env bash

# Setup script for mutant-landscape project
# This script sets up a Conda environment and installs all required dependencies

# Exit on error, unset variable, or pipe failure
set -euo pipefail

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to handle errors
error_handler() {
    log "Error occurred in setup script on line $1"
    exit 1
}

# Set error handler
trap 'error_handler ${LINENO}' ERR

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    log "Error: Conda is not installed. Please install Conda first."
    exit 1
fi

# Check if git is installed
if ! command -v git &> /dev/null; then
    log "Error: Git is not installed. Please install Git first."
    exit 1
fi

# Set environment variables
ENV_NAME="mutant-landscape"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine requirements file based on CUDA availability
if command -v nvidia-smi &> /dev/null; then
    log "CUDA detected - using GPU requirements"
    REQ_FILE="requirements_gpu.txt"
else
    log "No CUDA detected - using CPU requirements"
    REQ_FILE="requirements_cpu.txt"
fi

# Create and activate conda environment
if ! conda env list | grep -q "^$ENV_NAME "; then
    log "Creating conda environment: $ENV_NAME..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# Ensure proper conda activation
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install project requirements
log "Installing requirements from $REQ_FILE..."
pip install -r "$PROJECT_ROOT/$REQ_FILE"

# Initialize PYTHONPATH if not set
PYTHONPATH=${PYTHONPATH:-}

# Clone and setup ProSST
PROSST_DIR="$PROJECT_ROOT/ProSST"
if [ ! -d "$PROSST_DIR" ]; then
    log "Cloning ProSST repository..."
    git clone https://github.com/ai4protein/ProSST.git "$PROSST_DIR"
    
    # Add ProSST to PYTHONPATH in conda environment
    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
    echo "export PYTHONPATH=\"$PROSST_DIR\${PYTHONPATH:+:\$PYTHONPATH}\"" > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
    chmod +x "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
    
    # Update current session's PYTHONPATH
    export PYTHONPATH="$PROSST_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

log "Setup completed successfully!"
log "To activate the environment, run: conda activate $ENV_NAME"
