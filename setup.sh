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

# Clone and setup ProSST
PROSST_DIR="$PROJECT_ROOT/ProSST"
if [ ! -d "$PROSST_DIR" ]; then
    log "Cloning ProSST repository..."
    git clone https://github.com/ai4protein/ProSST.git "$PROSST_DIR"
fi

EVOEF2_DIR="$PROJECT_ROOT/EvoEF2"
if [ ! -d "$EVOEF2_DIR" ]; then
    log "Cloning EvoEF2 repository..."
    git clone https://github.com/tommyhuangthu/EvoEF2 "$EVOEF2_DIR"
fi

log "Building EvoEF2..."
cd "$EVOEF2_DIR"
if ./build.sh; then
    log "EvoEF2 built successfully with build.sh"
elif g++ -O3 --fast-math -o EvoEF2 src/*.cpp; then
    log "EvoEF2 built successfully with g++ --fast-math"
elif g++ -O3 -o EvoEF2 src/*.cpp; then
    log "EvoEF2 built successfully with g++ (without --fast-math)"
else
    log "Error: Failed to build EvoEF2. Please check the build instructions for your system."
    cd "$PROJECT_ROOT" 
    exit 1
fi
cd "$PROJECT_ROOT"

# Install DE-STRESS
log "Setting up DE-STRESS..."
DESTRESS_DIR="$PROJECT_ROOT/de-stress"

# Clone DE-STRESS if not present
if [ ! -d "$DESTRESS_DIR" ]; then
    log "Cloning DE-STRESS repository..."
    git clone https://github.com/wells-wood-research/de-stress.git "$DESTRESS_DIR" || {
        log "Error: Failed to clone DE-STRESS repository"
        exit 1
    }
fi

# Copy environment file if not present
if [ ! -f "$DESTRESS_DIR/.env-headless" ]; then
    log "Setting up DE-STRESS environment file..."
    if [ -f "$DESTRESS_DIR/.env-headless-testing" ]; then
        cp "$DESTRESS_DIR/.env-headless-testing" "$DESTRESS_DIR/.env-headless" || {
            log "Error: Failed to copy .env-headless-testing"
            exit 1
        }
    else
        log "Error: .env-headless-testing not found in DE-STRESS directory"
        exit 1
    fi
fi

# Run DE-STRESS setup script
if [ -f "$DESTRESS_DIR/setup.sh" ]; then
    log "Running DE-STRESS setup script..."
    cd "$DESTRESS_DIR"
    # Run setup.sh in headless mode (it will handle requirements installation)
    ./setup.sh || {
        log "Error: DE-STRESS setup.sh failed"
        cd "$PROJECT_ROOT"
        exit 1
    }
    cd "$PROJECT_ROOT"
    log "DE-STRESS setup completed successfully"
else
    log "Error: setup.sh not found in DE-STRESS directory"
    exit 1
fi

log "Setup completed successfully!"
log "To activate the environment, run: conda activate $ENV_NAME"
