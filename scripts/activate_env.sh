#!/bin/bash

# Script to activate the conda environment for the project

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Activate the environment
echo "Activating conda environment: rocm"
source /home/gna/anaconda3/bin/activate rocm

if [ $? -eq 0 ]; then
    echo "Environment activated successfully"
    echo "You can now run make commands or python scripts"
else
    echo "Failed to activate environment"
    exit 1
fi