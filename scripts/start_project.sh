#!/bin/bash

# Script to initialize the project with the correct conda environment

PROJECT_DIR=$(pwd)
ENV_PATH="/home/gna/anaconda3/envs/rocm"

echo "Initializing project in $PROJECT_DIR"
echo "Using conda environment: $ENV_PATH"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if the environment exists
if [ ! -d "$ENV_PATH" ]; then
    echo "Error: conda environment not found at $ENV_PATH"
    echo "Please create the environment first"
    exit 1
fi

echo "Environment found. You can now run make commands or activate the environment manually:"
echo ""
echo "To activate the environment manually, run:"
echo "  conda activate $ENV_PATH"
echo ""
echo "To run the complete pipeline, run:"
echo "  make all"
echo ""
echo "To run individual steps, use:"
echo "  make data"
echo "  make features"
echo "  make train"
echo "  make evaluate"
echo "  make report"
echo ""
echo "For VS Code users, the interpreter should be automatically set to:"
echo "  $ENV_PATH/bin/python"