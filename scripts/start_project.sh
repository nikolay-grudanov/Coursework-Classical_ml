#!/bin/bash

# Script to initialize the project with the correct conda environment

# Load .env if present
if [ -f .env ]; then
    # shellcheck disable=SC1091
    source .env
fi

PROJECT_DIR=$(pwd)
# Determine env path from .env variables
if [ -n "$CONDA_PREFIX" ]; then
    ENV_PATH="$CONDA_PREFIX"
elif [ -n "$CONDA_ENV_NAME" ]; then
    ENV_PATH="$CONDA_ENV_NAME"
elif [ -n "$OLD_CONDA_PREFIX" ]; then
    ENV_PATH="$OLD_CONDA_PREFIX"
else
    ENV_PATH=""
fi

if [ -z "$ENV_PATH" ]; then
    echo "Warning: No CONDA_PREFIX or CONDA_ENV_NAME set. Please copy .env.example to .env and set values."
else
    echo "Initializing project in $PROJECT_DIR"
    echo "Using conda environment: $ENV_PATH"
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Warning: conda is not in PATH"
    if [ -n "$CONDA_BIN" ] && [ -x "$CONDA_BIN/activate" ]; then
        echo "Sourcing activate script from $CONDA_BIN"
        # shellcheck disable=SC1090
        source "$CONDA_BIN/activate"
    else
        echo "Note: conda not found. You can still create the environment manually using the provided environment files."
    fi
fi

# If ENV_PATH points to an existing directory, warn if missing
if [ -n "$CONDA_PREFIX" ] && [ ! -d "$CONDA_PREFIX" ]; then
    echo "Warning: CONDA_PREFIX directory does not exist: $CONDA_PREFIX"
fi

if [ -n "$CONDA_ENV_NAME" ]; then
    # We cannot reliably check environment by name; suggest user to create it if missing
    echo "If environment named '$CONDA_ENV_NAME' does not exist, create it with 'conda env create -f environment.yml -n $CONDA_ENV_NAME' or update .env"
fi

echo "You can now run make commands or activate the environment manually:"
if [ -n "$CONDA_PREFIX" ]; then
    echo "  conda activate $CONDA_PREFIX"
elif [ -n "$CONDA_ENV_NAME" ]; then
    echo "  conda activate $CONDA_ENV_NAME"
else
    echo "  # set CONDA_PREFIX or CONDA_ENV_NAME in .env and run this script again"
fi

echo "To run the complete pipeline, run:"
echo "  make all"
echo "To run individual steps, use:"
echo "  make data"
echo "  make features"
echo "  make train"
echo "  make evaluate"
echo "  make report"
echo "For VS Code users, set interpreter to:"
if [ -n "$CONDA_PREFIX" ]; then
    echo "  $CONDA_PREFIX/bin/python"
fi
