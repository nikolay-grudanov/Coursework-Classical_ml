#!/bin/bash

# Script to activate the conda environment for the project

# Load .env if present (contains CONDA_PREFIX and CONDA_BIN)
if [ -f .env ]; then
    # shellcheck disable=SC1091
    source .env
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Warning: 'conda' not in PATH"
    # Try to use CONDA_BIN from .env
    if [ -n "$CONDA_BIN" ] && [ -x "$CONDA_BIN/activate" ]; then
        echo "Sourcing activate script from $CONDA_BIN"
        # shellcheck disable=SC1090
        source "$CONDA_BIN/activate"
    else
        echo "Error: conda is not installed or not in PATH and CONDA_BIN not set"
        echo "Please install Anaconda or Miniconda first or set CONDA_BIN in .env"
        exit 1
    fi
fi

# Determine which environment prefix/name to activate
# Priority: CONDA_PREFIX (full path) -> CONDA_ENV_NAME (name) -> fallback to OLD_CONDA_PREFIX if present
ACTIVATE_TARGET=""
if [ -n "$CONDA_PREFIX" ]; then
    ACTIVATE_TARGET="$CONDA_PREFIX"
elif [ -n "$CONDA_ENV_NAME" ]; then
    ACTIVATE_TARGET="$CONDA_ENV_NAME"
elif [ -n "$OLD_CONDA_PREFIX" ]; then
    ACTIVATE_TARGET="$OLD_CONDA_PREFIX"
fi

if [ -z "$ACTIVATE_TARGET" ]; then
    echo "Error: no conda environment specified. Please set CONDA_PREFIX or CONDA_ENV_NAME in .env"
    exit 1
fi

echo "Activating conda environment: $ACTIVATE_TARGET"

# Prefer 'conda activate' if available
if command -v conda &> /dev/null; then
    # Try activating by prefix or name
    conda activate "$ACTIVATE_TARGET"
else
    # Fallback: if CONDA_BIN provided, use its activate script
    if [ -n "$CONDA_BIN" ] && [ -x "$CONDA_BIN/activate" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_BIN/activate" "$ACTIVATE_TARGET"
    else
        echo "Error: cannot activate environment (no conda and no CONDA_BIN/activate)"
        exit 1
    fi
fi

if [ $? -eq 0 ]; then
    echo "Environment activated successfully"
    echo "You can now run make commands or python scripts"
else
    echo "Failed to activate environment"
    exit 1
fi