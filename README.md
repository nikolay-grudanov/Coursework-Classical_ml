# Classical ML Coursework: Drug Effectiveness Prediction

This project predicts the effectiveness of chemical compounds against the influenza virus using machine learning techniques.

## Project Structure

```
.
├── analysis/          # Exploratory data analysis results
├── configs/           # Configuration files
├── data/              # Data storage
│   ├── processed/     # Processed data ready for modeling
│   └── raw/           # Raw data files
├── experiments/       # Experiment tracking logs
├── figures/           # Generated plots and figures
├── logs/              # Log files
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks for exploration
├── reports/           # Generated reports
├── src/               # Source code
│   ├── data/          # Data processing scripts
│   ├── eval/          # Model evaluation scripts
│   ├── features/      # Feature engineering scripts
│   ├── models/        # Model training scripts
│   └── utils/         # Utility functions
├── scripts/           # Helper scripts (activation, env loading, etc.)
├── .github/           # GitHub Actions workflows
├── .pre-commit-config.yaml  # Pre-commit configuration
├── .env.example       # Example environment overrides
├── Makefile           # Build automation (reads .env if present)
├── requirements.lock.txt    # Environment lock file (read-only)
└── requirements.txt   # Base dependencies
```

## Data

The dataset contains 1,000 chemical compounds with features representing their chemical properties and effectiveness metrics:

- IC50: Half maximal inhibitory concentration (lower is better)
- CC50: Half maximal cytotoxic concentration (higher is better)
- SI: Selectivity index (CC50/IC50, higher is better)

## Tasks

1. Regression for IC50
2. Regression for CC50
3. Regression for SI
4. Classification: IC50 > median
5. Classification: CC50 > median
6. Classification: SI > median
7. Classification: SI > 8

## Environment Setup

This project uses a conda environment. Instead of hard-coding a path, configure your local conda environment via a .env file in the repository root.

Copy the example .env:
cp .env.example .env

Supported .env variables (set one or more as appropriate):

- CONDA_PREFIX — full path to your conda environment, e.g. /home/user/miniconda3/envs/ml
- CONDA_ENV_NAME — environment name (alternative, e.g. ml)
- CONDA_BIN — path to conda binary directory (optional fallback), e.g. /home/user/miniconda3/bin
- DATA_PATH — override default data directory (optional), e.g. /mnt/data/project/data

Example .env (see .env.example):
CONDA_PREFIX=/home/user/miniconda3/envs/ml
CONDA_ENV_NAME=ml
CONDA_BIN=/home/user/miniconda3/bin
DATA_PATH=/path/to/data

Activation and usage:

- Source the helper script which reads .env and prepares the environment for Makefile targets and scripts:
source scripts/activate_env.sh

- Or activate the environment manually using the variables from .env:
conda activate /path/from/.env/CONDA_PREFIX
# or
conda activate CONDA_ENV_NAME_from_.env

- You can also run scripts with conda run using the CONDA_PREFIX value:
conda run -p $CONDA_PREFIX python src/data/process_data.py

Makefile targets and project scripts will read .env if present. For data path overrides, set DATA_PATH in .env or export DATA_PATH before running scripts.
### Installation/Startup

1. Ensure .env is configured (copy .env.example and edit as needed)
2. **Activate the conda environment**:
   ```bash
   # Preferred: use helper script that reads .env
source scripts/activate_env.sh
   # Or activate manually using full prefix stored in .env (e.g. CONDA_PREFIX=/home/mrr/miniconda3/envs/ml)
   conda activate /home/mrr/miniconda3/envs/ml
   ```
3. Install dependencies (if needed):
   make install
4. Run the complete pipeline:
   make all
5. Or run individual pipeline steps:
make data        # Process raw data
make features    # Engineer features
make train       # Train models
make evaluate    # Evaluate models
make report      # Generate reports
make eda         # Run exploratory data analysis

Alternatively, run scripts directly with the configured conda environment:
conda run -p $CONDA_PREFIX python src/data/process_data.py
# Or activate the environment first:
conda activate $CONDA_PREFIX
python src/data/process_data.py
## Environment Lock File

A complete list of dependencies used in this project is available in requirements.lock.txt.

Note: This file is for reference purposes only and should not be used for installation. The frozen environment may contain platform-specific packages that might not work on other systems. Use requirements.txt for base dependencies instead.

## Pre-commit Hooks

This project uses pre-commit hooks for code quality and consistency:
- ruff: Linting and code formatting
- black: Code formatting
- isort: Import sorting
- nbstripout: Jupyter notebook output removal

To install and enable pre-commit hooks:
pre-commit install
## Experiment Tracking

Experiments are tracked in experiments/registry.jsonl with the append_jsonl utility.

## Project Documentation

This project includes documentation in the reports/ directory:
- reports/eda_report.md - Detailed analysis of the dataset, feature distributions, and preprocessing steps
- reports/final_report.md - Complete project findings, methodology, results, and conclusions
- docs/standards.md - Coding standards, project structure guidelines, and best practices

## Usage and Outputs
The project generates several outputs:
- Processed data in data/processed/
- Trained models in models/
- Evaluation results in reports/
- Visualizations in figures/
- Logs in logs/

All processing should be done through the Makefile targets which ensure the proper environment is used.

