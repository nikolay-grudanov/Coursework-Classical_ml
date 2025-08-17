# Makefile for Classical ML Coursework

# Variables
PYTHON := python3
# CONDA_ENV can be a full prefix path or an environment name. If a .env file exists, it will override this value.
CONDA_ENV := /home/gna/anaconda3/envs/rocm
SRC_DIR := src
DATA_DIR := data
MODELS_DIR := models
REPORTS_DIR := reports
FIGURES_DIR := figures
LOGS_DIR := logs

# If .env exists, try to load CONDA_PREFIX or CONDA_ENV_NAME
ifneq (,$(wildcard .env))
include .env
export
# If CONDA_PREFIX is set in .env, use it as CONDA_ENV
ifneq (,$(CONDA_PREFIX))
CONDA_ENV := $(CONDA_PREFIX)
endif
ifneq (,$(CONDA_ENV_NAME))
CONDA_ENV := $(CONDA_ENV_NAME)
endif
endif

# Resolve conda invocation: prefer prefix (-p) when CONDA_PREFIX set,
# otherwise use environment name (-n) when CONDA_ENV_NAME set.
ifeq (,$(strip $(CONDA_PREFIX)))
	ifeq (,$(strip $(CONDA_ENV_NAME)))
		CONDA_RUN := conda run -n base
	else
		CONDA_RUN := conda run -n $(CONDA_ENV_NAME)
	endif
else
	CONDA_RUN := conda run -p $(CONDA_PREFIX)
endif

# Default target
.PHONY: all
all: clean check-config data features train evaluate report

# Clean previous outputs
.PHONY: clean
clean:
	@echo "Cleaning previous outputs..."
	@rm -rf $(MODELS_DIR)/*
	@rm -rf $(REPORTS_DIR)/*
	@rm -rf $(FIGURES_DIR)/*
	@rm -rf $(LOGS_DIR)/*
	@mkdir -p $(MODELS_DIR) $(REPORTS_DIR) $(FIGURES_DIR) $(LOGS_DIR)

# Data processing
.PHONY: data
data:
	@echo "Processing raw data..."
	@$(CONDA_RUN) $(PYTHON) $(SRC_DIR)/data/process_data.py


# Quick config sanity check: validate that paths referenced in configs/model_config.yaml exist.
.PHONY: check-config
check-config:
	@echo "Validating configuration paths in configs/model_config.yaml..."
	@$(CONDA_RUN) $(PYTHON) scripts/check_config.py

# Feature engineering
.PHONY: features
features:
	@echo "Engineering features..."
	@$(CONDA_RUN) $(PYTHON) $(SRC_DIR)/features/engineer_features.py

# Model training
.PHONY: train
train:
	@echo "Training models..."
	@$(CONDA_RUN) $(PYTHON) $(SRC_DIR)/main.py

# Model evaluation
.PHONY: evaluate
evaluate:
	@echo "Evaluating models..."
	@$(CONDA_RUN) $(PYTHON) $(SRC_DIR)/eval/evaluate_models.py

# Generate report
.PHONY: report
report:
	@echo "Generating reports..."
	@$(CONDA_RUN) $(PYTHON) $(SRC_DIR)/eval/generate_report.py

# Run exploratory data analysis
.PHONY: eda
eda:
	@echo "Running exploratory data analysis..."
	@$(CONDA_RUN) $(PYTHON) $(SRC_DIR)/data/eda.py

# Quick data integrity check
.PHONY: check-data
check-data:
	@echo "Running data integrity checks..."
	@$(CONDA_RUN) $(PYTHON) scripts/check_data_integrity.py

# Install dependencies
.PHONY: install
install:
	@echo "Installing dependencies..."
	@$(CONDA_RUN) pip install -r requirements.txt

# Run Jupyter notebooks
.PHONY: notebooks
notebooks:
	@echo "Starting Jupyter notebook..."
	@conda run -p $(CONDA_ENV) jupyter notebook notebooks/

# Additional targets for coursework
.PHONY: ic50_stability
ic50_stability:
	@echo "Running IC50 stabilization experiments (do not run by default)"
	@$(CONDA_RUN) $(PYTHON) $(SRC_DIR)/ic50_stabilize_T3.py

.PHONY: si8_oof
si8_oof:
	@echo "Generate OOF predictions and plots for SI>=8 classification"
	@$(CONDA_RUN) $(PYTHON) -m pip install --quiet PyYAML >/dev/null 2>&1 || true
	@$(CONDA_RUN) $(PYTHON) $(SRC_DIR)/eval/generate_baselines_t3.py

.PHONY: baselines_t3
baselines_t3:
	@echo "Generate baselines T3 report"
	@$(PYTHON) -m pip install --quiet PyYAML >/dev/null 2>&1 || true
	@$(PYTHON) $(SRC_DIR)/eval/generate_baselines_t3.py

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all       - Run the complete pipeline"
	@echo "  clean     - Clean previous outputs"
	@echo "  data      - Process raw data"
	@echo "  features  - Engineer features"
	@echo "  train     - Train models"
	@echo "  evaluate  - Evaluate models"
	@echo "  report    - Generate reports"
	@echo "  eda       - Run exploratory data analysis"
	@echo "  install   - Install dependencies"
	@echo "  notebooks - Start Jupyter notebook"
	@echo "  ic50_stability - Run IC50 stabilization script (explicit)"
	@echo "  si8_oof   - Generate OOF predictions/plots for SI>=8 classification"
	@echo "  baselines_t3 - Generate baselines T3 summary report"
	@echo "  help      - Show this help message"