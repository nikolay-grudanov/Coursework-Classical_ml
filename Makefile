# Makefile for Classical ML Coursework

# Variables
PYTHON := python3
CONDA_ENV := /home/gna/anaconda3/envs/rocm
SRC_DIR := src
DATA_DIR := data
MODELS_DIR := models
REPORTS_DIR := reports
FIGURES_DIR := figures
LOGS_DIR := logs

# Default target
.PHONY: all
all: clean data features train evaluate report

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
	@conda run -p $(CONDA_ENV) $(PYTHON) $(SRC_DIR)/data/process_data.py

# Feature engineering
.PHONY: features
features:
	@echo "Engineering features..."
	@conda run -p $(CONDA_ENV) $(PYTHON) $(SRC_DIR)/features/engineer_features.py

# Model training
.PHONY: train
train:
	@echo "Training models..."
	@conda run -p $(CONDA_ENV) $(PYTHON) $(SRC_DIR)/models/train_models.py

# Model evaluation
.PHONY: evaluate
evaluate:
	@echo "Evaluating models..."
	@conda run -p $(CONDA_ENV) $(PYTHON) $(SRC_DIR)/eval/evaluate_models.py

# Generate report
.PHONY: report
report:
	@echo "Generating reports..."
	@conda run -p $(CONDA_ENV) $(PYTHON) $(SRC_DIR)/eval/generate_report.py

# Run exploratory data analysis
.PHONY: eda
eda:
	@echo "Running exploratory data analysis..."
	@conda run -p $(CONDA_ENV) $(PYTHON) $(SRC_DIR)/data/eda.py

# Install dependencies
.PHONY: install
install:
	@echo "Installing dependencies..."
	@conda run -p $(CONDA_ENV) pip install -r requirements.txt

# Run Jupyter notebooks
.PHONY: notebooks
notebooks:
	@echo "Starting Jupyter notebook..."
	@conda run -p $(CONDA_ENV) jupyter notebook notebooks/

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
	@echo "  help      - Show this help message"