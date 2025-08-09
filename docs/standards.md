# Data Science Project Standards

This document outlines the established standards for the MEPHI Coursework-Classical ML project based on industry best practices and project requirements.

## Project Structure
- `/data`: Contains raw and processed datasets
- `/src`: Source code for data processing, modeling, and analysis
- `/notebooks`: Jupyter notebooks for exploration and experimentation
- `/reports`: Generated reports including EDA, model evaluations
- `/models`: Saved trained models and preprocessing objects
- `/configs`: Configuration files for preprocessing, modeling, and validation
- `/figures`: Generated visualizations and figures
- `/analysis`: Detailed analysis outputs (feature importance, residuals, etc.)
- `/logs`: Log files from experiments and training runs

## Code Standards
- All Python code must follow PEP8 style guidelines
- Use type hints for function parameters and return values
- Modularize code with clear separation of concerns (data loading, preprocessing, modeling, evaluation)
- Document all functions with docstrings explaining purpose, parameters, and return values
- Use meaningful variable and function names that clearly describe their purpose

## Reproducibility Requirements
- Fix all random seeds for reproducibility (numpy, sklearn, pandas)
- Use virtual environment with fixed dependency versions (requirements.txt or pyproject.toml)
- Configuration files should control all hyperparameters and data paths
- All experiment results should be logged with parameters and metrics

## Git Workflow
- Use feature branches for development work
- Commit messages should be clear and descriptive
- Small, focused commits that accomplish a single task
- Branch naming convention: feature/task-name, fix/issue-description

## Data Handling
- Never commit raw data files to the repository
- Document data sources and transformations
- Preserve raw data and create processed versions
- Identify and handle missing values appropriately
- Check for data leakage between features and targets

## Modeling Standards
- Split data before any preprocessing to prevent leakage
- Use cross-validation for model evaluation
- For classification tasks with imbalance, apply techniques like SMOTE
- Validate splits are stratified for classification tasks
- Evaluate models with appropriate metrics:
  * Regression: RMSE, MAE, R2
  * Classification: ROC-AUC, PR-AUC, F1 (macro), accuracy

## Experiment Tracking
- Log all experiments with parameters and results
- Save model configurations and trained models
- Document findings and decisions in reports