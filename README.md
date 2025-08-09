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
├── figures/          # Generated plots and figures
├── logs/             # Log files
├── models/            # Trained models
├── notebooks/          # Jupyter notebooks for exploration
├── reports/           # Generated reports
├── src/               # Source code
│   ├── data/          # Data processing scripts
│   ├── eval/          # Model evaluation scripts
│   ├── features/      # Feature engineering scripts
│   ├── models/        # Model training scripts
│   └── utils/         # Utility functions
└── Makefile           # Build automation
```

## Data

The dataset contains 1,000 chemical compounds with features representing their chemical properties and effectiveness metrics:

- **IC50**: Half maximal inhibitory concentration (lower is better)
- **CC50**: Half maximal cytotoxic concentration (higher is better)
- **SI**: Selectivity index (CC50/IC50, higher is better)

## Tasks

1. Regression for IC50
2. Regression for CC50
3. Regression for SI
4. Classification: IC50 > median
5. Classification: CC50 > median
6. Classification: SI > median
7. Classification: SI > 8

## Environment Setup

This project uses a conda environment located at `/home/gna/anaconda3/envs/rocm`.

### Using the rocm environment

All experiments and pipelines are run only in the rocm environment for reproducibility and compatibility of the GPU/ROCm stack.

The project is configured to automatically use the rocm environment through:

- **VS Code Settings**: See [.vscode/settings.json](.vscode/settings.json) for Python interpreter configuration
- **Makefile Targets**: All Makefile commands automatically use the rocm environment

### Installation/Startup

To set up and run the project:

1. **Activate the conda environment**:
   ```bash
conda activate /home/gna/anaconda3/envs/rocm
```

2. **Install dependencies** (if needed):
```bash
   make install
```

3. **Run the complete pipeline**:
```bash
   make all
```

4. **Or run individual pipeline steps**:
   ```bash
   # Process raw data
   make data

   # Engineer features
   make features

   # Train models
   make train

   # Evaluate models
   make evaluate

   # Generate reports
   make report

   # Run exploratory data analysis
   make eda
   ```

Alternatively, you can run scripts directly with the conda environment:
```bash
# Using conda run (recommended)
conda run -p /home/gna/anaconda3/envs/rocm python src/data/process_data.py

# Or activate the environment first
conda activate /home/gna/anaconda3/envs/rocm
python src/data/process_data.py
```

## Project Documentation

This project includes comprehensive documentation in the `reports/` directory:

- [Exploratory Data Analysis Report](reports/eda_report.md) - Detailed analysis of the dataset, feature distributions, and preprocessing steps
- [Final Report](reports/final_report.md) - Complete project findings, methodology, results, and conclusions
- [Project Standards](docs/standards.md) - Coding standards, project structure guidelines, and best practices

## Usage

The project generates several outputs:
- Processed data in `data/processed/`
- Trained models in `models/`
- Evaluation results in `reports/`
- Visualizations in `figures/`
- Logs in `logs/`

All processing should be done through the Makefile targets which ensure the proper environment is used.

