"""
Main pipeline for drug effectiveness prediction coursework.
Orchestrates data loading, preprocessing, model training, and evaluation.
"""
import pandas as pd
import numpy as np
import logging
import yaml
import sys
from pathlib import Path

# Add src to path for module imports
sys.path.append(str(Path(__file__).parent))

from data.load_data import load_data, save_data
from data.preprocess import clean_data, create_features, split_data
from models.train_models import train_regression_models, train_classification_models, save_model_results
from eval.evaluate_models import generate_regression_comparison_report, generate_classification_comparison_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path="configs/model_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main pipeline function."""
    logger.info("Starting drug effectiveness prediction pipeline")
    
    # Load configuration
    config = load_config()
    logger.info("Loaded configuration")
    
    # Load data
    data = load_data(config['data_path'])
    
    # Clean data
    data = clean_data(data)
    
    # Create features
    data = create_features(data)
    
    # Save processed data
    save_data(data, config['processed_data_path'])
    
    # Split data
    (
        train_features, test_features,
        train_ic50, test_ic50,
        train_cc50, test_cc50,
        train_si, test_si,
        train_ic50_class, test_ic50_class,
        train_cc50_class, test_cc50_class,
        train_si_class, test_si_class,
        train_si_8_class, test_si_8_class
    ) = split_data(data, test_size=config['test_size'], random_state=config['random_state'])
    
    logger.info("Completed data preprocessing")
    
    # Train regression models
    results = {}
    
    logger.info("Training regression models for IC50")
    results['IC50_Regression'] = train_regression_models(
        train_features, train_ic50, test_features, test_ic50, "IC50"
    )
    
    logger.info("Training regression models for CC50")
    results['CC50_Regression'] = train_regression_models(
        train_features, train_cc50, test_features, test_cc50, "CC50"
    )
    
    logger.info("Training regression models for SI")
    results['SI_Regression'] = train_regression_models(
        train_features, train_si, test_features, test_si, "SI"
    )
    
    # Train classification models
    logger.info("Training classification models for IC50 > median")
    results['IC50_Classification'] = train_classification_models(
        train_features, train_ic50_class, test_features, test_ic50_class, "IC50 > median"
    )
    
    logger.info("Training classification models for CC50 > median")
    results['CC50_Classification'] = train_classification_models(
        train_features, train_cc50_class, test_features, test_cc50_class, "CC50 > median"
    )
    
    logger.info("Training classification models for SI > median")
    results['SI_Classification'] = train_classification_models(
        train_features, train_si_class, test_features, test_si_class, "SI > median"
    )
    
    logger.info("Training classification models for SI > 8")
    results['SI_8_Classification'] = train_classification_models(
        train_features, train_si_8_class, test_features, test_si_8_class, "SI > 8"
    )
    
    # Save model results
    save_model_results(results, "models/model_results.json")
    
    # Generate comparison reports
    generate_regression_comparison_report(results, "reports/regression_comparison.md")
    generate_classification_comparison_report(results, "reports/classification_comparison.md")
    
    logger.info("Pipeline completed successfully")
    
    # Print summary
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Processed {len(data)} samples")
    print(f"Used {len(train_features.columns)} features")
    print(f"Results saved to models/model_results.json")
    print(f"Reports saved to reports/")
    print("="*50)

if __name__ == "__main__":
    main()