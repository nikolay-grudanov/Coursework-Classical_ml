"""Module for data loading and documentation of data sources.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from Excel file and document source information.
    
    Args:
        filepath (str): Path to the Excel file
        
    Returns:
        pd.DataFrame: Loaded data
        
    Logs:
        - Size of the dataset
        - Column names and types
        - Missing values information
    """
    logger.info(f"Loading data from {filepath}")
    
    # Load data
    data = pd.read_excel(filepath)
    
    # Log dataset information
    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Columns: {list(data.columns)}")
    
    # Log column types
    for col in data.columns:
        logger.info(f"Column '{col}' type: {data[col].dtype}")
    
    # Log missing values
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        logger.warning(f"Missing values detected:")
        for col, count in missing_data[missing_data > 0].items():
            logger.warning(f"  {col}: {count} missing values")
    else:
        logger.info("No missing values detected")
    
    return data

def save_data(data: pd.DataFrame, output_filepath: str) -> None:
    """
    Save processed data to CSV format.
    
    Args:
        data (pd.DataFrame): Data to save
        output_filepath (str): Path to save the data
    """
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_filepath, index=False)
    logger.info(f"Data saved to {output_filepath}")