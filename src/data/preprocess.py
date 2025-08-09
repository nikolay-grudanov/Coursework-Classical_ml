"""
Module for data preprocessing including handling missing values, 
outlier detection, and feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by removing unnecessary columns and handling missing values.
    
    Args:
        data (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    logger.info("Starting data cleaning process")
    
    # Remove unnecessary columns
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
        logger.info("Dropped 'Unnamed: 0' column")
    
    # Handle missing values by filling with median
    missing_before = data.isnull().sum().sum()
    data = data.fillna(data.median())
    missing_after = data.isnull().sum().sum()
    
    logger.info(f"Filled {missing_before - missing_after} missing values with median")
    
    return data

def detect_outliers_iqr(data: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        data (pd.DataFrame): Data to check for outliers
        columns (list): Columns to check for outliers. If None, checks all numeric columns.
        
    Returns:
        pd.DataFrame: Boolean DataFrame indicating outliers
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = pd.DataFrame()
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = (data[col] < lower_bound) | (data[col] > upper_bound)
    
    return outliers

def remove_outliers_iqr(data: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        data (pd.DataFrame): Data to clean
        columns (list): Columns to check for outliers. If None, checks all numeric columns.
        
    Returns:
        pd.DataFrame: Data with outliers removed
    """
    outliers = detect_outliers_iqr(data, columns)
    outlier_rows = outliers.any(axis=1)
    cleaned_data = data[~outlier_rows]
    
    logger.info(f"Removed {outlier_rows.sum()} rows with outliers")
    
    return cleaned_data

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from existing data.
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with additional features
    """
    logger.info("Creating additional features")
    
    # Ensure SI is calculated correctly
    if 'SI' not in data.columns:
        data['SI'] = data['CC50, mM'] / data['IC50, mM']
        logger.info("Calculated SI as CC50/IC50")
    
    # Create binary targets for classification
    data['IC50_above_median'] = (data['IC50, mM'] > data['IC50, mM'].median()).astype(int)
    data['CC50_above_median'] = (data['CC50, mM'] > data['CC50, mM'].median()).astype(int)
    data['SI_above_median'] = (data['SI'] > data['SI'].median()).astype(int)
    data['SI_above_8'] = (data['SI'] > 8).astype(int)
    
    logger.info("Created binary classification targets")
    
    return data

def split_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets for all target variables.
    
    Args:
        data (pd.DataFrame): Input data
        test_size (float): Proportion of data for test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: Split datasets (train_features, test_features, and target sets)
    """
    # Define features (exclude target variables)
    feature_columns = [col for col in data.columns if col not in [
        'IC50, mM', 'CC50, mM', 'SI', 
        'IC50_above_median', 'CC50_above_median', 
        'SI_above_median', 'SI_above_8'
    ]]
    
    features = data[feature_columns]
    
    # Define targets
    target_ic50 = data['IC50, mM']
    target_cc50 = data['CC50, mM']
    target_si = data['SI']
    
    target_ic50_class = data['IC50_above_median']
    target_cc50_class = data['CC50_above_median']
    target_si_class = data['SI_above_median']
    target_si_8_class = data['SI_above_8']
    
    # Split for regression
    train_features, test_features, train_ic50, test_ic50 = train_test_split(
        features, target_ic50, test_size=test_size, random_state=random_state
    )
    
    # Use same split for other targets to maintain consistency
    _, _, train_cc50, test_cc50 = train_test_split(
        features, target_cc50, test_size=test_size, random_state=random_state
    )
    _, _, train_si, test_si = train_test_split(
        features, target_si, test_size=test_size, random_state=random_state
    )
    
    # Split for classification
    _, _, train_ic50_class, test_ic50_class = train_test_split(
        features, target_ic50_class, test_size=test_size, random_state=random_state, stratify=target_ic50_class
    )
    _, _, train_cc50_class, test_cc50_class = train_test_split(
        features, target_cc50_class, test_size=test_size, random_state=random_state, stratify=target_cc50_class
    )
    _, _, train_si_class, test_si_class = train_test_split(
        features, target_si_class, test_size=test_size, random_state=random_state, stratify=target_si_class
    )
    _, _, train_si_8_class, test_si_8_class = train_test_split(
        features, target_si_8_class, test_size=test_size, random_state=random_state, stratify=target_si_8_class
    )
    
    logger.info(f"Split data into train ({len(train_features)} samples) and test ({len(test_features)} samples) sets")
    
    return (
        train_features, test_features,
        train_ic50, test_ic50,
        train_cc50, test_cc50,
        train_si, test_si,
        train_ic50_class, test_ic50_class,
        train_cc50_class, test_cc50_class,
        train_si_class, test_si_class,
        train_si_8_class, test_si_8_class
    )