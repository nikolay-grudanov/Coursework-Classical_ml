"""
Module for preprocessing pipelines for regression and classification tasks.
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import logging

logger = logging.getLogger(__name__)

def create_regression_pipeline():
    """
    Create a preprocessing pipeline for regression tasks.
    
    Pipeline:
    1. SimpleImputer with median strategy for handling missing values
    2. StandardScaler for feature scaling
    
    Returns:
        sklearn.pipeline.Pipeline: Configured preprocessing pipeline
    """
    logger.info("Creating regression preprocessing pipeline")
    
    # Create pipeline: Imputation -> Scaling
    regression_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    logger.info("Regression preprocessing pipeline created")
    return regression_pipeline

def create_classification_pipeline(with_smote=False, random_state=42):
    """
    Create a preprocessing pipeline for classification tasks.
    
    For use inside cross-validation folds only.
    
    Pipeline:
    1. SMOTE for handling class imbalance (optional, only inside train folds)
    2. StandardScaler for feature scaling
    
    Args:
        with_smote (bool): Whether to include SMOTE in the pipeline
        random_state (int): Random state for reproducibility
        
    Returns:
        sklearn.pipeline.Pipeline or imblearn.pipeline.Pipeline: Configured preprocessing pipeline
    """
    logger.info(f"Creating classification preprocessing pipeline with SMOTE: {with_smote}")
    
    if with_smote:
        # Create pipeline with SMOTE: SMOTE -> Scaling
        classification_pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('scaler', StandardScaler())
        ])
    else:
        # Create pipeline without SMOTE: Scaling only
        classification_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
    
    logger.info("Classification preprocessing pipeline created")
    return classification_pipeline

def apply_preprocessing_pipeline(pipeline, X_train, y_train, X_test=None, y_test=None):
    """
    Apply preprocessing pipeline to training data and optionally to test data.
    
    Args:
        pipeline: Preprocessing pipeline (sklearn or imblearn)
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame, optional): Test features
        y_test (pd.Series, optional): Test target
        
    Returns:
        tuple: Preprocessed data (X_train_processed, y_train_processed, X_test_processed (optional))
    """
    logger.info("Applying preprocessing pipeline")
    
    # Fit and transform training data
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    y_train_processed = y_train
    
    logger.info(f"Training data preprocessed. Shape: {X_train_processed.shape}")
    
    if X_test is not None:
        # Transform test data using fitted pipeline
        X_test_processed = pipeline.transform(X_test)
        logger.info(f"Test data preprocessed. Shape: {X_test_processed.shape}")
        return X_train_processed, y_train_processed, X_test_processed
    
    return X_train_processed, y_train_processed