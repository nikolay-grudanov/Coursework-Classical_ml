"""
Module for evaluating model performance and generating comparison reports.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

def evaluate_regression_performance(test_target, predictions, model_name):
    """
    Evaluate regression model performance.
    
    Args:
        test_target: Actual target values
        predictions: Predicted values
        model_name: Name of the model
        
    Returns:
        dict: Performance metrics
    """
    mse = mean_squared_error(test_target, predictions)
    r2 = r2_score(test_target, predictions)
    
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2
    }
    
    logger.info(f"{model_name} - MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}, R2: {r2:.4f}")
    return metrics

def evaluate_classification_performance(test_target, predictions, model_name):
    """
    Evaluate classification model performance.
    
    Args:
        test_target: Actual target values
        predictions: Predicted values
        model_name: Name of the model
        
    Returns:
        dict: Performance metrics
    """
    accuracy = accuracy_score(test_target, predictions)
    
    metrics = {
        'accuracy': accuracy
    }
    
    logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")
    return metrics

def generate_regression_comparison_report(results, output_path):
    """
    Generate a comparison report for regression models.
    
    Args:
        results (dict): Results from all regression models
        output_path (str): Path to save the report
    """
    comparison_data = []
    
    # Collect metrics for all models
    for target_name, target_results in results.items():
        if 'Regression' in target_name:
            for model_name, model_results in target_results.items():
                if isinstance(model_results, dict) and 'r2' in model_results:
                    comparison_data.append({
                        'Target': target_name.replace('Regression', ''),
                        'Model': model_name,
                        'MSE': model_results['mse'],
                        'RMSE': model_results['rmse'],
                        'R2': model_results['r2']
                    })
    
    # Create DataFrame for easy comparison
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path.replace('.md', '.csv'), index=False)
    
    # Generate markdown report
    report = "# Regression Models Comparison Report\n\n"
    report += "## Performance Metrics\n\n"
    report += comparison_df.to_markdown(index=False) + "\n\n"
    
    # Add best performing models
    report += "## Best Performing Models by Target\n\n"
    best_models = comparison_df.loc[comparison_df.groupby('Target')['R2'].idxmax()]
    report += best_models.to_markdown(index=False) + "\n\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Generated regression comparison report at {output_path}")

def generate_classification_comparison_report(results, output_path):
    """
    Generate a comparison report for classification models.
    
    Args:
        results (dict): Results from all classification models
        output_path (str): Path to save the report
    """
    comparison_data = []
    
    # Collect metrics for all models
    for target_name, target_results in results.items():
        if 'Classification' in target_name:
            for model_name, model_results in target_results.items():
                if isinstance(model_results, dict) and 'accuracy' in model_results:
                    comparison_data.append({
                        'Target': target_name.replace('Classification', ''),
                        'Model': model_name,
                        'Accuracy': model_results['accuracy']
                    })
    
    # Create DataFrame for easy comparison
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path.replace('.md', '.csv'), index=False)
    
    # Generate markdown report
    report = "# Classification Models Comparison Report\n\n"
    report += "## Performance Metrics\n\n"
    report += comparison_df.to_markdown(index=False) + "\n\n"
    
    # Add best performing models
    report += "## Best Performing Models by Target\n\n"
    best_models = comparison_df.loc[comparison_df.groupby('Target')['Accuracy'].idxmax()]
    report += best_models.to_markdown(index=False) + "\n\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Generated classification comparison report at {output_path}")

def plot_actual_vs_predicted(test_targets, predictions, target_names, output_path):
    """
    Create plots comparing actual vs predicted values.
    
    Args:
        test_targets (list): List of actual target values
        predictions (list): List of predicted values
        target_names (list): Names of the targets
        output_path (str): Path to save the plots
    """
    fig, axes = plt.subplots(1, len(target_names), figsize=(5*len(target_names), 5))
    
    if len(target_names) == 1:
        axes = [axes]
    
    for i, (target, pred, name) in enumerate(zip(test_targets, predictions, target_names)):
        axes[i].scatter(target, pred, alpha=0.7)
        axes[i].plot([target.min(), target.max()], [target.min(), target.max()], 'r--', lw=2)
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        axes[i].set_title(f'{name} - Actual vs Predicted')
        
        # Calculate R2 for display
        r2 = r2_score(target, pred)
        axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved actual vs predicted plots to {output_path}")