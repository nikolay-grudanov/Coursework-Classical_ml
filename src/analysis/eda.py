"""
Exploratory Data Analysis for Drug Effectiveness Prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="configs/model_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(filepath):
    """Load data from Excel file."""
    logger.info(f"Loading data from {filepath}")
    data = pd.read_excel(filepath)
    logger.info(f"Data shape: {data.shape}")
    return data

def basic_info(data):
    """Display basic information about the dataset."""
    print("Dataset Info:")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print("\nData types:")
    print(data.dtypes.value_counts())
    return data

def check_missing_values(data):
    """Check for missing values in the dataset."""
    missing = data.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("\nMissing values:")
        print(missing)
        return missing
    else:
        print("\nNo missing values found.")
        return pd.Series(dtype=int)

def check_duplicates(data):
    """Check for duplicate rows."""
    duplicates = data.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    return duplicates

def analyze_target_distributions(data):
    """Analyze distributions of target variables."""
    targets = ['IC50, mM', 'CC50, mM', 'SI']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, target in enumerate(targets):
        if target in data.columns:
            axes[i].hist(data[target], bins=50, alpha=0.7)
            axes[i].set_title(f'{target} Distribution')
            axes[i].set_xlabel(target)
            axes[i].set_ylabel('Frequency')
            
            # Print statistics
            print(f"\n{target} Statistics:")
            print(f"  Mean: {data[target].mean():.4f}")
            print(f"  Median: {data[target].median():.4f}")
            print(f"  Std: {data[target].std():.4f}")
            print(f"  Min: {data[target].min():.4f}")
            print(f"  Max: {data[target].max():.4f}")
    
    plt.tight_layout()
    plt.savefig('figures/target_distributions.png')
    plt.show()
    
    return targets

def create_classification_targets(data):
    """Create binary classification targets."""
    if 'IC50, mM' in data.columns:
        data['IC50_above_median'] = (data['IC50, mM'] > data['IC50, mM'].median()).astype(int)
        
    if 'CC50, mM' in data.columns:
        data['CC50_above_median'] = (data['CC50, mM'] > data['CC50, mM'].median()).astype(int)
        
    if 'SI' in data.columns:
        data['SI_above_median'] = (data['SI'] > data['SI'].median()).astype(int)
        data['SI_above_8'] = (data['SI'] > 8).astype(int)
        
    return data

def analyze_class_balance(data):
    """Analyze class balance for classification targets."""
    class_targets = ['IC50_above_median', 'CC50_above_median', 'SI_above_median', 'SI_above_8']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, target in enumerate(class_targets):
        if target in data.columns:
            class_counts = data[target].value_counts()
            axes[i].bar(class_counts.index, class_counts.values)
            axes[i].set_title(f'{target} Class Distribution')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Count')
            
            # Print balance information
            print(f"\n{target} Class Balance:")
            for class_val, count in class_counts.items():
                percentage = 100 * count / len(data)
                print(f"  Class {class_val}: {count} ({percentage:.1f}%)")
    
    plt.tight_layout()
    plt.savefig('figures/class_distributions.png')
    plt.show()
    
    return class_targets

def check_data_leakage(data, feature_cols, target_cols):
    """Check for potential data leakage."""
    # Check if any features are highly correlated with targets
    correlations = {}
    for target in target_cols:
        if target in data.columns:
            target_corr = data[feature_cols].corrwith(data[target]).abs()
            high_corr_features = target_corr[target_corr > 0.95]
            if len(high_corr_features) > 0:
                correlations[target] = high_corr_features
                print(f"\nHigh correlations with {target}:")
                print(high_corr_features)
    
    return correlations

def correlation_analysis(data, top_n=20):
    """Analyze correlations between features and targets."""
    # Select numeric columns only
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Compute correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Focus on correlations with targets
    targets = ['IC50, mM', 'CC50, mM', 'SI']
    target_correlations = {}
    
    for target in targets:
        if target in corr_matrix.columns:
            # Get correlations with target, excluding the target itself
            target_corr = corr_matrix[target].drop(target).abs().sort_values(ascending=False)
            target_correlations[target] = target_corr.head(top_n)
            
            print(f"\nTop {top_n} features correlated with {target}:")
            print(target_correlations[target])
    
    # Plot correlation heatmap for targets
    target_data = data[targets].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(target_data, annot=True, cmap='coolwarm', center=0)
    plt.title('Target Variables Correlation Matrix')
    plt.savefig('figures/target_correlations.png')
    plt.show()
    
    return target_correlations

def detect_outliers(data, columns=None):
    """Detect outliers using IQR method."""
    if columns is None:
        columns = ['IC50, mM', 'CC50, mM', 'SI']
    
    outliers_info = {}
    
    for col in columns:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': 100 * len(outliers) / len(data),
                'bounds': (lower_bound, upper_bound)
            }
            
            print(f"\n{col} Outliers:")
            print(f"  Count: {len(outliers)} ({100 * len(outliers) / len(data):.2f}%)")
            print(f"  Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
    
    return outliers_info

def main():
    """Main function to run EDA."""
    # Create directories if they don't exist
    Path("figures").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    # Load configuration
    config = load_config()
    
    # Load data
    data = load_data(config['data_path'])
    
    # Basic information
    basic_info(data)
    
    # Check data quality
    missing = check_missing_values(data)
    duplicates = check_duplicates(data)
    
    # Create classification targets
    data = create_classification_targets(data)
    
    # Analyze target distributions
    targets = analyze_target_distributions(data)
    
    # Analyze class balance
    class_targets = analyze_class_balance(data)
    
    # Correlation analysis
    target_correlations = correlation_analysis(data)
    
    # Outlier detection
    outliers_info = detect_outliers(data)
    
    # Check for data leakage
    feature_cols = [col for col in data.columns if col not in ['Unnamed: 0', 'IC50, mM', 'CC50, mM', 'SI'] 
                   and not col.startswith('IC50_above') and not col.startswith('CC50_above') 
                   and not col.startswith('SI_above')]
    target_cols = ['IC50, mM', 'CC50, mM', 'SI']
    leakage = check_data_leakage(data, feature_cols, target_cols)
    
    # Save processed data
    output_path = "data/processed_data.csv"
    Path(output_path).parent.mkdir(exist_ok=True)
    data.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    
    # Create EDA report
    create_eda_report(data, missing, duplicates, targets, class_targets, target_correlations, outliers_info, leakage)
    
    return data

def create_eda_report(data, missing, duplicates, targets, class_targets, target_correlations, outliers_info, leakage):
    """Create a markdown report of the EDA findings."""
    report = []
    report.append("# Exploratory Data Analysis Report\n")
    report.append("## Dataset Overview\n")
    report.append(f"The dataset contains {data.shape[0]} chemical compounds with {data.shape[1]} features.\n")
    report.append("\n## Data Quality\n")
    
    if len(missing) > 0:
        report.append(f"- Missing values found in {len(missing)} columns\n")
    else:
        report.append("- No missing values found\n")
        
    report.append(f"- Duplicate rows: {duplicates}\n")
    
    report.append("\n## Target Variables\n")
    for target in ['IC50, mM', 'CC50, mM', 'SI']:
        if target in data.columns:
            report.append(f"\n### {target}")
            report.append(f"- Mean: {data[target].mean():.4f}")
            report.append(f"- Median: {data[target].median():.4f}")
            report.append(f"- Std: {data[target].std():.4f}")
            report.append(f"- Range: [{data[target].min():.4f}, {data[target].max():.4f}]")
    
    report.append("\n## Classification Targets\n")
    for target in class_targets:
        if target in data.columns:
            class_counts = data[target].value_counts()
            report.append(f"\n### {target}")
            for class_val, count in class_counts.items():
                percentage = 100 * count / len(data)
                report.append(f"- Class {class_val}: {count} ({percentage:.1f}%)")
    
    report.append("\n## Outliers\n")
    for col, info in outliers_info.items():
        report.append(f"- {col}: {info['count']} outliers ({info['percentage']:.2f}%)")
    
    if leakage:
        report.append("\n## Data Leakage Check")
        report.append("Potential data leakage detected in the following features:")
        for target, features in leakage.items():
            report.append(f"- {target}: {len(features)} highly correlated features")
    else:
        report.append("\n## Data Leakage Check")
        report.append("No significant data leakage detected.")
    
    # Save report
    with open("reports/eda_T1.md", "w") as f:
        f.write("\n".join(report))
    
    print("\nEDA report saved to reports/eda_T1.md")

if __name__ == "__main__":
    data = main()