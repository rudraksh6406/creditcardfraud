"""
Credit Card Fraud Detection System
===================================
This script implements a complete machine learning pipeline for detecting
credit card fraud using binary classification.

Author: Credit Card Fraud Detection Project
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(file_path=None):
    """
    Load the credit card fraud dataset.
    
    If no file path is provided, this function will use the advanced
    synthetic data generator to create realistic transaction data.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the CSV file containing the dataset
        
    Returns:
    --------
    df : pandas.DataFrame
        The loaded dataset
    """
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    if file_path:
        # Load from provided file path
        df = pd.read_csv(file_path)
        print(f"✓ Data loaded from: {file_path}")
    else:
        # Use advanced synthetic data generator
        print("⚠ No file path provided. Generating advanced synthetic data...")
        print("   (For real data, download from Kaggle and provide file path)")
        
        try:
            from data_generator import load_or_generate_dataset
            # Generate comprehensive synthetic dataset
            df = load_or_generate_dataset(n_samples=100000, force_regenerate=False)
            print("✓ Advanced synthetic data generated with realistic features")
        except ImportError:
            # Fallback to simple synthetic data if generator not available
            print("   Using fallback synthetic data generator...")
            n_samples = 10000
            n_features = 28
            
            data = np.random.randn(n_samples, n_features)
            time = np.random.randint(0, 172792, n_samples)
            amount = np.random.exponential(88, n_samples)
            
            fraud_rate = 0.0017
            n_fraud = int(n_samples * fraud_rate)
            fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
            
            target = np.zeros(n_samples)
            target[fraud_indices] = 1
            
            data[fraud_indices, :] += np.random.randn(n_fraud, n_features) * 2
            amount[fraud_indices] *= np.random.uniform(1.5, 3.0, n_fraud)
            
            feature_names = [f'V{i+1}' for i in range(n_features)]
            df = pd.DataFrame(data, columns=feature_names)
            df['Time'] = time
            df['Amount'] = amount
            df['Class'] = target.astype(int)
            
            print(f"✓ Basic synthetic data generated: {n_samples} samples")
    
    print(f"✓ Dataset shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)[:10]}..." if len(df.columns) > 10 else f"✓ Columns: {list(df.columns)}")
    print()
    
    return df


def explore_data(df):
    """
    Perform exploratory data analysis on the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to explore
    """
    print("=" * 60)
    print("STEP 2: Exploratory Data Analysis")
    print("=" * 60)
    
    # Basic statistics
    print("\n1. Dataset Overview:")
    print(f"   Total samples: {len(df)}")
    print(f"   Total features: {len(df.columns) - 1}")  # Excluding target
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    # Class distribution
    print("\n2. Class Distribution:")
    class_counts = df['Class'].value_counts()
    print(f"   Not Fraud (0): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.2f}%)")
    print(f"   Fraud (1): {class_counts[1]:,} ({class_counts[1]/len(df)*100:.2f}%)")
    
    # Check for class imbalance
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"   Imbalance Ratio: {imbalance_ratio:.2f}:1")
    print("   ⚠ Strong class imbalance detected!")
    
    # Statistical summary
    print("\n3. Statistical Summary (Amount):")
    print(df['Amount'].describe())
    
    print("\n4. Statistical Summary (Time):")
    print(df['Time'].describe())
    
    print()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Class distribution
    axes[0, 0].bar(['Not Fraud (0)', 'Fraud (1)'], class_counts.values, 
                   color=['green', 'red'], alpha=0.7)
    axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Amount distribution
    axes[0, 1].hist(df[df['Class'] == 0]['Amount'], bins=50, 
                    alpha=0.7, label='Not Fraud', color='green')
    axes[0, 1].hist(df[df['Class'] == 1]['Amount'], bins=50, 
                    alpha=0.7, label='Fraud', color='red')
    axes[0, 1].set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Amount')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Time distribution
    axes[1, 0].hist(df[df['Class'] == 0]['Time'], bins=50, 
                    alpha=0.7, label='Not Fraud', color='green')
    axes[1, 0].hist(df[df['Class'] == 1]['Time'], bins=50, 
                    alpha=0.7, label='Fraud', color='red')
    axes[1, 0].set_title('Transaction Time Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Correlation heatmap (sample of features)
    sample_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount', 'Class']
    if all(col in df.columns for col in sample_features):
        corr_matrix = df[sample_features].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=axes[1, 1], square=True)
        axes[1, 1].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/eda_visualizations.png', dpi=300, bbox_inches='tight')
    print("✓ EDA visualizations saved to: data/eda_visualizations.png")
    print()


def preprocess_data(df):
    """
    Preprocess the data for machine learning.
    
    Steps:
    1. Separate features and target
    2. Handle 'Time' feature (optional: create cyclical features)
    3. Scale features
    4. Split into train and test sets
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The raw dataset
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Preprocessed training and testing sets
    """
    print("=" * 60)
    print("STEP 3: Data Preprocessing")
    print("=" * 60)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Keep only numeric columns to avoid issues with string IDs/labels
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Numeric features used: {len(numeric_cols)}")

    # Optional: Create cyclical features from 'Time'
    # This helps the model understand that time is cyclical (e.g., hour 23 is close to hour 0)
    if 'Time' in X.columns:
        # Convert time to hours of day (assuming time is in seconds)
        X['Time_hour'] = (X['Time'] / 3600) % 24
        X['Time_sin'] = np.sin(2 * np.pi * X['Time_hour'] / 24)
        X['Time_cos'] = np.cos(2 * np.pi * X['Time_hour'] / 24)
        X = X.drop(['Time', 'Time_hour'], axis=1)
        print("✓ Created cyclical time features")

    # Split data into training and testing sets
    # Use stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=RANDOM_STATE,
        stratify=y  # Important for imbalanced datasets
    )
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Scale features (important for distance-based algorithms)
    # Note: We scale after splitting to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for better readability
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("✓ Features scaled using StandardScaler")
    print()
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def handle_class_imbalance(X_train, y_train, method='smote'):
    """
    Handle class imbalance using oversampling or undersampling techniques.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    method : str
        Method to use: 'smote', 'undersample', or 'none'
        
    Returns:
    --------
    X_resampled, y_resampled : arrays
        Resampled training data
    """
    print("=" * 60)
    print("STEP 4: Handling Class Imbalance")
    print("=" * 60)
    
    print(f"Before resampling:")
    print(f"  Class 0 (Not Fraud): {(y_train == 0).sum()}")
    print(f"  Class 1 (Fraud): {(y_train == 1).sum()}")
    
    if method == 'smote':
        # SMOTE: Synthetic Minority Oversampling Technique
        # Creates synthetic samples of the minority class
        print("\nUsing SMOTE (Synthetic Minority Oversampling)...")
        min_class_count = (y_train == 1).sum()
        if min_class_count < 2:
            print("⚠ Not enough minority samples for SMOTE. Skipping resampling.")
            X_resampled, y_resampled = X_train, y_train
        else:
            # Adjust neighbors to avoid errors when minority class is very small
            k_neighbors = max(1, min(5, min_class_count - 1))
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"✓ SMOTE applied successfully (k_neighbors={k_neighbors})")
        
    elif method == 'undersample':
        # Random Undersampling: Randomly remove samples from majority class
        print("\nUsing Random Undersampling...")
        undersampler = RandomUnderSampler(random_state=RANDOM_STATE)
        X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        print("✓ Undersampling applied successfully")
        
    elif method == 'none':
        # No resampling
        print("\nSkipping resampling (using original imbalanced data)")
        X_resampled, y_resampled = X_train, y_train
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"\nAfter resampling:")
    print(f"  Class 0 (Not Fraud): {(y_resampled == 0).sum()}")
    print(f"  Class 1 (Fraud): {(y_resampled == 1).sum()}")
    print()
    
    return X_resampled, y_resampled


def train_models(X_train, y_train):
    """
    Train multiple machine learning models.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    models : dict
        Dictionary of trained models
    """
    print("=" * 60)
    print("STEP 5: Training Machine Learning Models")
    print("=" * 60)
    
    models = {}
    
    # Model 1: Random Forest
    # Good for handling non-linear relationships and feature interactions
    print("\n1. Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'  # Helps with imbalanced data
    )
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    print("   ✓ Random Forest trained")
    
    # Model 2: Logistic Regression
    # Simple, interpretable baseline model
    print("\n2. Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced',  # Helps with imbalanced data
        solver='lbfgs'
    )
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    print("   ✓ Logistic Regression trained")
    
    print()
    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test data.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : array-like
        Test features
    y_test : array-like
        Test target
        
    Returns:
    --------
    results : dict
        Dictionary of evaluation results
    """
    print("=" * 60)
    print("STEP 6: Model Evaluation")
    print("=" * 60)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        # For imbalanced datasets, we focus on:
        # - Precision: Of all predicted frauds, how many were actually fraud?
        # - Recall: Of all actual frauds, how many did we catch?
        # - F1-Score: Harmonic mean of precision and recall
        # - ROC-AUC: Area under ROC curve (good for imbalanced data)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Not Fraud', 'Fraud']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Not Fraud  Fraud")
        print(f"Actual Not Fraud    {cm[0,0]:5d}   {cm[0,1]:5d}")
        print(f"Actual Fraud        {cm[1,0]:5d}   {cm[1,1]:5d}")
        
        # ROC-AUC Score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        
        # Average Precision Score (better for imbalanced data)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        print(f"Average Precision: {avg_precision:.4f}")
        
        # Store results
        results[name] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'confusion_matrix': cm
        }
        
        print()
    
    # Visualize results
    plot_evaluation_results(models, X_test, y_test, results)
    
    return results


def plot_evaluation_results(models, X_test, y_test, results):
    """
    Create visualization plots for model evaluation.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    results : dict
        Dictionary of evaluation results
    """
    n_models = len(results)
    
    # Adjust subplot layout based on number of models
    if n_models <= 2:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    else:
        # For more models, create a larger grid
        rows = 2
        cols = max(2, (n_models + 1) // 2)
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 12))
        if cols == 2:
            axes = axes.reshape(2, 2)
    
    # 1. ROC Curves
    ax1 = axes[0, 0] if n_models <= 2 else axes[0, 0]
    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                # For models without predict_proba, skip
                continue
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = results[name]['roc_auc']
            ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
        except Exception as e:
            print(f"Warning: Could not plot ROC for {name}: {e}")
            continue
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    
    # 2. Precision-Recall Curves
    ax2 = axes[0, 1] if n_models <= 2 else axes[0, 1]
    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                continue
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = results[name]['avg_precision']
            ax2.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})', linewidth=2)
        except Exception as e:
            print(f"Warning: Could not plot PR for {name}: {e}")
            continue
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(alpha=0.3)
    
    # 3. Confusion Matrices - only plot first 2 models in bottom row
    model_list = list(results.items())[:2]
    for idx, (name, result) in enumerate(model_list):
        if n_models <= 2:
            ax = axes[1, idx]
        else:
            # For more models, only show first 2 confusion matrices
            if idx < 2:
                ax = axes[1, idx]
            else:
                break
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Not Fraud', 'Fraud'],
                   yticklabels=['Not Fraud', 'Fraud'])
        ax.set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=11)
    
    # Hide unused subplots if we have fewer models
    if n_models < 2:
        for idx in range(n_models, 2):
            if n_models <= 2:
                axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('models/evaluation_results.png', dpi=300, bbox_inches='tight')
    print("✓ Evaluation visualizations saved to: models/evaluation_results.png")
    print()


def get_feature_importance(model, feature_names, top_n=10):
    """
    Extract and display feature importance from Random Forest model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained Random Forest model
    feature_names : list
        Names of features
    top_n : int
        Number of top features to display
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        print(f"\nTop {top_n} Most Important Features:")
        print("-" * 50)
        for i, idx in enumerate(indices, 1):
            print(f"{i:2d}. {feature_names[idx]:20s} {importances[idx]:.4f}")
        print()
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_n), importances[indices], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Feature importance plot saved to: models/feature_importance.png")


def main():
    """
    Main function to run the complete fraud detection pipeline.
    """
    print("\n" + "=" * 60)
    print("CREDIT CARD FRAUD DETECTION SYSTEM")
    print("=" * 60 + "\n")
    
    # Step 1: Load data
    # For real data, provide the path: load_data('path/to/creditcard.csv')
    df = load_data()
    
    # Step 2: Exploratory Data Analysis
    explore_data(df)
    
    # Step 3: Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Step 4: Handle class imbalance
    # Options: 'smote', 'undersample', or 'none'
    X_train_balanced, y_train_balanced = handle_class_imbalance(
        X_train, y_train, method='smote'
    )
    
    # Step 5: Train models
    models = train_models(X_train_balanced, y_train_balanced)
    
    # Step 6: Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Step 7: Feature importance (for Random Forest)
    if 'Random Forest' in models:
        get_feature_importance(models['Random Forest'], X_train.columns.tolist())
    
    # Step 8: Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nBest Model Performance:")
    best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
    print(f"  Model: {best_model[0]}")
    print(f"  ROC-AUC: {best_model[1]['roc_auc']:.4f}")
    print(f"  Average Precision: {best_model[1]['avg_precision']:.4f}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60 + "\n")
    
    return models, results, scaler


if __name__ == "__main__":
    # Create necessary directories
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the main pipeline
    models, results, scaler = main()

