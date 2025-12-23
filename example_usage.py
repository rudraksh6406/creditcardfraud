"""
Example Usage of Credit Card Fraud Detection System
====================================================
This script demonstrates how to use the fraud detection system
with your own data or the Kaggle dataset.
"""

from fraud_detection import (
    load_data,
    explore_data,
    preprocess_data,
    handle_class_imbalance,
    train_models,
    evaluate_models,
    get_feature_importance
)

def example_with_synthetic_data():
    """
    Example: Run the pipeline with synthetic data (no file needed).
    """
    print("Example 1: Using Synthetic Data")
    print("=" * 60)
    
    # Load synthetic data (no file path needed)
    df = load_data()
    
    # Run the complete pipeline
    explore_data(df)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    X_train_balanced, y_train_balanced = handle_class_imbalance(
        X_train, y_train, method='smote'
    )
    models = train_models(X_train_balanced, y_train_balanced)
    results = evaluate_models(models, X_test, y_test)
    
    # Get feature importance
    if 'Random Forest' in models:
        get_feature_importance(models['Random Forest'], X_train.columns.tolist())
    
    return models, results


def example_with_kaggle_dataset():
    """
    Example: Run the pipeline with Kaggle dataset.
    
    To use this:
    1. Download the dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    2. Place 'creditcard.csv' in the project directory
    3. Uncomment the code below
    """
    print("Example 2: Using Kaggle Dataset")
    print("=" * 60)
    
    # Uncomment and provide the path to your dataset:
    # df = load_data('creditcard.csv')
    # 
    # # Continue with the pipeline...
    # explore_data(df)
    # X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    # X_train_balanced, y_train_balanced = handle_class_imbalance(
    #     X_train, y_train, method='smote'
    # )
    # models = train_models(X_train_balanced, y_train_balanced)
    # results = evaluate_models(models, X_test, y_test)
    # 
    # return models, results
    
    print("Please download the dataset and uncomment the code above.")
    return None, None


def example_custom_prediction(model, scaler, transaction_data):
    """
    Example: Make predictions on new transaction data.
    
    Parameters:
    -----------
    model : trained sklearn model
    scaler : fitted StandardScaler
    transaction_data : dict or pandas.Series
        New transaction features
    """
    import pandas as pd
    import numpy as np
    
    # Convert to DataFrame if needed
    if isinstance(transaction_data, dict):
        df_new = pd.DataFrame([transaction_data])
    else:
        df_new = pd.DataFrame([transaction_data])
    
    # Apply same preprocessing as training data
    # (handle Time feature, scale, etc.)
    if 'Time' in df_new.columns:
        df_new['Time_hour'] = (df_new['Time'] / 3600) % 24
        df_new['Time_sin'] = np.sin(2 * np.pi * df_new['Time_hour'] / 24)
        df_new['Time_cos'] = np.cos(2 * np.pi * df_new['Time_hour'] / 24)
        df_new = df_new.drop(['Time', 'Time_hour'], axis=1)
    
    # Scale features
    X_new_scaled = scaler.transform(df_new)
    
    # Make prediction
    prediction = model.predict(X_new_scaled)
    probability = model.predict_proba(X_new_scaled)[:, 1]
    
    print(f"Prediction: {'FRAUD' if prediction[0] == 1 else 'NOT FRAUD'}")
    print(f"Fraud Probability: {probability[0]:.4f}")
    
    return prediction, probability


if __name__ == "__main__":
    # Run example with synthetic data
    models, results = example_with_synthetic_data()
    
    # Example: Make a prediction on new data
    # (This is just a demonstration - you would use real transaction data)
    if models:
        print("\n" + "=" * 60)
        print("Example: Making a Prediction on New Transaction")
        print("=" * 60)
        
        # This is a placeholder - in reality, you'd have actual transaction data
        print("\nNote: This is a demonstration. In production, you would:")
        print("1. Collect transaction features (V1-V28, Time, Amount)")
        print("2. Preprocess them the same way as training data")
        print("3. Use the trained model to predict")
        print("\nSee example_custom_prediction() function for code structure.")

