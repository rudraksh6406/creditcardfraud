# Credit Card Fraud Detection System

A complete machine learning system for detecting credit card fraud using binary classification. This project demonstrates best practices for handling imbalanced datasets and building production-ready fraud detection models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow Explanation](#workflow-explanation)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Key Concepts](#key-concepts)

## 🎯 Overview

This project implements a complete fraud detection pipeline that:
- Handles highly imbalanced datasets (fraud cases are rare)
- Preprocesses transaction data
- Trains multiple machine learning models
- Evaluates performance using appropriate metrics
- Provides visualizations and insights

## ✨ Features

### Machine Learning
- **Binary Classification**: Predicts fraud (1) vs. not fraud (0)
- **Class Imbalance Handling**: Uses SMOTE (Synthetic Minority Oversampling Technique)
- **Multiple Models**: Random Forest and Logistic Regression
- **Comprehensive Evaluation**: ROC-AUC, Precision-Recall, Confusion Matrices
- **Feature Engineering**: Cyclical time features, feature scaling
- **Visualizations**: EDA plots, model performance metrics, feature importance
- **Beginner-Friendly**: Well-commented code with step-by-step explanations

### Web Application
- **Interactive Web Interface**: Beautiful, modern UI for fraud detection
- **Real-Time Predictions**: Get instant fraud predictions on transactions
- **Sample Generator**: Generate random transaction samples for testing
- **Model Management**: View model metrics and retrain models
- **Visual Feedback**: Color-coded results with probability bars
- **Responsive Design**: Works on desktop and mobile devices

## 📁 Project Structure

```
creditcardfraud/
│
├── app.py                 # Flask web application (main entry point)
├── fraud_detection.py     # ML pipeline functions
├── example_usage.py       # Usage examples
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── templates/            # HTML templates (created automatically)
│   └── index.html        # Web interface
│
├── static/               # Static files (CSS, JS, images)
│
├── data/                 # Data directory (created automatically)
│   └── eda_visualizations.png
│
└── models/               # Models directory (created automatically)
    ├── fraud_model.pkl   # Trained model
    ├── scaler.pkl        # Feature scaler
    ├── model_metrics.json # Model performance metrics
    ├── evaluation_results.png
    └── feature_importance.png
```

## 🚀 Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Option 1: Run the Web Application (Recommended) 🌐

The easiest way to use the fraud detection system is through the web interface:

1. **Install dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask web application**:
   ```bash
   python app.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

4. **Use the web interface** to:
   - Enter transaction details manually
   - Generate random sample transactions
   - Get real-time fraud predictions
   - View model performance metrics
   - Retrain the model if needed

The web app will automatically train a model on first run if one doesn't exist.

### Option 2: Run the Python Script

```bash
python fraud_detection.py
```

This runs the complete ML pipeline and generates visualizations.

### Option 3: Use with Real Kaggle Dataset

1. Download the Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place the `creditcard.csv` file in the project directory
3. Modify the `load_data()` call in `app.py` or `fraud_detection.py`:
   ```python
   df = load_data('creditcard.csv')
   ```

### Option 4: Use in Jupyter Notebook

The code can be easily adapted for Jupyter notebooks. Simply copy the functions into notebook cells.

## 🔄 Workflow Explanation

The fraud detection pipeline follows these steps:

### Step 1: Data Loading
- Loads the credit card transaction dataset
- If no file is provided, generates synthetic data for demonstration
- Displays basic dataset information

### Step 2: Exploratory Data Analysis (EDA)
- Analyzes class distribution (highly imbalanced: ~0.17% fraud)
- Examines transaction amounts and time patterns
- Creates visualizations to understand the data
- Identifies data quality issues

### Step 3: Data Preprocessing
- **Feature-Target Separation**: Splits features (X) from target (y)
- **Feature Engineering**: 
  - Creates cyclical time features (sin/cos) to capture time patterns
  - Handles the 'Time' feature appropriately
- **Scaling**: Standardizes features using StandardScaler
- **Train-Test Split**: 80/20 split with stratification to maintain class distribution

### Step 4: Handling Class Imbalance
- **Problem**: Fraud cases are extremely rare (~0.17% of transactions)
- **Solution**: Uses SMOTE to create synthetic fraud samples
- **Result**: Balanced dataset for training

### Step 5: Model Training
- **Random Forest**: 
  - Handles non-linear relationships
  - Provides feature importance
  - Uses class_weight='balanced'
- **Logistic Regression**: 
  - Simple, interpretable baseline
  - Fast training and prediction

### Step 6: Model Evaluation
- **Classification Report**: Precision, Recall, F1-Score
- **Confusion Matrix**: True/False Positives and Negatives
- **ROC-AUC Score**: Overall model performance
- **Average Precision**: Better metric for imbalanced data
- **Visualizations**: ROC curves, Precision-Recall curves

### Step 7: Feature Importance
- Identifies which features are most predictive
- Helps understand what drives fraud detection

## 📊 Dataset

### Kaggle Credit Card Fraud Dataset

The real dataset contains:
- **284,807 transactions** (492 frauds, 0.17% fraud rate)
- **Features**: 
  - `Time`: Seconds elapsed between transaction and first transaction
  - `Amount`: Transaction amount
  - `V1-V28`: PCA-transformed features (for privacy)
  - `Class`: Target variable (0 = Not Fraud, 1 = Fraud)

### Synthetic Data

If you don't have the real dataset, the script generates synthetic data with similar characteristics for demonstration purposes.

## 📈 Model Performance

The models are evaluated using metrics appropriate for imbalanced datasets:

- **ROC-AUC**: Measures the model's ability to distinguish between classes
- **Average Precision**: Focuses on the positive class (fraud)
- **Precision**: Of predicted frauds, how many are actually fraud?
- **Recall**: Of actual frauds, how many did we catch?

**Note**: For fraud detection, we typically prioritize **Recall** (catching all frauds) over Precision (avoiding false alarms), but the balance depends on business requirements.

## 🧠 Key Concepts

### 1. Class Imbalance
- **Problem**: When one class (fraud) is much rarer than another (normal transactions)
- **Impact**: Models may predict everything as the majority class
- **Solutions**: 
  - SMOTE (oversampling minority class)
  - Undersampling (reducing majority class)
  - Class weights (penalizing misclassification of minority class)

### 2. Evaluation Metrics for Imbalanced Data
- **Accuracy** is misleading (can be 99%+ by predicting all as "not fraud")
- **ROC-AUC**: Good overall metric
- **Precision-Recall**: Better for imbalanced data
- **F1-Score**: Balance between precision and recall

### 3. Feature Scaling
- Important for distance-based algorithms (Logistic Regression)
- StandardScaler: Centers data (mean=0) and scales (std=1)
- Applied after train-test split to avoid data leakage

### 4. Stratified Splitting
- Ensures both train and test sets have similar class distributions
- Critical for imbalanced datasets

## 🔧 Customization

### Adjust Resampling Method
In `main()`, change:
```python
X_train_balanced, y_train_balanced = handle_class_imbalance(
    X_train, y_train, method='smote'  # Options: 'smote', 'undersample', 'none'
)
```

### Add More Models
In `train_models()`, add:
```python
from sklearn.svm import SVC
svm_model = SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE)
svm_model.fit(X_train, y_train)
models['SVM'] = svm_model
```

### Adjust Model Hyperparameters
Modify parameters in `train_models()` function.

## 📝 Notes

- The code is designed to be educational and beginner-friendly
- All steps are clearly commented
- Visualizations are automatically saved
- The pipeline is modular and easy to extend

## 🤝 Contributing

Feel free to fork this project and add improvements:
- Additional models (XGBoost, Neural Networks)
- Advanced feature engineering
- Hyperparameter tuning
- Model deployment scripts

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Kaggle for providing the Credit Card Fraud Detection dataset
- scikit-learn and imbalanced-learn communities for excellent ML libraries

---

**Happy Learning! 🎓**

