"""
Advanced Machine Learning Models for Fraud Detection
=====================================================
Implements state-of-the-art models including XGBoost, LightGBM, and Neural Networks.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings

# Optional imports with graceful degradation
XGB_AVAILABLE = True
LGBM_AVAILABLE = True
TF_AVAILABLE = True

try:
    import xgboost as xgb
except Exception:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
except Exception:
    LGBM_AVAILABLE = False

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models
except Exception:
    TF_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42


def train_xgboost(X_train, y_train, **kwargs):
    """
    Train XGBoost model - one of the best performing models for fraud detection.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    **kwargs : dict
        Additional XGBoost parameters
        
    Returns:
    --------
    model : XGBClassifier
        Trained XGBoost model
    """
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0.1,
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        **kwargs
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model


def train_lightgbm(X_train, y_train, **kwargs):
    """
    Train LightGBM model - fast and efficient gradient boosting.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    **kwargs : dict
        Additional LightGBM parameters
        
    Returns:
    --------
    model : LGBMClassifier
        Trained LightGBM model
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
        **kwargs
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    return model


def train_neural_network(X_train, y_train, epochs=50, batch_size=32):
    """
    Train a Neural Network for fraud detection.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    model : Keras Sequential model
        Trained neural network
    """
    n_features = X_train.shape[1]
    
    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # Build neural network
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(n_features,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        verbose=0,
        validation_split=0.2
    )
    
    return model


def train_advanced_models(X_train, y_train):
    """
    Train all advanced models for fraud detection.
    
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
    print("Training Advanced ML Models")
    print("=" * 60)
    
    models = {}
    
    # 1. XGBoost
    print("\n1. Training XGBoost...")
    try:
        models['XGBoost'] = train_xgboost(X_train, y_train)
        print("   ✓ XGBoost trained successfully")
    except Exception as e:
        print(f"   ✗ XGBoost training failed: {e}")
    
    # 2. LightGBM
    print("\n2. Training LightGBM...")
    try:
        models['LightGBM'] = train_lightgbm(X_train, y_train)
        print("   ✓ LightGBM trained successfully")
    except Exception as e:
        print(f"   ✗ LightGBM training failed: {e}")
    
    # 3. Random Forest (Enhanced)
    print("\n3. Training Enhanced Random Forest...")
    try:
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        models['Random Forest'].fit(X_train, y_train)
        print("   ✓ Random Forest trained successfully")
    except Exception as e:
        print(f"   ✗ Random Forest training failed: {e}")
    
    # 4. Gradient Boosting
    print("\n4. Training Gradient Boosting...")
    try:
        models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE
        )
        models['Gradient Boosting'].fit(X_train, y_train)
        print("   ✓ Gradient Boosting trained successfully")
    except Exception as e:
        print(f"   ✗ Gradient Boosting training failed: {e}")
    
    # 5. Neural Network (if TensorFlow available)
    print("\n5. Training Neural Network...")
    try:
        models['Neural Network'] = train_neural_network(X_train, y_train, epochs=30)
        print("   ✓ Neural Network trained successfully")
    except Exception as e:
        print(f"   ✗ Neural Network training failed: {e}")
        print("   (TensorFlow may not be available)")
    
    # 6. Logistic Regression (Baseline)
    print("\n6. Training Logistic Regression...")
    try:
        models['Logistic Regression'] = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            solver='lbfgs',
            C=1.0
        )
        models['Logistic Regression'].fit(X_train, y_train)
        print("   ✓ Logistic Regression trained successfully")
    except Exception as e:
        print(f"   ✗ Logistic Regression training failed: {e}")
    
    print(f"\n✓ Total models trained: {len(models)}")
    print()
    
    return models


def predict_with_ensemble(models, X_test):
    """
    Make predictions using ensemble of all models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : array-like
        Test features
        
    Returns:
    --------
    predictions : dict
        Dictionary with predictions from each model
    probabilities : dict
        Dictionary with probabilities from each model
    ensemble_prediction : array
        Ensemble prediction (majority vote)
    ensemble_probability : array
        Average probability across all models
    """
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        try:
            if name == 'Neural Network':
                # Neural network returns probabilities directly
                prob = model.predict(X_test, verbose=0).flatten()
                pred = (prob > 0.5).astype(int)
            else:
                # Standard sklearn models
                pred = model.predict(X_test)
                prob = model.predict_proba(X_test)[:, 1]
            
            predictions[name] = pred
            probabilities[name] = prob
        except Exception as e:
            print(f"Warning: Prediction failed for {name}: {e}")
    
    # Ensemble: Average probabilities and majority vote
    if probabilities:
        prob_array = np.array(list(probabilities.values()))
        ensemble_probability = np.mean(prob_array, axis=0)
        ensemble_prediction = (ensemble_probability > 0.5).astype(int)
        
        return predictions, probabilities, ensemble_prediction, ensemble_probability
    
    return predictions, probabilities, None, None

