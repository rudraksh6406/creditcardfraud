"""
Flask Web Application for Credit Card Fraud Detection
======================================================
This is the main application file that runs the web server.
"""

from flask import Flask, render_template, request, jsonify
import os
import json
from fraud_detection import (
    load_data, preprocess_data, handle_class_imbalance, 
    train_models, evaluate_models
)
from advanced_models import train_advanced_models, predict_with_ensemble
from api_key_manager import (
    require_api_key, generate_api_key, validate_api_key,
    list_api_keys, revoke_api_key
)
from chatbot import get_chatbot_response, get_fraud_explanation
from analytics import analytics_tracker
from logger_config import logger
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fraud-detection-secret-key-2024'

# Global variables to store model and scaler
model = None
scaler = None
model_name = None
model_metrics = None
all_models = None  # Store all trained models for ensemble
feature_names = None  # Store feature names used during training

# Paths
MODEL_DIR = 'models'
STATIC_DIR = 'static'
TEMPLATES_DIR = 'templates'

# Create directories if they don't exist
for directory in [MODEL_DIR, STATIC_DIR, TEMPLATES_DIR]:
    os.makedirs(directory, exist_ok=True)


def load_or_train_model():
    """
    Load existing model or train a new one if it doesn't exist.
    """
    global model, scaler, model_name, model_metrics, all_models, feature_names
    
    model_path = os.path.join(MODEL_DIR, 'fraud_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    metrics_path = os.path.join(MODEL_DIR, 'model_metrics.json')
    all_models_path = os.path.join(MODEL_DIR, 'all_models.pkl')
    feature_names_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
    
    # Check if model exists
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            print("Loading existing model...")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Load feature names
            if os.path.exists(feature_names_path):
                feature_names = joblib.load(feature_names_path)
                print(f"✓ Loaded {len(feature_names)} feature names")
            else:
                feature_names = None
                print("⚠ Warning: Feature names not found, may cause prediction errors")
            
            # Try to load all models for ensemble
            if os.path.exists(all_models_path):
                try:
                    all_models = joblib.load(all_models_path)
                    print(f"✓ Loaded {len(all_models)} models for ensemble")
                except:
                    all_models = None
            
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    model_metrics = json.load(f)
                    model_name = model_metrics.get('best_model', 'Random Forest')
            else:
                model_name = 'Random Forest'
            
            print(f"✓ Model loaded successfully: {model_name}")
            logger.info(f"Model loaded: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            print(f"Error loading model: {e}")
            return False
    else:
        print("No existing model found. Training new model...")
        logger.info("Training new model...")
        return train_new_model(use_advanced=True)


def train_new_model(use_advanced=True):
    """
    Train a new fraud detection model with advanced algorithms.
    """
    global model, scaler, model_name, model_metrics, all_models
    
    try:
        logger.info("Starting model training...")
        
        # Load and preprocess data
        df = load_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = handle_class_imbalance(
            X_train, y_train, method='smote'
        )
        
        # Train models (advanced if available, otherwise basic)
        if use_advanced:
            try:
                models = train_advanced_models(X_train_balanced, y_train_balanced)
                all_models = models
            except Exception as e:
                logger.warning(f"Advanced models failed, using basic models: {e}")
                models = train_models(X_train_balanced, y_train_balanced)
                all_models = models
        else:
            models = train_models(X_train_balanced, y_train_balanced)
            all_models = models
        
        # Evaluate and get metrics
        results = evaluate_models(models, X_test, y_test)
        
        # Select best model based on ROC-AUC
        best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
        model = models[best_model_name]
        model_name = best_model_name
        
        # Save metrics for all models
        model_metrics = {
            'best_model': best_model_name,
            'models': {},
            'trained_at': datetime.now().isoformat()
        }
        
        for name, result in results.items():
            model_metrics['models'][name] = {
                'roc_auc': float(result['roc_auc']),
                'avg_precision': float(result['avg_precision'])
            }
        
        model_metrics['roc_auc'] = float(results[best_model_name]['roc_auc'])
        model_metrics['avg_precision'] = float(results[best_model_name]['avg_precision'])
        
        # Update analytics
        analytics_tracker.update_model_performance(best_model_name, model_metrics['models'][best_model_name])
        
        # Save feature names used during training
        global feature_names
        feature_names = list(X_train.columns)
        
        # Save model and scaler
        joblib.dump(model, os.path.join(MODEL_DIR, 'fraud_model.pkl'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
        joblib.dump(models, os.path.join(MODEL_DIR, 'all_models.pkl'))
        joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))
        
        # Save metrics
        with open(os.path.join(MODEL_DIR, 'model_metrics.json'), 'w') as f:
            json.dump(model_metrics, f, indent=2)
        
        logger.info(f"Model training completed. Best model: {best_model_name}")
        print("✓ Model trained and saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        traceback.print_exc()
        return False


def preprocess_transaction(transaction_data):
    """
    Preprocess a single transaction for prediction.
    Matches the exact features used during model training.
    
    Parameters:
    -----------
    transaction_data : dict
        Dictionary containing transaction features
        
    Returns:
    --------
    processed_data : numpy array
        Preprocessed transaction ready for prediction
    """
    global scaler, feature_names
    
    # Convert to DataFrame
    df = pd.DataFrame([transaction_data])
    
    # Extract Time if present
    time_value = df['Time'].iloc[0] if 'Time' in df.columns else 12345
    
    # Create all features that match training data
    # Handle Time feature (create cyclical features)
    if 'Time' in df.columns or time_value is not None:
        time_val = time_value if 'Time' in df.columns else 12345
        df['Time_hour'] = (time_val / 3600) % 24
        df['Time_sin'] = np.sin(2 * np.pi * df['Time_hour'] / 24)
        df['Time_cos'] = np.cos(2 * np.pi * df['Time_hour'] / 24)
        if 'Time' in df.columns:
            df = df.drop(['Time'], axis=1)
        df = df.drop(['Time_hour'], axis=1)
    
    # Ensure we have all numeric features that were used during training
    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].copy()
    
    # If we have feature names from training, ensure all are present
    if feature_names is not None:
        # Create a DataFrame with all required features
        df_final = pd.DataFrame(index=[0])
        
        for feat_name in feature_names:
            if feat_name in df_numeric.columns:
                df_final[feat_name] = df_numeric[feat_name].iloc[0]
            else:
                # Fill missing features with default values
                # For time-derived features, calculate from Time if available
                if feat_name == 'Time_sin' and 'Time_sin' not in df_final.columns:
                    time_val = transaction_data.get('Time', 12345)
                    df_final[feat_name] = np.sin(2 * np.pi * ((time_val / 3600) % 24) / 24)
                elif feat_name == 'Time_cos' and 'Time_cos' not in df_final.columns:
                    time_val = transaction_data.get('Time', 12345)
                    df_final[feat_name] = np.cos(2 * np.pi * ((time_val / 3600) % 24) / 24)
                elif feat_name.startswith('V') and feat_name[1:].isdigit():
                    # V1-V28 features - use provided value or 0
                    df_final[feat_name] = transaction_data.get(feat_name, 0.0)
                else:
                    # Other features - use provided value or reasonable default
                    defaults = {
                        'Hour': transaction_data.get('Hour', int((time_value / 3600) % 24)) if 'Time' in transaction_data else 12,
                        'DayOfWeek': transaction_data.get('DayOfWeek', 0),
                        'DayOfMonth': transaction_data.get('DayOfMonth', 15),
                        'DistanceFromHome': transaction_data.get('DistanceFromHome', 50.0),
                        'TimeSinceLastTransaction': transaction_data.get('TimeSinceLastTransaction', 3600.0),
                        'TransactionsLast24H': transaction_data.get('TransactionsLast24H', 2),
                        'Latitude': transaction_data.get('Latitude', 40.7128),
                        'Longitude': transaction_data.get('Longitude', -74.0060),
                        'MCC': transaction_data.get('MCC', 5411),
                        'Amount': transaction_data.get('Amount', 100.0)
                    }
                    df_final[feat_name] = transaction_data.get(feat_name, defaults.get(feat_name, 0.0))
        
        # Ensure columns are in the same order as training
        df_final = df_final[feature_names]
        df = df_final
    else:
        # Fallback: use what we have
        df = df_numeric
    
    # Scale features
    if scaler is not None:
        try:
            processed_data = scaler.transform(df)
        except Exception as e:
            logger.error(f"Scaling error: {e}. Feature mismatch detected.")
            # Try to align features
            if feature_names:
                df_aligned = pd.DataFrame(index=[0])
                for feat in feature_names:
                    if feat in df.columns:
                        df_aligned[feat] = df[feat].iloc[0]
                    else:
                        df_aligned[feat] = 0.0
                df_aligned = df_aligned[feature_names]
                processed_data = scaler.transform(df_aligned)
            else:
                raise e
    else:
        processed_data = df.values
    
    return processed_data


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
@require_api_key
def predict():
    """
    API endpoint to predict fraud for a transaction.
    Uses ensemble method if multiple models are available.
    """
    global model, scaler, all_models
    
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded. Please train a model first.'
        }), 500
    
    try:
        # Get transaction data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess transaction
        processed_data = preprocess_transaction(data)
        
        # Use ensemble if available, otherwise single model
        use_ensemble = all_models is not None and len(all_models) > 1
        
        if use_ensemble:
            try:
                _, _, ensemble_pred, ensemble_prob = predict_with_ensemble(
                    all_models, processed_data
                )
                prediction = ensemble_pred[0] if ensemble_pred is not None else model.predict(processed_data)[0]
                fraud_probability = float(ensemble_prob[0]) if ensemble_prob is not None else float(model.predict_proba(processed_data)[0][1])
                not_fraud_probability = 1 - fraud_probability
                model_used = 'Ensemble'
            except:
                # Fallback to single model
                prediction = model.predict(processed_data)[0]
                probability = model.predict_proba(processed_data)[0]
                fraud_probability = float(probability[1])
                not_fraud_probability = float(probability[0])
                model_used = model_name
        else:
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0]
            fraud_probability = float(probability[1])
            not_fraud_probability = float(probability[0])
            model_used = model_name
        
        result = {
            'prediction': int(prediction),
            'is_fraud': bool(prediction == 1),
            'fraud_probability': fraud_probability,
            'not_fraud_probability': not_fraud_probability,
            'confidence': max(fraud_probability, not_fraud_probability),
            'model_used': model_used
        }
        
        # Log to analytics
        analytics_tracker.log_prediction(result, data)
        logger.info(f"Prediction made: Fraud={result['is_fraud']}, Prob={fraud_probability:.3f}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/api/train', methods=['POST'])
@require_api_key
def train():
    """
    API endpoint to train a new model.
    """
    try:
        success = train_new_model()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model trained successfully',
                'metrics': model_metrics
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to train model'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Training error: {str(e)}'
        }), 500


@app.route('/api/model_info', methods=['GET'])
def model_info():
    """
    API endpoint to get information about the current model.
    """
    global model, model_name, model_metrics
    
    if model is None:
        return jsonify({
            'loaded': False,
            'message': 'No model loaded'
        })
    
    info = {
        'loaded': True,
        'model_name': model_name or 'Unknown',
        'metrics': model_metrics or {}
    }
    
    return jsonify(info)


@app.route('/api/generate_sample', methods=['GET'])
@require_api_key
def generate_sample():
    """
    API endpoint to generate a sample transaction for testing.
    Includes all features that the model expects.
    """
    # Generate a random sample transaction with all required features
    time_val = int(np.random.randint(0, 172792))
    hour = int((time_val / 3600) % 24)
    
    sample = {
        'Time': time_val,
        'Amount': float(np.random.exponential(88)),
        'Hour': hour,
        'DayOfWeek': np.random.randint(0, 7),
        'DayOfMonth': np.random.randint(1, 29),
        'DistanceFromHome': float(np.random.exponential(50)),
        'TimeSinceLastTransaction': float(np.random.exponential(3600)),
        'TransactionsLast24H': np.random.poisson(2),
        'Latitude': float(40.7128 + np.random.normal(0, 0.1)),
        'Longitude': float(-74.0060 + np.random.normal(0, 0.1)),
        'MCC': int(np.random.choice([5411, 5542, 5812, 5999, 5732, 4722, 7011]))
    }
    
    # Generate V1-V28 features
    for i in range(1, 29):
        sample[f'V{i}'] = float(np.random.randn())
    
    return jsonify(sample)


@app.route('/api/generate_key', methods=['POST'])
def generate_key():
    """
    API endpoint to generate a new API key.
    Requires no authentication (public endpoint for initial setup).
    """
    try:
        data = request.get_json() or {}
        name = data.get('name', 'Web Interface Key')
        expires_days = data.get('expires_days')
        
        api_key = generate_api_key(name=name, expires_days=expires_days)
        
        return jsonify({
            'success': True,
            'api_key': api_key,
            'message': 'API key generated successfully. Save it securely!'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/validate_key', methods=['POST'])
def validate_key_endpoint():
    """
    API endpoint to validate an API key.
    """
    try:
        data = request.get_json()
        if not data or 'api_key' not in data:
            return jsonify({
                'valid': False,
                'error': 'API key not provided'
            }), 400
        
        api_key = data['api_key']
        valid, key_info = validate_api_key(api_key)
        
        if valid:
            return jsonify({
                'valid': True,
                'key_info': {
                    'name': key_info.get('name'),
                    'usage_count': key_info.get('usage_count', 0),
                    'last_used': key_info.get('last_used')
                }
            })
        else:
            return jsonify({
                'valid': False,
                'error': 'Invalid or expired API key'
            }), 401
            
    except Exception as e:
        return jsonify({
            'valid': False,
            'error': str(e)
        }), 500


@app.route('/api/chatbot', methods=['POST'])
@require_api_key
def chatbot():
    """
    API endpoint for the AI chatbot.
    """
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Message not provided'
            }), 400
        
        user_message = data['message']
        conversation_history = data.get('history', [])
        
        # Get chatbot response
        response, error = get_chatbot_response(user_message, conversation_history)
        
        if error:
            return jsonify({
                'error': error
            }), 500
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Chatbot error: {str(e)}'
        }), 500


@app.route('/api/explain_prediction', methods=['POST'])
@require_api_key
def explain_prediction():
    """
    API endpoint to get AI explanation of a fraud prediction.
    """
    try:
        data = request.get_json()
        if not data or 'prediction' not in data:
            return jsonify({
                'error': 'Prediction data not provided'
            }), 400
        
        prediction_result = data['prediction']
        
        # Get explanation
        explanation = get_fraud_explanation(prediction_result)
        
        return jsonify({
            'success': True,
            'explanation': explanation
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Explanation error: {str(e)}'
        }), 500


@app.route('/api/batch_predict', methods=['POST'])
@require_api_key
def batch_predict():
    """API endpoint for batch predictions on multiple transactions."""
    global model, scaler, all_models
    
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Please train a model first.'}), 500
    
    try:
        data = request.get_json()
        if not data or 'transactions' not in data:
            return jsonify({'error': 'Transactions array not provided'}), 400
        
        transactions = data['transactions']
        if not isinstance(transactions, list):
            return jsonify({'error': 'Transactions must be a list'}), 400
        
        results = []
        for i, transaction in enumerate(transactions):
            try:
                processed_data = preprocess_transaction(transaction)
                
                if all_models is not None and len(all_models) > 1:
                    try:
                        _, _, ensemble_pred, ensemble_prob = predict_with_ensemble(all_models, processed_data)
                        prediction = ensemble_pred[0] if ensemble_pred is not None else model.predict(processed_data)[0]
                        fraud_probability = float(ensemble_prob[0]) if ensemble_prob is not None else float(model.predict_proba(processed_data)[0][1])
                    except:
                        prediction = model.predict(processed_data)[0]
                        probability = model.predict_proba(processed_data)[0]
                        fraud_probability = float(probability[1])
                else:
                    prediction = model.predict(processed_data)[0]
                    probability = model.predict_proba(processed_data)[0]
                    fraud_probability = float(probability[1])
                
                result = {
                    'index': i,
                    'prediction': int(prediction),
                    'is_fraud': bool(prediction == 1),
                    'fraud_probability': fraud_probability,
                    'not_fraud_probability': 1 - fraud_probability,
                    'confidence': max(fraud_probability, 1 - fraud_probability)
                }
                
                analytics_tracker.log_prediction(result, transaction)
                results.append(result)
            except Exception as e:
                results.append({'index': i, 'error': str(e)})
        
        logger.info(f"Batch prediction completed: {len(results)} transactions")
        return jsonify({'success': True, 'total': len(transactions), 'results': results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        return jsonify({'error': f'Batch prediction error: {str(e)}'}), 500


@app.route('/api/analytics', methods=['GET'])
@require_api_key
def get_analytics():
    """API endpoint to get analytics dashboard data."""
    try:
        stats = analytics_tracker.get_dashboard_stats()
        return jsonify({'success': True, 'analytics': stats})
    except Exception as e:
        logger.error(f"Analytics error: {e}", exc_info=True)
        return jsonify({'error': f'Analytics error: {str(e)}'}), 500


@app.route('/api/models', methods=['GET'])
@require_api_key
def list_models():
    """API endpoint to list all available models and their performance."""
    global model_metrics, all_models
    
    try:
        models_info = {'current_model': model_name, 'available_models': []}
        
        if model_metrics and 'models' in model_metrics:
            for name, metrics in model_metrics['models'].items():
                models_info['available_models'].append({
                    'name': name,
                    'roc_auc': metrics.get('roc_auc', 0),
                    'avg_precision': metrics.get('avg_precision', 0),
                    'is_active': name == model_name
                })
        
        return jsonify({'success': True, 'models': models_info})
    except Exception as e:
        return jsonify({'error': f'Error getting models: {str(e)}'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Credit Card Fraud Detection Web Application")
    print("=" * 60)
    print("\nInitializing...")
    
    # Initialize API key system - generate default key if none exists
    from api_key_manager import load_api_keys
    keys = load_api_keys()
    if not keys:
        print("\n⚠️  No API keys found. Generating default API key...")
        default_key = generate_api_key(name="Default Web Key")
        print(f"\n✓ Default API key generated!")
        print(f"  You can use this key in the web interface.")
        print(f"  Or generate a new one via the web interface.\n")
    
    # Load or train model on startup
    load_or_train_model()
    
    print("\n" + "=" * 60)
    print("Server starting...")
    print("Open your browser and go to: http://localhost:5000")
    print("=" * 60 + "\n")
    
    # Run the Flask app
    # Try port 5000, if busy try 5001
    import socket
    port = 5000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    if result != 0:
        port = 5000  # Port is free
    else:
        port = 5001  # Port is in use, use alternative
        print(f"Port 5000 is busy, using port {port} instead")
    
    app.run(debug=True, host='0.0.0.0', port=port)

