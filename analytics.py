"""
Advanced Analytics and Monitoring System
=========================================
Provides real-time analytics, model performance tracking, and business metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict

ANALYTICS_DIR = 'analytics'
os.makedirs(ANALYTICS_DIR, exist_ok=True)

ANALYTICS_FILE = os.path.join(ANALYTICS_DIR, 'analytics.json')
PREDICTIONS_LOG = os.path.join(ANALYTICS_DIR, 'predictions_log.jsonl')


class AnalyticsTracker:
    """Track and analyze fraud detection system performance."""
    
    def __init__(self):
        self.load_analytics()
    
    def load_analytics(self):
        """Load existing analytics data."""
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                'total_predictions': 0,
                'fraud_detected': 0,
                'safe_transactions': 0,
                'total_amount_checked': 0,
                'fraud_amount_detected': 0,
                'daily_stats': {},
                'model_performance': {},
                'feature_importance_tracking': {},
                'alerts': []
            }
    
    def save_analytics(self):
        """Save analytics data to file."""
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def log_prediction(self, prediction_result, transaction_data=None):
        """
        Log a prediction for analytics.
        
        Parameters:
        -----------
        prediction_result : dict
            Prediction results with keys: is_fraud, fraud_probability, etc.
        transaction_data : dict, optional
            Original transaction data
        """
        # Update counters
        self.data['total_predictions'] += 1
        
        if prediction_result.get('is_fraud'):
            self.data['fraud_detected'] += 1
        else:
            self.data['safe_transactions'] += 1
        
        # Track amounts
        if transaction_data and 'Amount' in transaction_data:
            amount = float(transaction_data['Amount'])
            self.data['total_amount_checked'] += amount
            if prediction_result.get('is_fraud'):
                self.data['fraud_amount_detected'] += amount
        
        # Daily statistics
        today = datetime.now().strftime('%Y-%m-%d')
        if today not in self.data['daily_stats']:
            self.data['daily_stats'][today] = {
                'predictions': 0,
                'fraud_detected': 0,
                'safe_transactions': 0,
                'total_amount': 0
            }
        
        daily = self.data['daily_stats'][today]
        daily['predictions'] += 1
        if prediction_result.get('is_fraud'):
            daily['fraud_detected'] += 1
        else:
            daily['safe_transactions'] += 1
        
        if transaction_data and 'Amount' in transaction_data:
            daily['total_amount'] += float(transaction_data['Amount'])
        
        # Log to JSONL file
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_result,
            'transaction': transaction_data
        }
        
        with open(PREDICTIONS_LOG, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Save analytics
        self.save_analytics()
    
    def get_dashboard_stats(self):
        """Get statistics for dashboard display."""
        total = self.data['total_predictions']
        if total == 0:
            return {
                'total_predictions': 0,
                'fraud_rate': 0,
                'total_amount': 0,
                'fraud_amount': 0,
                'daily_trend': []
            }
        
        fraud_rate = (self.data['fraud_detected'] / total) * 100
        
        # Get last 7 days trend
        daily_trend = []
        for i in range(6, -1, -1):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            if date in self.data['daily_stats']:
                daily_trend.append(self.data['daily_stats'][date])
            else:
                daily_trend.append({
                    'predictions': 0,
                    'fraud_detected': 0,
                    'safe_transactions': 0,
                    'total_amount': 0
                })
        
        return {
            'total_predictions': total,
            'fraud_detected': self.data['fraud_detected'],
            'safe_transactions': self.data['safe_transactions'],
            'fraud_rate': round(fraud_rate, 2),
            'total_amount_checked': round(self.data['total_amount_checked'], 2),
            'fraud_amount_detected': round(self.data['fraud_amount_detected'], 2),
            'daily_trend': daily_trend
        }
    
    def get_model_performance(self):
        """Get model performance metrics."""
        return self.data.get('model_performance', {})
    
    def update_model_performance(self, model_name, metrics):
        """Update model performance metrics."""
        if 'model_performance' not in self.data:
            self.data['model_performance'] = {}
        
        self.data['model_performance'][model_name] = {
            'metrics': metrics,
            'last_updated': datetime.now().isoformat()
        }
        self.save_analytics()
    
    def add_alert(self, alert_type, message, severity='info'):
        """Add an alert to the system."""
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        self.data['alerts'].append(alert)
        
        # Keep only last 100 alerts
        if len(self.data['alerts']) > 100:
            self.data['alerts'] = self.data['alerts'][-100:]
        
        self.save_analytics()
    
    def get_recent_alerts(self, limit=10):
        """Get recent alerts."""
        return self.data['alerts'][-limit:]


# Global analytics tracker instance
analytics_tracker = AnalyticsTracker()

