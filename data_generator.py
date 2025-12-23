"""
Advanced Synthetic Credit Card Transaction Data Generator
==========================================================
Generates realistic synthetic credit card transaction data with comprehensive features
including amount, time, location, merchant category, and fraud labels.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from typing import Optional, Tuple

# Merchant Category Codes (MCC) - Common categories
MERCHANT_CATEGORIES = {
    'Grocery Stores': {'mcc': 5411, 'fraud_rate': 0.001},
    'Gas Stations': {'mcc': 5542, 'fraud_rate': 0.002},
    'Restaurants': {'mcc': 5812, 'fraud_rate': 0.0015},
    'Online Retail': {'mcc': 5999, 'fraud_rate': 0.005},
    'Electronics': {'mcc': 5732, 'fraud_rate': 0.003},
    'Travel': {'mcc': 4722, 'fraud_rate': 0.004},
    'Hotels': {'mcc': 7011, 'fraud_rate': 0.003},
    'Entertainment': {'mcc': 7922, 'fraud_rate': 0.002},
    'Healthcare': {'mcc': 8011, 'fraud_rate': 0.001},
    'Education': {'mcc': 8211, 'fraud_rate': 0.0005},
    'ATM Withdrawal': {'mcc': 6011, 'fraud_rate': 0.006},
    'Cash Advance': {'mcc': 6012, 'fraud_rate': 0.008},
    'Jewelry': {'mcc': 5944, 'fraud_rate': 0.004},
    'Department Stores': {'mcc': 5311, 'fraud_rate': 0.002},
    'Clothing': {'mcc': 5651, 'fraud_rate': 0.0025},
}

# US Cities with coordinates (sample)
CITIES = [
    {'city': 'New York', 'state': 'NY', 'lat': 40.7128, 'lon': -74.0060, 'timezone': 'America/New_York'},
    {'city': 'Los Angeles', 'state': 'CA', 'lat': 34.0522, 'lon': -118.2437, 'timezone': 'America/Los_Angeles'},
    {'city': 'Chicago', 'state': 'IL', 'lat': 41.8781, 'lon': -87.6298, 'timezone': 'America/Chicago'},
    {'city': 'Houston', 'state': 'TX', 'lat': 29.7604, 'lon': -95.3698, 'timezone': 'America/Chicago'},
    {'city': 'Phoenix', 'state': 'AZ', 'lat': 33.4484, 'lon': -112.0740, 'timezone': 'America/Phoenix'},
    {'city': 'Philadelphia', 'state': 'PA', 'lat': 39.9526, 'lon': -75.1652, 'timezone': 'America/New_York'},
    {'city': 'San Antonio', 'state': 'TX', 'lat': 29.4241, 'lon': -98.4936, 'timezone': 'America/Chicago'},
    {'city': 'San Diego', 'state': 'CA', 'lat': 32.7157, 'lon': -117.1611, 'timezone': 'America/Los_Angeles'},
    {'city': 'Dallas', 'state': 'TX', 'lat': 32.7767, 'lon': -96.7970, 'timezone': 'America/Chicago'},
    {'city': 'San Jose', 'state': 'CA', 'lat': 37.3382, 'lon': -121.8863, 'timezone': 'America/Los_Angeles'},
    {'city': 'Austin', 'state': 'TX', 'lat': 30.2672, 'lon': -97.7431, 'timezone': 'America/Chicago'},
    {'city': 'Jacksonville', 'state': 'FL', 'lat': 30.3322, 'lon': -81.6557, 'timezone': 'America/New_York'},
    {'city': 'San Francisco', 'state': 'CA', 'lat': 37.7749, 'lon': -122.4194, 'timezone': 'America/Los_Angeles'},
    {'city': 'Columbus', 'state': 'OH', 'lat': 39.9612, 'lon': -82.9988, 'timezone': 'America/New_York'},
    {'city': 'Fort Worth', 'state': 'TX', 'lat': 32.7555, 'lon': -97.3308, 'timezone': 'America/Chicago'},
]

# Transaction types
TRANSACTION_TYPES = ['Purchase', 'Withdrawal', 'Transfer', 'Payment', 'Refund']


def generate_synthetic_transactions(
    n_samples: int = 100000,
    fraud_rate: float = 0.0017,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate comprehensive synthetic credit card transaction data.
    
    Parameters:
    -----------
    n_samples : int
        Number of transactions to generate
    fraud_rate : float
        Proportion of fraudulent transactions (default: 0.17%)
    start_date : datetime, optional
        Start date for transactions
    end_date : datetime, optional
        End date for transactions
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with synthetic transaction data
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Set date range
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    date_range = (end_date - start_date).total_seconds()
    
    # Generate base features
    data = []
    
    # Determine fraud cases
    n_fraud = int(n_samples * fraud_rate)
    fraud_indices = set(np.random.choice(n_samples, n_fraud, replace=False))
    
    print(f"Generating {n_samples:,} transactions ({n_fraud:,} fraudulent)...")
    
    for i in range(n_samples):
        is_fraud = i in fraud_indices
        
        # Transaction timestamp
        transaction_time = start_date + timedelta(seconds=np.random.uniform(0, date_range))
        
        # Merchant category
        if is_fraud:
            # Fraud transactions more likely in high-risk categories
            high_risk_categories = ['Online Retail', 'ATM Withdrawal', 'Cash Advance', 
                                  'Travel', 'Jewelry']
            category = random.choice(high_risk_categories)
        else:
            category = random.choice(list(MERCHANT_CATEGORIES.keys()))
        
        mcc_info = MERCHANT_CATEGORIES[category]
        
        # Transaction amount
        if is_fraud:
            # Fraud transactions tend to be larger or very small (testing)
            if np.random.random() < 0.7:
                amount = np.random.lognormal(mean=5.5, sigma=1.2)  # Larger amounts
            else:
                amount = np.random.uniform(0.01, 10)  # Small test transactions
        else:
            # Normal transactions follow log-normal distribution
            amount = np.random.lognormal(mean=4.0, sigma=1.0)
        
        amount = round(amount, 2)
        
        # Location
        city_info = random.choice(CITIES)
        
        # Add location noise for fraud (unusual locations)
        if is_fraud and np.random.random() < 0.6:
            # Fraud might occur in different location than usual
            lat = city_info['lat'] + np.random.normal(0, 0.5)
            lon = city_info['lon'] + np.random.normal(0, 0.5)
        else:
            lat = city_info['lat'] + np.random.normal(0, 0.1)
            lon = city_info['lon'] + np.random.normal(0, 0.1)
        
        # Transaction type
        if is_fraud:
            transaction_type = random.choice(['Purchase', 'Withdrawal', 'Transfer'])
        else:
            transaction_type = random.choice(TRANSACTION_TYPES)
        
        # Time-based features
        hour = transaction_time.hour
        day_of_week = transaction_time.weekday()
        day_of_month = transaction_time.day
        
        # Fraud more likely at unusual times
        if is_fraud:
            if np.random.random() < 0.4:
                hour = np.random.choice([0, 1, 2, 3, 4, 5, 22, 23])  # Late night/early morning
        
        # Customer ID (simulate multiple customers)
        customer_id = f"CUST_{np.random.randint(1000, 9999)}"
        
        # Card number (masked)
        card_last_4 = np.random.randint(1000, 9999)
        
        # Transaction ID
        transaction_id = f"TXN_{i+1:08d}"
        
        # Generate PCA-like features (V1-V28) - these would normally come from feature engineering
        # For synthetic data, we'll generate them with correlations to fraud
        v_features = {}
        for j in range(1, 29):
            if is_fraud:
                # Fraud transactions have different patterns
                v_features[f'V{j}'] = np.random.normal(
                    loc=np.random.choice([-2, 2]),
                    scale=1.5
                )
            else:
                v_features[f'V{j}'] = np.random.normal(loc=0, scale=1)
        
        # Additional derived features
        # Time since last transaction (simulated)
        time_since_last = np.random.exponential(scale=3600)  # seconds
        
        # Distance from home (simulated - fraud might be far from home)
        if is_fraud:
            distance_from_home = np.random.exponential(scale=500)  # km
        else:
            distance_from_home = np.random.exponential(scale=50)  # km
        
        # Number of transactions in last 24 hours
        if is_fraud:
            transactions_last_24h = np.random.poisson(lam=8)  # More transactions
        else:
            transactions_last_24h = np.random.poisson(lam=2)
        
        # Build transaction record
        transaction = {
            'TransactionID': transaction_id,
            'CustomerID': customer_id,
            'CardNumber': f"****-****-****-{card_last_4}",
            'Timestamp': transaction_time,
            'Time': int((transaction_time - start_date).total_seconds()),
            'Amount': amount,
            'MerchantCategory': category,
            'MCC': mcc_info['mcc'],
            'TransactionType': transaction_type,
            'City': city_info['city'],
            'State': city_info['state'],
            'Latitude': round(lat, 6),
            'Longitude': round(lon, 6),
            'Hour': hour,
            'DayOfWeek': day_of_week,
            'DayOfMonth': day_of_month,
            'TimeSinceLastTransaction': round(time_since_last, 2),
            'DistanceFromHome': round(distance_from_home, 2),
            'TransactionsLast24H': transactions_last_24h,
            'Class': 1 if is_fraud else 0,
            **v_features
        }
        
        data.append(transaction)
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1:,} transactions...")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure proper data types
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Class'] = df['Class'].astype(int)
    
    print(f"✓ Generated {len(df):,} transactions")
    print(f"  - Fraudulent: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")
    print(f"  - Legitimate: {(df['Class']==0).sum():,} ({(df['Class']==0).mean()*100:.2f}%)")
    
    return df


def save_dataset(df: pd.DataFrame, filepath: str = 'data/creditcard_transactions.csv'):
    """
    Save the generated dataset to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction dataframe
    filepath : str
        Path to save the CSV file
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"✓ Dataset saved to: {filepath}")
    print(f"  File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")


def load_or_generate_dataset(
    filepath: Optional[str] = None,
    n_samples: int = 100000,
    force_regenerate: bool = False
) -> pd.DataFrame:
    """
    Load existing dataset or generate a new one.
    
    Parameters:
    -----------
    filepath : str, optional
        Path to dataset file
    n_samples : int
        Number of samples if generating new dataset
    force_regenerate : bool
        Force regeneration even if file exists
        
    Returns:
    --------
    df : pd.DataFrame
        Transaction dataframe
    """
    if filepath is None:
        filepath = 'data/creditcard_transactions.csv'
    
    if os.path.exists(filepath) and not force_regenerate:
        print(f"Loading existing dataset from: {filepath}")
        df = pd.read_csv(filepath)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        print(f"✓ Loaded {len(df):,} transactions")
        return df
    else:
        print("Generating new synthetic dataset...")
        df = generate_synthetic_transactions(n_samples=n_samples)
        save_dataset(df, filepath)
        return df


if __name__ == "__main__":
    # Generate and save a sample dataset
    print("=" * 60)
    print("Synthetic Credit Card Transaction Data Generator")
    print("=" * 60)
    print()
    
    df = generate_synthetic_transactions(n_samples=100000)
    
    # Display sample
    print("\nSample transactions:")
    print(df[['TransactionID', 'Amount', 'MerchantCategory', 'City', 'State', 'Class']].head(10))
    
    # Save dataset
    save_dataset(df)
    
    # Statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"\nTotal transactions: {len(df):,}")
    print(f"Fraudulent: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")
    print(f"Legitimate: {(df['Class']==0).sum():,} ({(df['Class']==0).mean()*100:.2f}%)")
    print(f"\nTotal amount: ${df['Amount'].sum():,.2f}")
    print(f"Average amount: ${df['Amount'].mean():.2f}")
    print(f"Fraud amount: ${df[df['Class']==1]['Amount'].sum():,.2f}")
    
    print(f"\nMerchant categories:")
    print(df['MerchantCategory'].value_counts())
    
    print(f"\nTop cities:")
    print(df['City'].value_counts().head(10))

