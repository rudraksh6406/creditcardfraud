"""
API Key Management System
=========================
Handles generation, validation, and storage of API keys for the fraud detection API.
"""

import os
import json
import secrets
import hashlib
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify

# Path to store API keys
API_KEYS_FILE = 'api_keys.json'
API_KEYS_DIR = 'config'

# Create config directory if it doesn't exist
os.makedirs(API_KEYS_DIR, exist_ok=True)

API_KEYS_PATH = os.path.join(API_KEYS_DIR, API_KEYS_FILE)


def generate_api_key(name="default", expires_days=None):
    """
    Generate a new API key.
    
    Parameters:
    -----------
    name : str
        Name/description for the API key
    expires_days : int, optional
        Number of days until the key expires
        
    Returns:
    --------
    api_key : str
        The generated API key (store this securely!)
    """
    # Generate a secure random API key
    api_key = f"fraud_api_{secrets.token_urlsafe(32)}"
    
    # Hash the key for storage
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Load existing keys
    keys = load_api_keys()
    
    # Create key entry
    key_entry = {
        'name': name,
        'key_hash': key_hash,
        'created_at': datetime.now().isoformat(),
        'last_used': None,
        'usage_count': 0,
        'active': True
    }
    
    if expires_days:
        expiry_date = datetime.now() + timedelta(days=expires_days)
        key_entry['expires_at'] = expiry_date.isoformat()
    
    # Store the key
    keys[key_hash] = key_entry
    save_api_keys(keys)
    
    print(f"\n{'='*60}")
    print("API KEY GENERATED")
    print(f"{'='*60}")
    print(f"Name: {name}")
    print(f"API Key: {api_key}")
    print(f"\n⚠️  IMPORTANT: Save this key securely!")
    print(f"   You won't be able to see it again.")
    print(f"{'='*60}\n")
    
    return api_key


def load_api_keys():
    """Load API keys from file."""
    if os.path.exists(API_KEYS_PATH):
        try:
            with open(API_KEYS_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading API keys: {e}")
            return {}
    return {}


def save_api_keys(keys):
    """Save API keys to file."""
    try:
        with open(API_KEYS_PATH, 'w') as f:
            json.dump(keys, f, indent=2)
    except Exception as e:
        print(f"Error saving API keys: {e}")


def validate_api_key(api_key):
    """
    Validate an API key.
    
    Parameters:
    -----------
    api_key : str
        The API key to validate
        
    Returns:
    --------
    valid : bool
        True if key is valid, False otherwise
    key_info : dict
        Information about the key (if valid)
    """
    if not api_key:
        return False, None
    
    # Hash the provided key
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Load keys
    keys = load_api_keys()
    
    # Check if key exists
    if key_hash not in keys:
        return False, None
    
    key_info = keys[key_hash]
    
    # Check if key is active
    if not key_info.get('active', True):
        return False, key_info
    
    # Check if key has expired
    if 'expires_at' in key_info:
        expires_at = datetime.fromisoformat(key_info['expires_at'])
        if datetime.now() > expires_at:
            return False, key_info
    
    # Update usage statistics
    key_info['last_used'] = datetime.now().isoformat()
    key_info['usage_count'] = key_info.get('usage_count', 0) + 1
    keys[key_hash] = key_info
    save_api_keys(keys)
    
    return True, key_info


def require_api_key(f):
    """
    Decorator to require API key for an endpoint.
    
    Usage:
    ------
    @app.route('/api/endpoint')
    @require_api_key
    def my_endpoint():
        return jsonify({'message': 'Success'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from header or query parameter
        api_key = None
        
        # Check Authorization header (Bearer token)
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            api_key = auth_header.split(' ')[1]
        
        # Check X-API-Key header
        if not api_key:
            api_key = request.headers.get('X-API-Key')
        
        # Check query parameter
        if not api_key:
            api_key = request.args.get('api_key')
        
        # Check JSON body
        if not api_key:
            try:
                data = request.get_json()
                if data and 'api_key' in data:
                    api_key = data['api_key']
            except:
                pass
        
        # Validate API key
        valid, key_info = validate_api_key(api_key)
        
        if not valid:
            return jsonify({
                'error': 'Invalid or missing API key',
                'message': 'Please provide a valid API key in the Authorization header, X-API-Key header, or as a query parameter'
            }), 401
        
        # Add key info to request context
        request.api_key_info = key_info
        
        return f(*args, **kwargs)
    
    return decorated_function


def list_api_keys():
    """List all API keys (without exposing the actual keys)."""
    keys = load_api_keys()
    
    key_list = []
    for key_hash, key_info in keys.items():
        key_list.append({
            'name': key_info.get('name', 'Unknown'),
            'created_at': key_info.get('created_at'),
            'last_used': key_info.get('last_used'),
            'usage_count': key_info.get('usage_count', 0),
            'active': key_info.get('active', True),
            'expires_at': key_info.get('expires_at')
        })
    
    return key_list


def revoke_api_key(api_key):
    """Revoke (deactivate) an API key."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    keys = load_api_keys()
    
    if key_hash in keys:
        keys[key_hash]['active'] = False
        save_api_keys(keys)
        return True
    
    return False


if __name__ == "__main__":
    # Generate a default API key if none exists
    keys = load_api_keys()
    if not keys:
        print("No API keys found. Generating default API key...")
        api_key = generate_api_key(name="Default Key")
        print(f"\nYour API key: {api_key}")
        print("\nUse this key to authenticate API requests.")
    else:
        print("Existing API keys found.")
        print("\nTo generate a new key, use:")
        print("  from api_key_manager import generate_api_key")
        print("  generate_api_key('My Key Name')")

