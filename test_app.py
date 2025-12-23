#!/usr/bin/env python
"""
Quick test script to verify the app is working
"""
import requests
import time
import sys

def test_app():
    """Test if the Flask app is running and responding"""
    ports = [5000, 5001]
    
    for port in ports:
        try:
            url = f"http://localhost:{port}"
            print(f"Testing {url}...")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"✓ App is running on port {port}!")
                print(f"✓ Response length: {len(response.text)} characters")
                if "Credit Card Fraud Detection" in response.text:
                    print("✓ HTML content looks correct!")
                print(f"\nOpen your browser and go to: {url}")
                return True
        except requests.exceptions.ConnectionError:
            print(f"✗ Port {port} not responding")
            continue
        except Exception as e:
            print(f"✗ Error testing port {port}: {e}")
            continue
    
    print("\n✗ App is not running on any port")
    print("Please run: python app.py")
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Credit Card Fraud Detection App")
    print("=" * 60)
    print()
    
    # Wait a bit for app to start if it's starting
    print("Waiting 3 seconds for app to start...")
    time.sleep(3)
    
    success = test_app()
    sys.exit(0 if success else 1)

