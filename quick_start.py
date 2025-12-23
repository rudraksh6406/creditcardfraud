#!/usr/bin/env python
"""
Quick Start Script - Runs the app with minimal setup
"""
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify
from app import app

if __name__ == '__main__':
    print("=" * 60)
    print("Credit Card Fraud Detection - Quick Start")
    print("=" * 60)
    print("\nStarting server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except OSError as e:
        if "Address already in use" in str(e):
            print("\nPort 5000 is busy. Trying port 5001...")
            app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
        else:
            raise

