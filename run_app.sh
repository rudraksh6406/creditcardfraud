#!/bin/bash
# Simple script to run the Flask app

cd /Users/rudrakshdubey/creditcardfraud

echo "=========================================="
echo "Starting Credit Card Fraud Detection App"
echo "=========================================="
echo ""

# Kill any existing processes on ports 5000/5001
echo "Clearing ports..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true
lsof -ti:5001 | xargs kill -9 2>/dev/null || true
sleep 1

# Make sure we're not in venv
deactivate 2>/dev/null || true

# Run the app
echo "Starting Flask app..."
echo ""
python app.py

