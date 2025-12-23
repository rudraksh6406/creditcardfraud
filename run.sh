#!/bin/bash

# Credit Card Fraud Detection - Web Application Launcher
# This script helps you run the web application easily

echo "=========================================="
echo "Credit Card Fraud Detection System"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
mkdir -p data models templates static

echo ""
echo "Starting web application..."
echo "Open your browser and go to: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Flask app
python app.py

