#!/bin/bash

echo "Setting up Disease Prediction App..."

# Detect Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Using $PYTHON_VERSION"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip tools
echo "⬆Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing requirements..."
pip install -r requirements.txt

# Done
echo "Setup complete!"
echo "Run: source venv/bin/activate && streamlit run app.py"