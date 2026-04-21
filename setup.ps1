Write-Host "Setting up Disease Prediction App..."

# Check Python
python --version

# Create virtual environment
Write-Host "Creating virtual environment..."
python -m venv venv

# Activate venv
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Upgrade pip tools
Write-Host "⬆Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies
Write-Host "Installing requirements..."
pip install -r requirements.txt

Write-Host "Setup complete!"
Write-Host "Run: .\venv\Scripts\Activate.ps1 ; streamlit run app.py"