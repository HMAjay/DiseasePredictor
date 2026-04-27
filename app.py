import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    APP_PATH = BASE_DIR / "app" / "app.py"
    
    print(f"Starting HealthAI Streamlit server...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_PATH)], check=True)
    except KeyboardInterrupt:
        print("\nStopping HealthAI...")
    except subprocess.CalledProcessError as e:
        print(f"Error starting Streamlit: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


