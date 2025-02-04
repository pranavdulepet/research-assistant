import subprocess
import sys
import os
from pathlib import Path

def setup_environment():
    print("Setting up Research Paper Assistant...")
    
    # Create virtual environment
    subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Determine the pip path
    pip_path = "venv/Scripts/pip" if os.name == 'nt' else "venv/bin/pip"
    
    # Install requirements
    subprocess.run([pip_path, "install", "-r", "requirements.txt"])
    
    # Create necessary directories
    Path("app/static/figures").mkdir(parents=True, exist_ok=True)
    Path("app/static/uploads").mkdir(parents=True, exist_ok=True)
    
    # Create .env file if it doesn't exist
    if not Path(".env").exists():
        Path(".env").write_text(Path(".env").read_text())
        print("\nPlease edit .env file with your API keys!")

if __name__ == "__main__":
    setup_environment()
    print("\nSetup complete! Run 'python run.py' to start the application.")
