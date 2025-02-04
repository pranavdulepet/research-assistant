import subprocess
import sys
import os
from pathlib import Path

def setup_environment():
    print("Setting up Research Paper Assistant...")
    
    # Create virtual environment
    subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Determine the activation script and pip path based on OS
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate.bat"
        pip_path = "pip"
    else:  # Unix/Linux/Mac
        activate_script = "venv/bin/activate"
        pip_path = "pip"
    
    # Activate venv and install requirements
    activate_command = f"source {activate_script}" if os.name != 'nt' else activate_script
    install_command = f"{pip_path} install -r requirements.txt"
    
    if os.name == 'nt':
        subprocess.run(f"{activate_command} && {install_command}", shell=True)
    else:
        subprocess.run(['bash', '-c', f'{activate_command} && {install_command}'])
    
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
