#!/usr/bin/env python3
"""
Setup script for Search Engine Project
This script helps set up the virtual environment and install dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Search Engine Project Environment")
    print("=" * 50)
    
    # Check if Python is available
    if not run_command("python --version", "Checking Python version"):
        print("❌ Python is not installed or not in PATH")
        return
    
    # Create virtual environment
    if not run_command("python -m venv .venv", "Creating virtual environment"):
        return
    
    # Activate virtual environment and install requirements
    if os.name == 'nt':  # Windows
        activate_cmd = ".venv\\Scripts\\activate"
        pip_cmd = ".venv\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        activate_cmd = "source .venv/bin/activate"
        pip_cmd = ".venv/bin/pip"
    
    # Install requirements
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing project dependencies"):
        return
    
    print("\n🎉 Environment setup complete!")
    print("\n📋 Next steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    print("2. Start Jupyter notebook:")
    print("   jupyter notebook Part1_Data_Preparation_Embedding.ipynb")
    print("\n💡 To deactivate the virtual environment later, just run: deactivate")

if __name__ == "__main__":
    main()
