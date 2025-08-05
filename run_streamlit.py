#!/usr/bin/env python3
"""
Streamlit App Launcher for Crop Recommendation System

This script launches the Streamlit web application.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app"""
    print("🌾 Starting Crop Recommendation System...")
    print("📱 Opening web interface...")
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        print("✅ Streamlit installed successfully")
    
    # Launch the app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

if __name__ == "__main__":
    main() 