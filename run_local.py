#!/usr/bin/env python3
"""
Local launcher for the Crop Recommendation System
This script launches the Streamlit app on localhost
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app locally"""
    print("🌾 Starting Crop Recommendation System...")
    print("📱 Opening web interface on localhost...")
    
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
        print("🚀 Starting Streamlit server...")
        print("📋 App will open at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
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