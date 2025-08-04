#!/usr/bin/env python3
"""
Gemini API Installation Script for AI Task Management System
"""

import subprocess
import sys
import os

def install_gemini():
    """Install Gemini API requirements"""
    print("🚀 Installing Gemini API for AI Task Management System")
    print("=" * 60)
    
    try:
        # Install google-generativeai package
        print("📦 Installing google-generativeai...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai>=0.3.0"])
        print("✅ google-generativeai installed successfully!")
        
        # Test import
        print("🧪 Testing Gemini API import...")
        import google.generativeai as genai
        print("✅ Gemini API import successful!")
        
        print("\n🎉 Gemini API installation completed!")
        print("\n📋 Next steps:")
        print("1. Get your API key from: https://makersuite.google.com/app/apikey")
        print("2. Open the app and go to Settings > AI Configuration")
        print("3. Enable Gemini API and enter your API key")
        print("4. Test the natural language features!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    install_gemini() 