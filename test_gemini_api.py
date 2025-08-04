#!/usr/bin/env python3
"""
Test script to verify Gemini API integration
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

print("🔍 Debugging API Key Loading...")

# Test direct import
try:
    import config
    print(f"✅ Config module imported")
    print(f"🔑 Direct API Key: {config.GEMINI_API_KEY[:20] if config.GEMINI_API_KEY else 'None'}...")
    print(f"🚀 Direct ENABLE_GEMINI: {config.ENABLE_GEMINI}")
except Exception as e:
    print(f"❌ Error importing config: {e}")

# Test environment variable
api_key_from_env = os.getenv("GEMINI_API_KEY", "AIzaSyARWvjHmmn-2M005CAU3bbNV5Qfb4Q74zA")
print(f"🔑 API Key from env: {api_key_from_env[:20]}...")

# Test Gemini API import
try:
    import google.generativeai as genai
    print("✅ google-generativeai imported successfully")
    
    # Test API key configuration
    if api_key_from_env:
        genai.configure(api_key=api_key_from_env)
        print("✅ API key configured successfully")
        
        # Test model initialization
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Model initialized successfully")
            
            # Test simple query
            try:
                response = model.generate_content("Hello! Can you respond with 'Gemini API is working!' if you can see this message?")
                if response.text:
                    print(f"✅ Gemini API Test Response: {response.text}")
                    print("🎉 Gemini API is working perfectly!")
                else:
                    print("❌ No response from Gemini API")
                    
            except Exception as e:
                print(f"❌ Error testing Gemini API: {e}")
                
        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            
    else:
        print("❌ No API key provided")
        
except ImportError:
    print("❌ google-generativeai not installed. Run: pip install google-generativeai")

print("\n" + "="*50)
print("🔧 Next Steps:")
print("1. Open the Streamlit app at http://localhost:8518")
print("2. Go to Settings > AI Configuration")
print("3. Test Advanced AI features in Task Analysis")
print("4. Try Natural Language queries")
print("="*50) 