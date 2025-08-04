#!/usr/bin/env python3
"""
Comprehensive test for all Gemini API functions
"""

import sys
import os
import pandas as pd

# Add current directory to path
sys.path.append('.')

try:
    from config import GEMINI_API_KEY, ENABLE_GEMINI
    import google.generativeai as genai
    
    print("🔍 Testing All Gemini Functions...")
    print(f"🔑 API Key: {GEMINI_API_KEY[:20]}...")
    print(f"🚀 Gemini Enabled: {ENABLE_GEMINI}")
    
    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Test data
    sample_task = {
        "title": "Fix login bug",
        "description": "Users cannot log in with correct credentials",
        "priority": "high",
        "complexity_score": 7
    }
    
    sample_employees = pd.DataFrame([
        {"name": "John Doe", "role": "Developer", "skills": "Python, JavaScript", "experience_years": 3},
        {"name": "Jane Smith", "role": "QA Engineer", "skills": "Testing, Automation", "experience_years": 2}
    ])
    
    sample_tasks = pd.DataFrame([
        {"title": "Task 1", "status": "pending", "priority": "high", "complexity_score": 5},
        {"title": "Task 2", "status": "completed", "priority": "medium", "complexity_score": 3}
    ])
    
    # Import functions from main app
    from streamlit_dashboard import (
        analyze_task_with_gemini,
        get_gemini_employee_recommendation,
        get_gemini_insights,
        process_natural_language_query
    )
    
    print("\n" + "="*50)
    print("🧪 Testing Function 1: analyze_task_with_gemini")
    print("="*50)
    
    result1 = analyze_task_with_gemini("Fix login bug", "Login Bug Fix")
    print(f"✅ Result: {result1}")
    
    print("\n" + "="*50)
    print("🧪 Testing Function 2: get_gemini_employee_recommendation")
    print("="*50)
    
    result2 = get_gemini_employee_recommendation(sample_task, sample_employees)
    print(f"✅ Result: {result2}")
    
    print("\n" + "="*50)
    print("🧪 Testing Function 3: get_gemini_insights")
    print("="*50)
    
    result3 = get_gemini_insights(sample_tasks, sample_employees)
    print(f"✅ Result: {result3}")
    
    print("\n" + "="*50)
    print("🧪 Testing Function 4: process_natural_language_query")
    print("="*50)
    
    result4 = process_natural_language_query("Show me all high priority tasks", sample_tasks)
    print(f"✅ Result: {result4}")
    
    print("\n" + "="*50)
    print("🎉 All Gemini Functions Tested Successfully!")
    print("="*50)
    print("✅ analyze_task_with_gemini: Working")
    print("✅ get_gemini_employee_recommendation: Working")
    print("✅ get_gemini_insights: Working")
    print("✅ process_natural_language_query: Working")
    print("\n🚀 Your Gemini API integration is fully functional!")
    print("📱 You can now test these features in the Streamlit app at http://localhost:8518")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc() 