#!/usr/bin/env python3
"""
AI Task Management System - Main Entry Point for Streamlit Cloud
This file serves as the main entry point for Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import io
import sys
import os
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Any, Optional
import logging
import base64
import time
import joblib
import random
import re
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page config - MUST be the very first Streamlit command
st.set_page_config(
    page_title="AI Task Management System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import configuration
try:
    from config import GEMINI_API_KEY, ENABLE_GEMINI
except ImportError:
    GEMINI_API_KEY = ""
    ENABLE_GEMINI = False

# Gemini API Configuration
GEMINI_CONFIG = {
    "enabled": ENABLE_GEMINI,
    "api_key": GEMINI_API_KEY,
    "model": "gemini-1.5-flash",
    "max_tokens": 1000,
    "temperature": 0.3,
    "features": {
        "task_analysis": True,
        "employee_matching": True,
        "priority_prediction": True,
        "natural_queries": True,
        "insights": True
    }
}

# Try to import Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("⚠️ Gemini API not available. Install with: pip install google-generativeai")

# Initialize Gemini if available and enabled
if GEMINI_AVAILABLE and GEMINI_CONFIG["enabled"] and GEMINI_CONFIG["api_key"]:
    try:
        genai.configure(api_key=GEMINI_CONFIG["api_key"])
        gemini_model = genai.GenerativeModel(GEMINI_CONFIG["model"])
        GEMINI_READY = True
    except Exception as e:
        st.error(f"❌ Gemini API initialization failed: {e}")
        GEMINI_READY = False
else:
    GEMINI_READY = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database initialization function for Streamlit Cloud
def initialize_database_safe():
    """Initialize database with proper error handling for Streamlit Cloud"""
    try:
        # Use a timeout and proper connection handling for Streamlit Cloud
        conn = sqlite3.connect('ai_task_management.db', timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")  # Use WAL mode for better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        conn.execute("PRAGMA cache_size=10000")  # Increase cache size
        conn.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
        cursor = conn.cursor()
        
        # Create basic tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT,
                priority TEXT,
                urgency_score INTEGER,
                complexity_score INTEGER,
                business_impact INTEGER,
                estimated_hours REAL,
                days_until_deadline INTEGER,
                status TEXT DEFAULT 'pending',
                assigned_to TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT,
                skills TEXT,
                expertise_areas TEXT,
                preferred_task_types TEXT,
                current_workload INTEGER,
                max_capacity INTEGER,
                experience_years INTEGER,
                location TEXT,
                availability TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                role TEXT DEFAULT 'user',
                department TEXT DEFAULT 'General',
                is_active BOOLEAN DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"❌ Database initialization failed: {e}")
        return False

# Simple main function for Streamlit Cloud
def main_simple():
    """Simple main function for Streamlit Cloud deployment"""
    
    # Initialize database first
    if not initialize_database_safe():
        st.error("❌ Failed to initialize database. Please refresh the page.")
        return
    
    # Initialize session state
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    
    if not st.session_state.is_authenticated:
        # Show simple login
        st.title("🤖 AI Task Management System")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔐 Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_btn"):
                if username == "admin" and password == "admin123":
                    st.session_state.is_authenticated = True
                    st.session_state.current_user = username
                    st.session_state.user_role = "admin"
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Use admin/admin123")
        
        with col2:
            st.subheader("📝 Quick Start")
            st.info("""
            **Demo Credentials:**
            - Username: `admin`
            - Password: `admin123`
            
            **Features Available:**
            - Task Management
            - Employee Assignment
            - AI-Powered Recommendations
            - Analytics Dashboard
            """)
    else:
        # Show main dashboard
        st.title("🤖 AI Task Management System")
        
        # Sidebar
        with st.sidebar:
            st.markdown("### 👤 User Info")
            st.write(f"**User:** {st.session_state.current_user}")
            st.write(f"**Role:** {st.session_state.user_role}")
            
            if st.button("🚪 Logout"):
                st.session_state.is_authenticated = False
                st.rerun()
        
        # Main content
        st.markdown("### 📊 Dashboard")
        
        # Sample data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tasks", "25", "+5")
        
        with col2:
            st.metric("Completed", "18", "+3")
        
        with col3:
            st.metric("In Progress", "5", "-2")
        
        with col4:
            st.metric("Pending", "2", "+1")
        
        # Sample chart
        st.markdown("### 📈 Task Status Overview")
        data = {
            'Status': ['Completed', 'In Progress', 'Pending', 'Overdue'],
            'Count': [18, 5, 2, 1]
        }
        df = pd.DataFrame(data)
        
        fig = px.pie(df, values='Count', names='Status', title="Task Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample table
        st.markdown("### 📋 Recent Tasks")
        sample_tasks = pd.DataFrame({
            'Task': ['Design UI Mockups', 'Database Optimization', 'API Integration', 'Testing'],
            'Priority': ['High', 'Medium', 'High', 'Low'],
            'Status': ['In Progress', 'Completed', 'Pending', 'Completed'],
            'Assignee': ['John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Wilson']
        })
        st.dataframe(sample_tasks, use_container_width=True)

# Run the application
if __name__ == "__main__":
    main_simple() 