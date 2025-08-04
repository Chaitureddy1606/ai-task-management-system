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

# Add src directory to path for imports
sys.path.append('src')
try:
    from src.utils import load_employee_profiles, connect_db, create_tasks_table
    from src.preprocessing import TaskDataPreprocessor
    from src.priority_model import TaskPriorityModel
    from src.classifier import TaskClassifier
    from src.task_assigner import IntelligentTaskAssigner
    from src.feature_engineering import TaskFeatureEngineer
except ImportError as e:
    st.error(f"❌ Error importing required modules: {e}")
    st.info("Please ensure all required files are present in the repository")

# Import the main dashboard function
try:
    from streamlit_dashboard import main
    # Run the main application
    if __name__ == "__main__":
        main()
except Exception as e:
    st.error(f"❌ Error running application: {e}")
    st.info("Please check the application logs for more details") 