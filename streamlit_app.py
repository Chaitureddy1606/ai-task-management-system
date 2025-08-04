#!/usr/bin/env python3
"""
AI Task Management System - Main Entry Point for Streamlit Cloud
This file serves as the main entry point for Streamlit Cloud deployment
"""

import streamlit as st
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main dashboard
from streamlit_dashboard import main

if __name__ == "__main__":
    main() 