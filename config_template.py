# Configuration Template for AI Task Management System
# Copy this file to config.py and add your API keys

import os

# Load from environment variables if available (recommended for production)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Add your Gemini API key here

# Database Configuration
DATABASE_PATH = "ai_task_management.db"

# App Configuration
APP_TITLE = "AI Task Management System"
APP_ICON = "🤖"

# Feature Flags
ENABLE_GEMINI = True
ENABLE_ADVANCED_AI = True
ENABLE_TEAM_CHAT = True
ENABLE_GANTT_CHARTS = True

# UI Configuration
THEME = "default"  # Options: "default", "dark", "light"
SIDEBAR_STYLE = "compact"  # Options: "compact", "expanded"

# Security Configuration
ENCRYPT_API_KEYS = False  # Set to True for production

# Instructions:
# 1. Copy this file to config.py
# 2. Add your Gemini API key from: https://makersuite.google.com/app/apikey
# 3. For production, use environment variables instead of hardcoding
# 4. Never commit config.py with real API keys to version control 