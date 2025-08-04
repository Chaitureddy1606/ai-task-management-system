# Configuration file for AI Task Management System
# Store your API keys and configuration settings here

import os

# Load from environment variables if available
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyARWvjHmmn-2M005CAU3bbNV5Qfb4Q74zA")  # Add your Gemini API key here

# Ensure API key is not empty
if not GEMINI_API_KEY:
    GEMINI_API_KEY = "AIzaSyARWvjHmmn-2M005CAU3bbNV5Qfb4Q74zA"

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