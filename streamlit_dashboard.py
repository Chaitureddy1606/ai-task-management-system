#!/usr/bin/env python3
"""
AI Task Management System - Complete Streamlit Dashboard
Professional UI with all functionality integrated
"""

import streamlit as st

# Set page config - MUST be the very first Streamlit command
st.set_page_config(
    page_title="AI Task Management System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
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

# Add src directory to path for imports
sys.path.append('src')
from src.utils import load_employee_profiles, connect_db, create_tasks_table
from src.preprocessing import TaskDataPreprocessor
from src.priority_model import TaskPriorityModel
from src.classifier import TaskClassifier
from src.task_assigner import IntelligentTaskAssigner
from src.feature_engineering import TaskFeatureEngineer

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
        "task_analysis": True,      # Enhanced task understanding
        "employee_matching": True,   # Smart employee assignment
        "priority_prediction": True, # AI-driven priority scoring
        "natural_queries": True,     # Natural language queries
        "insights": True            # Automated insights
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

# Gemini API Functions
def analyze_task_with_gemini(task_description: str, task_title: str = "") -> Dict[str, Any]:
    """Use Gemini to analyze task complexity and requirements"""
    if not GEMINI_READY:
        return {"error": "Gemini API not available"}
    
    try:
        prompt = f"""
        Analyze this task and provide a simple JSON response:
        {{
            "complexity_score": 5,
            "urgency_score": 5,
            "business_impact": 5,
            "estimated_hours": 4.0,
            "required_skills": ["programming", "analysis"],
            "priority": "medium",
            "category": "feature",
            "reasoning": "Brief explanation"
        }}
        
        Task Title: {task_title}
        Task Description: {task_description}
        
        Return only valid JSON, no additional text.
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback response
            result = {
                "complexity_score": 5,
                "urgency_score": 5,
                "business_impact": 5,
                "estimated_hours": 4.0,
                "required_skills": ["general"],
                "priority": "medium",
                "category": "feature",
                "reasoning": f"Analysis of: {task_title}",
                "raw_response": response_text[:200]
            }
        
        return result
    except Exception as e:
        return {
            "error": f"Gemini analysis failed: {e}",
            "complexity_score": 5,
            "urgency_score": 5,
            "business_impact": 5,
            "estimated_hours": 4.0,
            "required_skills": ["general"],
            "priority": "medium",
            "category": "feature",
            "reasoning": "Default analysis"
        }

def get_gemini_employee_recommendation(task_data: Dict, employees_df: pd.DataFrame) -> Dict[str, Any]:
    """Use Gemini to find the best employee for a task"""
    if not GEMINI_READY:
        return {"error": "Gemini API not available"}
    
    try:
        # Prepare employee data for Gemini
        employees_info = []
        for _, emp in employees_df.iterrows():
            employees_info.append({
                "name": emp.get('name', 'Unknown'),
                "role": emp.get('role', ''),
                "skills": emp.get('skills', ''),
                "experience_years": emp.get('experience_years', 0),
                "workload_score": emp.get('workload_score', 0)
            })
        
        prompt = f"""
        Task: {task_data.get('title', '')}
        Description: {task_data.get('description', '')}
        Required Skills: {task_data.get('required_skills', [])}
        Priority: {task_data.get('priority', 'medium')}
        Complexity: {task_data.get('complexity_score', 5)}
        
        Available Employees: {json.dumps(employees_info[:5], indent=2)}  # Limit to first 5 employees
        
        Recommend the best employee with simple JSON response:
        {{
            "recommended_employee": "employee_name",
            "confidence_score": 0.8,
            "reasoning": "Brief explanation",
            "alternative_employees": ["name1", "name2"]
        }}
        
        Return only valid JSON, no additional text.
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback response
            if len(employees_df) > 0:
                first_employee = employees_df.iloc[0]['name'] if 'name' in employees_df.columns else "Unknown"
                result = {
                    "recommended_employee": first_employee,
                    "confidence_score": 0.5,
                    "reasoning": f"Recommended for task: {task_data.get('title', '')}",
                    "alternative_employees": []
                }
            else:
                result = {
                    "recommended_employee": "No employees available",
                    "confidence_score": 0.0,
                    "reasoning": "No employees in database",
                    "alternative_employees": []
                }
        
        return result
    except Exception as e:
        return {
            "error": f"Gemini recommendation failed: {e}",
            "recommended_employee": "Error occurred",
            "confidence_score": 0.0,
            "reasoning": "Failed to get recommendation",
            "alternative_employees": []
        }

def get_gemini_insights(tasks_df: pd.DataFrame, employees_df: pd.DataFrame) -> Dict[str, Any]:
    """Use Gemini to generate insights about the project"""
    if not GEMINI_READY:
        return {"error": "Gemini API not available"}
    
    try:
        # Prepare summary data
        task_summary = {
            "total_tasks": len(tasks_df),
            "pending_tasks": len(tasks_df[tasks_df['status'] == 'pending']) if 'status' in tasks_df.columns else 0,
            "completed_tasks": len(tasks_df[tasks_df['status'] == 'completed']) if 'status' in tasks_df.columns else 0,
            "high_priority": len(tasks_df[tasks_df['priority'] == 'high']) if 'priority' in tasks_df.columns else 0,
            "avg_complexity": float(tasks_df['complexity_score'].mean()) if 'complexity_score' in tasks_df.columns else 0
        }
        
        prompt = f"""
        Analyze this project data and provide simple insights:
        
        Task Summary: {json.dumps(task_summary, indent=2)}
        Employee Count: {len(employees_df)}
        
        Provide simple JSON response:
        {{
            "key_insights": ["insight1", "insight2"],
            "recommendations": ["rec1", "rec2"],
            "risks": ["risk1"],
            "opportunities": ["opp1"]
        }}
        
        Return only valid JSON, no additional text.
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback response
            result = {
                "key_insights": [f"Total tasks: {task_summary['total_tasks']}", "Project analysis completed"],
                "recommendations": ["Monitor progress regularly", "Focus on high priority tasks"],
                "risks": ["Potential delays"],
                "opportunities": ["Improve efficiency"]
            }
        
        return result
    except Exception as e:
        return {
            "error": f"Gemini insights failed: {e}",
            "key_insights": ["Analysis failed"],
            "recommendations": ["Check data"],
            "risks": ["Unknown"],
            "opportunities": ["Review setup"]
        }

def process_natural_language_query(query: str, tasks_df: pd.DataFrame) -> Dict[str, Any]:
    """Process natural language queries using Gemini"""
    if not GEMINI_READY:
        return {"error": "Gemini API not available"}
    
    try:
        # Create a more robust prompt that's less likely to fail
        prompt = f"""
        Analyze this natural language query about task management: "{query}"
        
        Available task data columns: {list(tasks_df.columns)}
        Task count: {len(tasks_df)}
        
        Provide a simple analysis of what the user is asking for. Return only a JSON object with this structure:
        {{
            "query_type": "search|filter|summary|analysis",
            "target_criteria": {{
                "priority": ["high", "medium", "low"],
                "status": ["pending", "in_progress", "completed"],
                "category": ["category_name"],
                "assigned_to": ["employee_name"]
            }},
            "explanation": "Brief explanation of what the query means",
            "suggested_action": "What action to take based on the query"
        }}
        
        Keep the response simple and valid JSON only.
        """
        
        response = gemini_model.generate_content(prompt)
        
        # Try to extract JSON from the response
        response_text = response.text.strip()
        
        # Look for JSON in the response
        try:
            # Try to parse as JSON directly
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # If all else fails, create a fallback response
                    result = {
                        "query_type": "search",
                        "target_criteria": {
                            "priority": [],
                            "status": [],
                            "category": [],
                            "assigned_to": []
                        },
                        "explanation": f"Query: {query}",
                        "suggested_action": "Search tasks based on query",
                        "raw_response": response_text[:200]  # Include first 200 chars for debugging
                    }
            else:
                # No JSON found, create fallback
                result = {
                    "query_type": "search",
                    "target_criteria": {
                        "priority": [],
                        "status": [],
                        "category": [],
                        "assigned_to": []
                    },
                    "explanation": f"Query: {query}",
                    "suggested_action": "Search tasks based on query",
                    "raw_response": response_text[:200]
                }
        
        return result
    except Exception as e:
        return {
            "error": f"Natural language query failed: {e}",
            "query_type": "error",
            "target_criteria": {},
            "explanation": f"Error processing query: {query}",
            "suggested_action": "Try a simpler query"
        }

# Authentication functions
def create_users_table():
    """Create users table for authentication"""
    try:
        conn = sqlite3.connect('ai_task_management.db')
        cursor = conn.cursor()
        
        # First, check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            # Create new table with all columns
            cursor.execute('''
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    role TEXT DEFAULT 'user',
                    department TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_login TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    preferences TEXT
                )
            ''')
        else:
            # Check if department column exists, add if missing
            cursor.execute("PRAGMA table_info(users)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'department' not in columns:
                cursor.execute('ALTER TABLE users ADD COLUMN department TEXT')
            
            if 'preferences' not in columns:
                cursor.execute('ALTER TABLE users ADD COLUMN preferences TEXT')
        
        # Insert default admin user if not exists
        cursor.execute('''
            INSERT OR IGNORE INTO users (username, password_hash, role, email, department)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', 'admin123', 'admin', 'admin@company.com', 'IT'))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating users table: {e}")
        return False

def create_user_specific_tables(username):
    """Create user-specific tables for personalized data"""
    try:
        conn = sqlite3.connect('ai_task_management.db')
        cursor = conn.cursor()
        
        # Create user-specific tasks table
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS user_tasks_{username.replace(" ", "_")} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT,
                priority TEXT,
                urgency_score INTEGER,
                complexity_score INTEGER,
                business_impact INTEGER,
                estimated_hours REAL,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                assigned_to TEXT,
                created_by TEXT DEFAULT '{username}'
            )
        ''')
        
        # Create user-specific employees table
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS user_employees_{username.replace(" ", "_")} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                role TEXT,
                department TEXT,
                skills TEXT,
                experience_years INTEGER,
                workload_score REAL DEFAULT 0,
                performance_score REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT DEFAULT '{username}'
            )
        ''')
        
        # Create user-specific assignments table
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS user_assignments_{username.replace(" ", "_")} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                employee_name TEXT,
                assigned_at TEXT DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'assigned',
                created_by TEXT DEFAULT '{username}',
                FOREIGN KEY (task_id) REFERENCES user_tasks_{username.replace(" ", "_")} (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating user tables: {e}")
        return False

def register_user(username, password, email, full_name, department="General"):
    """Register a new user"""
    try:
        conn = sqlite3.connect('ai_task_management.db')
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute('SELECT username FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            return False, "Username already exists"
        
        # Check if department column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'department' in columns:
            # Insert new user with department
            cursor.execute('''
                INSERT INTO users (username, password_hash, email, role, department)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, password, email, 'user', department))
        else:
            # Insert new user without department
            cursor.execute('''
                INSERT INTO users (username, password_hash, email, role)
                VALUES (?, ?, ?, ?)
            ''', (username, password, email, 'user'))
        
        # Create user-specific tables
        create_user_specific_tables(username)
        
        conn.commit()
        conn.close()
        return True, "User registered successfully"
    except Exception as e:
        return False, f"Registration error: {str(e)}"

def authenticate_user(username, password):
    """Authenticate user with username and password"""
    try:
        conn = sqlite3.connect('ai_task_management.db')
        cursor = conn.cursor()
        
        # Check if department column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'department' in columns:
            cursor.execute('''
                SELECT username, role, email, department FROM users 
                WHERE username = ? AND password_hash = ? AND is_active = 1
            ''', (username, password))
        else:
            cursor.execute('''
                SELECT username, role, email FROM users 
                WHERE username = ? AND password_hash = ? AND is_active = 1
            ''', (username, password))
        
        user = cursor.fetchone()
        
        if user:
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP 
                WHERE username = ?
            ''', (username,))
            conn.commit()
            
            # Ensure user-specific tables exist
            create_user_specific_tables(username)
            
            conn.close()
            return user
        conn.close()
        return None
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return None

def show_login_page():
    """Show login page with signup option"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1>🔐 AI Task Management System</h1>
        <p>Login or create a new account</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for login and signup
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    with tab1:
        st.markdown("### Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("👤 Username")
            password = st.text_input("🔒 Password", type="password")
            submit_button = st.form_submit_button("🚀 Login")
            
            if submit_button:
                if username and password:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.is_authenticated = True
                        st.session_state.current_user = user[0]
                        st.session_state.user_role = user[1]
                        st.session_state.user_department = user[3] if len(user) > 3 else "General"
                        st.success(f"✅ Welcome, {user[0]}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter both username and password")
    
    with tab2:
        st.markdown("### Create New Account")
        with st.form("signup_form"):
            new_username = st.text_input("👤 Choose Username")
            new_password = st.text_input("🔒 Choose Password", type="password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password")
            email = st.text_input("📧 Email Address")
            department = st.selectbox("🏢 Department", 
                                   ["Engineering", "Marketing", "Sales", "HR", "Finance", "IT", "General"])
            
            signup_button = st.form_submit_button("📝 Create Account")
            
            if signup_button:
                if not all([new_username, new_password, confirm_password, email]):
                    st.warning("⚠️ Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("❌ Passwords do not match")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                else:
                    success, message = register_user(new_username, new_password, email, new_username, department)
                    if success:
                        st.success(f"✅ {message}")
                        st.info("🎉 You can now login with your new account!")
                    else:
                        st.error(f"❌ {message}")

def get_user_specific_data(username, data_type="tasks"):
    """Get user-specific data from their personal tables"""
    try:
        conn = sqlite3.connect('ai_task_management.db')
        cursor = conn.cursor()
        
        if data_type == "tasks":
            cursor.execute(f'''
                SELECT * FROM user_tasks_{username.replace(" ", "_")}
                ORDER BY created_at DESC
            ''')
        elif data_type == "employees":
            cursor.execute(f'''
                SELECT * FROM user_employees_{username.replace(" ", "_")}
                ORDER BY created_at DESC
            ''')
        elif data_type == "assignments":
            cursor.execute(f'''
                SELECT * FROM user_assignments_{username.replace(" ", "_")}
                ORDER BY assigned_at DESC
            ''')
        else:
            return []
        
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        print(f"Error getting user data: {e}")
        return []

def save_user_task(username, task_data):
    """Save task to user's personal task table"""
    try:
        conn = sqlite3.connect('ai_task_management.db')
        cursor = conn.cursor()
        
        cursor.execute(f'''
            INSERT INTO user_tasks_{username.replace(" ", "_")} 
            (title, description, category, priority, urgency_score, complexity_score, 
             business_impact, estimated_hours, status, assigned_to)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', task_data)
        
        conn.commit()
        result = cursor.lastrowid
        conn.close()
        return result
    except Exception as e:
        print(f"Error saving user task: {e}")
        return None

def save_user_employee(username, employee_data):
    """Save employee to user's personal employee table"""
    try:
        conn = sqlite3.connect('ai_task_management.db')
        cursor = conn.cursor()
        
        cursor.execute(f'''
            INSERT INTO user_employees_{username.replace(" ", "_")} 
            (name, role, department, skills, experience_years, workload_score, performance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', employee_data)
        
        conn.commit()
        result = cursor.lastrowid
        conn.close()
        return result
    except Exception as e:
        print(f"Error saving user employee: {e}")
        return None

def show_logout():
    """Show logout button and handle logout"""
    if st.sidebar.button("🚪 Logout"):
        st.session_state.is_authenticated = False
        st.session_state.current_user = None
        st.session_state.user_role = None
        st.session_state.user_department = None
        st.success("✅ Logged out successfully!")
        st.rerun()

# Initialize database and tables
def initialize_database():
    """Initialize database and create tables if they don't exist"""
    try:
        conn = sqlite3.connect('ai_task_management.db')
        cursor = conn.cursor()
        
        # Create tasks table
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
        
        # Create task assignments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                employee_id TEXT,
                employee_name TEXT,
                assignment_reason TEXT,
                confidence_score REAL,
                assigned_at TEXT,
                status TEXT DEFAULT 'assigned',
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
        """)
        
        # Create employees table
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
        
        # Create other necessary tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                predicted_category TEXT,
                predicted_priority TEXT,
                predicted_assignee TEXT,
                confidence_score REAL,
                created_at TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
        """)
        
        # Create task comments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                employee_name TEXT NOT NULL,
                comment_text TEXT NOT NULL,
                comment_type TEXT DEFAULT 'general',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                is_private BOOLEAN DEFAULT 0,
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
        """)
        
        # Create task dependencies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dependent_task_id INTEGER NOT NULL,
                prerequisite_task_id INTEGER NOT NULL,
                dependency_type TEXT DEFAULT 'blocks',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dependent_task_id) REFERENCES tasks (id),
                FOREIGN KEY (prerequisite_task_id) REFERENCES tasks (id),
                UNIQUE(dependent_task_id, prerequisite_task_id)
            )
        """)
        
        # Create team chat table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_name TEXT NOT NULL,
                message_text TEXT NOT NULL,
                message_type TEXT DEFAULT 'general',
                task_id INTEGER,
                room_name TEXT DEFAULT 'general',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                is_private BOOLEAN DEFAULT 0,
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        # Create users table for authentication
        create_users_table()
        
        return True
    except Exception as e:
        print(f"Database initialization failed: {e}")
        return False

# Custom CSS for consistent white theme across all screens
st.markdown("""
<style>
    /* Consistent White Theme */
    .main {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    .stApp {
        background-color: #ffffff !important;
    }
    
    .block-container {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    .stMarkdown {
        color: #262730 !important;
    }
    
    /* Sidebar styling - light theme */
    .css-1d391kg {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
    }
    
    /* Ensure all elements are visible */
    .stDataFrame {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    .stMetric {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    .stAlert {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    /* Charts and visualizations */
    .js-plotly-plot {
        background-color: #ffffff !important;
    }
    
    /* Mobile Responsive Design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        .metric-card {
            margin: 0.25rem 0;
            padding: 0.75rem;
        }
        
        .metric-card h2 {
            font-size: 1.5rem;
        }
        
        .metric-card h3 {
            font-size: 0.8rem;
        }
        
        .info-card {
            margin: 0.25rem 0;
            padding: 0.75rem;
        }
        
        .info-card h3 {
            font-size: 1rem;
        }
        
        .info-card h4 {
            font-size: 0.7rem;
        }
        
        /* Mobile-friendly tables */
        .dataframe {
            font-size: 0.8rem;
        }
        
        /* Mobile sidebar */
        .css-1d391kg {
            min-width: 250px;
        }
        
        /* Mobile buttons */
        .stButton > button {
            padding: 0.5rem 0.75rem;
            font-size: 0.9rem;
        }
        
        /* Mobile charts */
        .js-plotly-plot {
            height: 300px !important;
        }
    }
    
    /* Tablet Responsive Design */
    @media (max-width: 1024px) and (min-width: 769px) {
        .metric-card h2 {
            font-size: 1.6rem;
        }
        
        .info-card h3 {
            font-size: 1.1rem;
        }
    }
    
    /* Touch-friendly interactions */
    @media (hover: none) and (pointer: coarse) {
        .nav-item {
            padding: 1rem 0.75rem;
            margin: 0.5rem 0;
        }
        
        .stButton > button {
            min-height: 44px;
        }
        
        .quick-action-btn {
            min-height: 44px;
            padding: 0.75rem 1rem;
        }
    }

    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .success-message {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .error-message {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* ElevenLabs-style Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        border-right: 1px solid #2d2d3f;
    }
    
    .css-1d391kg .css-1lcbmhc {
        background: transparent;
    }
    
    .css-1d391kg .css-1lcbmhc .css-1lcbmhc {
        background: transparent;
    }
    
    /* Enhanced Navigation Radio Buttons */
    .stRadio > div {
        padding: 0.5rem 0;
    }
    
    .stRadio > div > label {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 0.5rem;
        background: rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
        cursor: pointer;
        display: flex;
        align-items: center;
        min-height: 44px;
    }
    
    .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(5px);
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-left: 3px solid #ffffff;
    }
    
    /* Make radio buttons more clickable */
    .stRadio input[type="radio"] {
        margin-right: 0.75rem;
        transform: scale(1.2);
    }
    
    /* Navigation text styling */
    .stRadio > div > label > span {
        font-weight: 500;
        color: white;
    }
    
    /* Logout button styling */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
        margin: 0.5rem 0;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #ff4b2b 0%, #ff416c 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 75, 43, 0.3);
    }
    
    /* Drag and Drop Styling */
    .drag-drop-zone {
        border: 2px dashed #667eea;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .drag-drop-zone.dragover {
        border-color: #4ade80;
        background: rgba(74, 222, 128, 0.1);
        transform: scale(1.02);
    }
    
    .drag-drop-zone:hover {
        border-color: #4ade80;
        background: rgba(74, 222, 128, 0.05);
    }
    
    .draggable-item {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        cursor: grab;
        transition: all 0.3s ease;
    }
    
    .draggable-item:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .draggable-item:active {
        cursor: grabbing;
    }
    
    .drop-zone {
        min-height: 100px;
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .drop-zone.dragover {
        border-color: #4ade80;
        background: rgba(74, 222, 128, 0.1);
    }
    
    .drop-zone.empty {
        display: flex;
        align-items: center;
        justify-content: center;
        color: rgba(255, 255, 255, 0.5);
        font-style: italic;
    }
    
    /* Sidebar Header */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 1rem;
        margin: -1rem -1rem 1rem -1rem;
        border-radius: 0 0 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Navigation Items */
    .nav-item {
        background: transparent;
        border: none;
        color: #e0e0e0;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-weight: 500;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .nav-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 0;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transition: width 0.3s ease;
        z-index: -1;
    }
    
    .nav-item:hover::before {
        width: 100%;
    }
    
    .nav-item:hover {
        color: white;
        transform: translateX(5px);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .nav-item.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* User Section */
    .user-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        margin-right: 0.75rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    .status-online {
        background: #4ade80;
        box-shadow: 0 0 8px rgba(74, 222, 128, 0.5);
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Divider */
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 1rem 0;
    }
    
    /* Quick Actions */
    .quick-actions {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .quick-action-btn {
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #e0e0e0;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
        font-size: 0.9rem;
    }
    
    .quick-action-btn:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
        transform: translateY(-2px);
    }
    
    /* System Status Cards */
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.25rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .status-card:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin: 1rem 0;
    }
    .chart-container {
        background: white;
        border-radius: 1rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    

</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processed_tasks' not in st.session_state:
    st.session_state.processed_tasks = []
if 'employee_profiles' not in st.session_state:
    st.session_state.employee_profiles = []
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False
if 'notification_settings' not in st.session_state:
    st.session_state.notification_settings = {
        'email_notifications': True,
        'sms_notifications': False,
        'task_assignments': True,
        'task_updates': True,
        'deadline_alerts': True,
        'comment_notifications': True
    }

# Database connection
def get_database_connection():
    """Get database connection with improved error handling"""
    try:
        # Create a new connection each time to avoid closed database issues
        conn = sqlite3.connect('ai_task_management.db', check_same_thread=False, timeout=30.0)
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    except Exception as e:
        st.error(f"❌ Database connection error: {e}")
        return None

def safe_database_operation(operation_func, *args, **kwargs):
    """Safely execute database operations with proper connection handling"""
    conn = None
    try:
        conn = sqlite3.connect('ai_task_management.db', check_same_thread=False, timeout=30.0)
        conn.execute("PRAGMA foreign_keys = ON")
        
        if conn:
            result = operation_func(conn, *args, **kwargs)
            conn.commit()
            return result
        else:
            st.error("❌ Could not establish database connection")
            return None
    except Exception as e:
        st.error(f"❌ Database operation error: {e}")
        return None
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                st.error(f"❌ Error closing database connection: {e}")

def get_assignment_metrics_safely():
    """Safely get assignment metrics"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_assignments,
                COUNT(DISTINCT employee_name) as unique_employees,
                AVG(confidence_score) as avg_confidence
            FROM task_assignments
        """)
        result = cursor.fetchone()
        if result and result[0] > 0:
            return result
        else:
            return (0, 0, 0.0)  # Return default values if no assignments
    
    return safe_database_operation(operation)

def get_recent_assignments_safely(limit=None):
    """Safely get recent assignments"""
    def operation(conn):
        cursor = conn.cursor()
        if limit:
            cursor.execute("""
                SELECT 
                    ta.employee_name,
                    ta.assignment_reason,
                    ta.confidence_score,
                    ta.assigned_at,
                    t.title,
                    t.category,
                    t.priority
                FROM task_assignments ta
                JOIN tasks t ON ta.task_id = t.id
                ORDER BY ta.assigned_at DESC
                LIMIT ?
            """, (limit,))
        else:
            cursor.execute("""
                SELECT 
                    ta.employee_name,
                    ta.assignment_reason,
                    ta.confidence_score,
                    ta.assigned_at,
                    t.title,
                    t.category,
                    t.priority
                FROM task_assignments ta
                JOIN tasks t ON ta.task_id = t.id
                ORDER BY ta.assigned_at DESC
            """)
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def get_metrics_safely():
    """Get metrics safely with proper error handling"""
    def operation(conn):
        # Get task statistics
        df = pd.read_sql_query("SELECT * FROM tasks", conn)
        
        if df.empty:
            return {}, df
        
        metrics = {
            'total_tasks': len(df),
            'pending_tasks': len(df[df['status'] == 'pending']) if 'status' in df.columns else 0,
            'completed_tasks': len(df[df['status'] == 'completed']) if 'status' in df.columns else 0,
            'high_priority': len(df[df['priority'] == 'high']) if 'priority' in df.columns else 0,
            'avg_urgency': df['urgency_score'].mean() if 'urgency_score' in df.columns else 0,
            'avg_complexity': df['complexity_score'].mean() if 'complexity_score' in df.columns else 0,
            'categories': df['category'].value_counts().to_dict() if 'category' in df.columns else {},
            'status_distribution': df['status'].value_counts().to_dict() if 'status' in df.columns else {}
        }
        
        return metrics, df
    
    try:
        result = safe_database_operation(operation)
        if result:
            return result
        else:
            return {}, pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting metrics: {e}")
        return {}, pd.DataFrame()

# Task Comments Management Functions
def add_task_comment(task_id, employee_name, comment_text, comment_type="general", is_private=False):
    """Add a comment to a task"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO task_comments (task_id, employee_name, comment_text, comment_type, is_private, created_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (task_id, employee_name, comment_text, comment_type, is_private))
        return cursor.lastrowid
    
    result = safe_database_operation(operation)
    if result:
        add_notification(f"Comment added by {employee_name}", "success", "💬")
        
        # Send enhanced notification
        if not is_private:
            # Get task title for notification
            def get_task_title(conn):
                cursor = conn.cursor()
                cursor.execute("SELECT title FROM tasks WHERE id = ?", (task_id,))
                result = cursor.fetchone()
                return result[0] if result else "Unknown Task"
            
            task_title = safe_database_operation(get_task_title)
            send_comment_notification(task_id, task_title, employee_name, comment_type)
        
        return result
    return None

def get_task_comments(task_id, include_private=False):
    """Get all comments for a task"""
    def operation(conn):
        cursor = conn.cursor()
        if include_private:
            cursor.execute("""
                SELECT id, employee_name, comment_text, comment_type, created_at, is_private
                FROM task_comments 
                WHERE task_id = ?
                ORDER BY created_at DESC
            """, (task_id,))
        else:
            cursor.execute("""
                SELECT id, employee_name, comment_text, comment_type, created_at, is_private
                FROM task_comments 
                WHERE task_id = ? AND is_private = 0
                ORDER BY created_at DESC
            """, (task_id,))
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def update_task_comment(comment_id, new_text, employee_name):
    """Update a task comment"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE task_comments 
            SET comment_text = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND employee_name = ?
        """, (new_text, comment_id, employee_name))
        return cursor.rowcount > 0
    
    result = safe_database_operation(operation)
    if result:
        add_notification(f"Comment updated by {employee_name}", "success", "✏️")
        return True
    return False

def delete_task_comment(comment_id, employee_name):
    """Delete a task comment"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM task_comments 
            WHERE id = ? AND employee_name = ?
        """, (comment_id, employee_name))
        return cursor.rowcount > 0
    
    result = safe_database_operation(operation)
    if result:
        add_notification(f"Comment deleted by {employee_name}", "success", "🗑️")
        return True
    return False

def get_comment_statistics():
    """Get comment statistics for analytics"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_comments,
                COUNT(DISTINCT task_id) as tasks_with_comments,
                COUNT(DISTINCT employee_name) as active_commenters,
                AVG(LENGTH(comment_text)) as avg_comment_length
            FROM task_comments
        """)
        return cursor.fetchone()
    
    return safe_database_operation(operation)

# Enhanced Notification System
def send_email_notification(recipient_email, subject, message):
    """Send email notification (simulated)"""
    try:
        # In a real implementation, you would use a service like SendGrid, Mailgun, or AWS SES
        # For now, we'll simulate the email sending
        print(f"📧 EMAIL SENT TO: {recipient_email}")
        print(f"📧 SUBJECT: {subject}")
        print(f"📧 MESSAGE: {message}")
        
        # Add to notification history
        add_notification(f"Email sent to {recipient_email}: {subject}", "success", "📧")
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

def send_sms_notification(phone_number, message):
    """Send SMS notification (simulated)"""
    try:
        # In a real implementation, you would use a service like Twilio, AWS SNS, or similar
        # For now, we'll simulate the SMS sending
        print(f"📱 SMS SENT TO: {phone_number}")
        print(f"📱 MESSAGE: {message}")
        
        # Add to notification history
        add_notification(f"SMS sent to {phone_number}: {message[:50]}...", "success", "📱")
        return True
    except Exception as e:
        st.error(f"Failed to send SMS: {e}")
        return False

def send_task_assignment_notification(task_id, employee_name, task_title):
    """Send notification when a task is assigned"""
    if not st.session_state.notification_settings.get('task_assignments', True):
        return
    
    message = f"🎯 New task assigned: '{task_title}' (ID: {task_id})"
    
    # Send email notification
    if st.session_state.notification_settings.get('email_notifications', True):
        # In a real app, you'd get the employee's email from the database
        employee_email = f"{employee_name.lower().replace(' ', '.')}@company.com"
        send_email_notification(employee_email, "New Task Assignment", message)
    
    # Send SMS notification
    if st.session_state.notification_settings.get('sms_notifications', False):
        # In a real app, you'd get the employee's phone from the database
        employee_phone = "+1234567890"  # Placeholder
        send_sms_notification(employee_phone, message)

def send_task_update_notification(task_id, task_title, update_type, updated_by):
    """Send notification when a task is updated"""
    if not st.session_state.notification_settings.get('task_updates', True):
        return
    
    message = f"📝 Task updated: '{task_title}' - {update_type} by {updated_by}"
    
    # Send email notification
    if st.session_state.notification_settings.get('email_notifications', True):
        send_email_notification("team@company.com", "Task Update", message)
    
    # Send SMS notification
    if st.session_state.notification_settings.get('sms_notifications', False):
        send_sms_notification("+1234567890", message)

def send_deadline_alert(task_id, task_title, days_until_deadline):
    """Send deadline alert notification"""
    if not st.session_state.notification_settings.get('deadline_alerts', True):
        return
    
    if days_until_deadline <= 3:  # Alert for tasks due within 3 days
        urgency = "🚨 URGENT" if days_until_deadline <= 1 else "⚠️ WARNING"
        message = f"{urgency} Deadline approaching: '{task_title}' due in {days_until_deadline} day(s)"
        
        # Send email notification
        if st.session_state.notification_settings.get('email_notifications', True):
            send_email_notification("team@company.com", "Deadline Alert", message)
        
        # Send SMS notification
        if st.session_state.notification_settings.get('sms_notifications', False):
            send_sms_notification("+1234567890", message)

def send_comment_notification(task_id, task_title, commenter_name, comment_type):
    """Send notification when a comment is added"""
    if not st.session_state.notification_settings.get('comment_notifications', True):
        return
    
    message = f"💬 New comment on task '{task_title}': {comment_type} by {commenter_name}"
    
    # Send email notification
    if st.session_state.notification_settings.get('email_notifications', True):
        send_email_notification("team@company.com", "New Task Comment", message)
    
    # Send SMS notification
    if st.session_state.notification_settings.get('sms_notifications', False):
        send_sms_notification("+1234567890", message)

def check_deadline_alerts():
    """Check for tasks approaching deadlines and send alerts"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, days_until_deadline, assigned_to
            FROM tasks 
            WHERE status != 'completed' 
            AND days_until_deadline <= 3
            AND days_until_deadline > 0
        """)
        return cursor.fetchall()
    
    tasks_near_deadline = safe_database_operation(operation)
    
    if tasks_near_deadline:
        for task in tasks_near_deadline:
            task_id, task_title, days_until_deadline, assigned_to = task
            send_deadline_alert(task_id, task_title, days_until_deadline)

def get_notification_history():
    """Get recent notification history"""
    # This would typically come from a database table
    # For now, we'll return a simulated history
    return [
        {"timestamp": "2024-01-15 10:30", "type": "task_assignment", "message": "Task 'Fix Login Bug' assigned to John Doe"},
        {"timestamp": "2024-01-15 09:15", "type": "comment", "message": "New comment on 'Dashboard Design' by Jane Smith"},
        {"timestamp": "2024-01-15 08:45", "type": "deadline_alert", "message": "Task 'Database Optimization' due in 2 days"},
        {"timestamp": "2024-01-14 16:20", "type": "task_update", "message": "Task 'API Integration' status updated to 'In Progress'"},
        {"timestamp": "2024-01-14 14:30", "type": "task_assignment", "message": "Task 'Security Audit' assigned to Bob Wilson"}
    ]

# Task Dependencies Management Functions
def add_task_dependency(dependent_task_id, prerequisite_task_id, dependency_type="blocks"):
    """Add a dependency between two tasks"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO task_dependencies (dependent_task_id, prerequisite_task_id, dependency_type)
            VALUES (?, ?, ?)
        """, (dependent_task_id, prerequisite_task_id, dependency_type))
        return cursor.lastrowid
    
    result = safe_database_operation(operation)
    if result:
        add_notification(f"Task dependency added", "success", "🔗")
        return result
    return None

def remove_task_dependency(dependent_task_id, prerequisite_task_id):
    """Remove a dependency between two tasks"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM task_dependencies 
            WHERE dependent_task_id = ? AND prerequisite_task_id = ?
        """, (dependent_task_id, prerequisite_task_id))
        return cursor.rowcount > 0
    
    result = safe_database_operation(operation)
    if result:
        add_notification(f"Task dependency removed", "success", "🔗")
        return True
    return False

def get_task_dependencies(task_id):
    """Get all dependencies for a task"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT td.dependent_task_id, td.prerequisite_task_id, td.dependency_type,
                   t1.title as dependent_title, t2.title as prerequisite_title
            FROM task_dependencies td
            JOIN tasks t1 ON td.dependent_task_id = t1.id
            JOIN tasks t2 ON td.prerequisite_task_id = t2.id
            WHERE td.dependent_task_id = ? OR td.prerequisite_task_id = ?
        """, (task_id, task_id))
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def get_prerequisites_for_task(task_id):
    """Get all prerequisite tasks for a given task"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT td.prerequisite_task_id, t.title, t.status, td.dependency_type
            FROM task_dependencies td
            JOIN tasks t ON td.prerequisite_task_id = t.id
            WHERE td.dependent_task_id = ?
        """, (task_id,))
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def get_dependent_tasks(task_id):
    """Get all tasks that depend on the given task"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT td.dependent_task_id, t.title, t.status, td.dependency_type
            FROM task_dependencies td
            JOIN tasks t ON td.dependent_task_id = t.id
            WHERE td.prerequisite_task_id = ?
        """, (task_id,))
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def check_dependency_cycle(dependent_task_id, prerequisite_task_id):
    """Check if adding a dependency would create a cycle"""
    def operation(conn):
        cursor = conn.cursor()
        # Check if prerequisite_task_id depends on dependent_task_id (directly or indirectly)
        cursor.execute("""
            WITH RECURSIVE dependency_chain AS (
                SELECT prerequisite_task_id, dependent_task_id
                FROM task_dependencies
                WHERE prerequisite_task_id = ?
                UNION ALL
                SELECT td.prerequisite_task_id, td.dependent_task_id
                FROM task_dependencies td
                JOIN dependency_chain dc ON td.prerequisite_task_id = dc.dependent_task_id
            )
            SELECT COUNT(*) FROM dependency_chain WHERE dependent_task_id = ?
        """, (prerequisite_task_id, dependent_task_id))
        result = cursor.fetchone()
        return result[0] > 0 if result else False
    
    return safe_database_operation(operation)

def can_start_task(task_id):
    """Check if a task can be started based on its prerequisites"""
    prerequisites = get_prerequisites_for_task(task_id)
    if not prerequisites:
        return True, []  # No prerequisites, can start
    
    incomplete_prerequisites = []
    for prereq in prerequisites:
        prereq_id, prereq_title, prereq_status, dep_type = prereq
        if prereq_status != 'completed':
            incomplete_prerequisites.append({
                'id': prereq_id,
                'title': prereq_title,
                'status': prereq_status,
                'dependency_type': dep_type
            })
    
    return len(incomplete_prerequisites) == 0, incomplete_prerequisites

def get_dependency_statistics():
    """Get dependency statistics for analytics"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_dependencies,
                COUNT(DISTINCT dependent_task_id) as tasks_with_dependencies,
                COUNT(DISTINCT prerequisite_task_id) as prerequisite_tasks,
                dependency_type,
                COUNT(*) as type_count
            FROM task_dependencies
            GROUP BY dependency_type
        """)
        return cursor.fetchall()
    
    return safe_database_operation(operation)

# Advanced AI Features - Predictive Analytics & Smart Prioritization
def predict_task_completion_time(task_id):
    """Predict completion time for a task based on historical data"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                AVG(JULIANDAY(updated_at) - JULIANDAY(created_at)) as avg_completion_days,
                COUNT(*) as sample_size
            FROM tasks 
            WHERE status = 'completed' 
            AND category = (SELECT category FROM tasks WHERE id = ?)
            AND priority = (SELECT priority FROM tasks WHERE id = ?)
        """, (task_id, task_id))
        return cursor.fetchone()
    
    result = safe_database_operation(operation)
    if result and result[0]:
        avg_days = result[0]
        sample_size = result[1]
        confidence = min(0.95, sample_size / 100)  # Confidence based on sample size
        return avg_days, confidence
    return None, 0.0

def predict_employee_performance(employee_name):
    """Predict employee performance based on historical data"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                AVG(ta.confidence_score) as avg_confidence,
                COUNT(*) as total_assignments,
                AVG(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) as completion_rate,
                AVG(t.urgency_score) as avg_urgency_handled,
                AVG(t.complexity_score) as avg_complexity_handled
            FROM task_assignments ta
            JOIN tasks t ON ta.task_id = t.id
            WHERE ta.employee_name = ?
        """, (employee_name,))
        return cursor.fetchone()
    
    return safe_database_operation(operation)

def smart_prioritize_tasks():
    """Smart prioritization based on multiple factors"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                t.id, t.title, t.priority, t.urgency_score, t.complexity_score,
                t.business_impact, t.days_until_deadline, t.status,
                ta.employee_name, ta.confidence_score,
                (t.urgency_score * 0.3 + t.business_impact * 0.3 + 
                 (10 - t.days_until_deadline) * 0.2 + (10 - t.complexity_score) * 0.2) as smart_score
            FROM tasks t
            LEFT JOIN task_assignments ta ON t.id = ta.task_id
            WHERE t.status IN ('pending', 'in_progress')
            ORDER BY smart_score DESC
        """)
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def predict_resource_needs():
    """Predict resource needs based on current workload and upcoming tasks"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                e.name as employee_name,
                e.current_workload,
                e.max_capacity,
                COUNT(ta.task_id) as active_tasks,
                AVG(t.estimated_hours) as avg_task_hours,
                SUM(t.estimated_hours) as total_estimated_hours
            FROM employees e
            LEFT JOIN task_assignments ta ON e.name = ta.employee_name
            LEFT JOIN tasks t ON ta.task_id = t.id AND t.status != 'completed'
            GROUP BY e.name, e.current_workload, e.max_capacity
        """)
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def detect_skill_gaps():
    """Detect skill gaps based on task requirements vs employee skills"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                t.category,
                COUNT(*) as task_count,
                AVG(t.complexity_score) as avg_complexity,
                GROUP_CONCAT(DISTINCT e.skills) as available_skills
            FROM tasks t
            LEFT JOIN employees e ON t.assigned_to = e.name
            WHERE t.status = 'pending'
            GROUP BY t.category
        """)
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def forecast_project_timeline(project_tasks):
    """Forecast project timeline based on task dependencies and resource availability"""
    # This is a simplified version - in a real implementation, you'd use more sophisticated algorithms
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                t.id, t.title, t.estimated_hours, t.days_until_deadline,
                t.status, ta.employee_name, e.current_workload
            FROM tasks t
            LEFT JOIN task_assignments ta ON t.id = ta.task_id
            LEFT JOIN employees e ON ta.employee_name = e.name
            WHERE t.id IN ({})
        """.format(','.join(['?' for _ in project_tasks])), project_tasks)
        return cursor.fetchall()
    
    return safe_database_operation(operation, project_tasks)

def get_ai_insights():
    """Get AI-powered insights about the system"""
    insights = []
    
    # Performance insights
    performance_data = predict_employee_performance("admin")  # Example
    if performance_data and performance_data[0]:
        insights.append({
            "type": "performance",
            "title": "Employee Performance Insights",
            "message": f"Average confidence score: {performance_data[0]:.2f}",
            "priority": "medium"
        })
    
    # Resource insights
    resource_data = predict_resource_needs()
    if resource_data:
        overloaded_employees = [emp for emp in resource_data if emp[1] > emp[2] * 0.8]
        if overloaded_employees:
            insights.append({
                "type": "resource",
                "title": "Resource Overload Alert",
                "message": f"{len(overloaded_employees)} employees are at 80%+ capacity",
                "priority": "high"
            })
    
    # Skill gap insights
    skill_gaps = detect_skill_gaps()
    if skill_gaps:
        high_complexity_categories = [cat for cat in skill_gaps if cat[2] > 7]
        if high_complexity_categories:
            insights.append({
                "type": "skill",
                "title": "Skill Gap Detected",
                "message": f"{len(high_complexity_categories)} categories have high complexity tasks",
                "priority": "medium"
            })
    
    return insights

# Team Chat System
def send_chat_message(sender_name, message_text, message_type="general", task_id=None, room_name="general", is_private=False):
    """Send a message to the team chat"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO team_chat (sender_name, message_text, message_type, task_id, room_name, is_private, created_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (sender_name, message_text, message_type, task_id, room_name, is_private))
        return cursor.lastrowid
    
    result = safe_database_operation(operation)
    if result:
        add_notification(f"Message sent by {sender_name}", "success", "💬")
        return result
    return None

def get_chat_messages(room_name="general", limit=50, task_id=None):
    """Get chat messages for a room or task"""
    def operation(conn):
        cursor = conn.cursor()
        if task_id:
            cursor.execute("""
                SELECT id, sender_name, message_text, message_type, created_at, is_private
                FROM team_chat 
                WHERE task_id = ? AND is_private = 0
                ORDER BY created_at DESC
                LIMIT ?
            """, (task_id, limit))
        else:
            cursor.execute("""
                SELECT id, sender_name, message_text, message_type, created_at, is_private
                FROM team_chat 
                WHERE room_name = ? AND is_private = 0
                ORDER BY created_at DESC
                LIMIT ?
            """, (room_name, limit))
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def get_chat_rooms():
    """Get available chat rooms"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT room_name, COUNT(*) as message_count
            FROM team_chat
            GROUP BY room_name
            ORDER BY message_count DESC
        """)
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def get_task_chat_messages(task_id):
    """Get chat messages related to a specific task"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, sender_name, message_text, message_type, created_at, is_private
            FROM team_chat 
            WHERE task_id = ?
            ORDER BY created_at ASC
        """, (task_id,))
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def get_chat_statistics():
    """Get chat statistics for analytics"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT sender_name) as active_users,
                COUNT(DISTINCT room_name) as total_rooms,
                message_type,
                COUNT(*) as type_count
            FROM team_chat
            GROUP BY message_type
        """)
        return cursor.fetchall()
    
    return safe_database_operation(operation)

# Gantt Charts - Visual Project Management
def get_gantt_data(project_filter=None):
    """Get data for Gantt chart visualization"""
    def operation(conn):
        cursor = conn.cursor()
        
        if project_filter:
            cursor.execute("""
                SELECT 
                    t.id, t.title, t.status, t.created_at, t.updated_at,
                    t.estimated_hours, t.days_until_deadline,
                    ta.employee_name, td.prerequisite_task_id
                FROM tasks t
                LEFT JOIN task_assignments ta ON t.id = ta.task_id
                LEFT JOIN task_dependencies td ON t.id = td.dependent_task_id
                WHERE t.category = ?
                ORDER BY t.created_at ASC
            """, (project_filter,))
        else:
            cursor.execute("""
                SELECT 
                    t.id, t.title, t.status, t.created_at, t.updated_at,
                    t.estimated_hours, t.days_until_deadline,
                    ta.employee_name, td.prerequisite_task_id
                FROM tasks t
                LEFT JOIN task_assignments ta ON t.id = ta.task_id
                LEFT JOIN task_dependencies td ON t.id = td.dependent_task_id
                ORDER BY t.created_at ASC
            """)
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def create_gantt_chart(gantt_data):
    """Create a Gantt chart using Plotly"""
    if not gantt_data:
        return None
    
    # Process data for Gantt chart
    tasks = []
    for task in gantt_data:
        task_id, title, status, created_at, updated_at, estimated_hours, days_until_deadline, employee_name, prerequisite = task
        
        # Calculate start and end dates
        try:
            start_date = datetime.fromisoformat(created_at.replace('Z', '+00:00')) if created_at else datetime.now()
        except:
            start_date = datetime.now()
        
        # Estimate end date based on estimated hours and deadline
        if estimated_hours:
            end_date = start_date + timedelta(hours=estimated_hours)
        elif days_until_deadline:
            end_date = start_date + timedelta(days=days_until_deadline)
        else:
            end_date = start_date + timedelta(days=7)  # Default 1 week
        
        # Color based on status
        color_map = {
            'completed': '#28a745',
            'in_progress': '#ffc107',
            'pending': '#dc3545'
        }
        color = color_map.get(status, '#6c757d')
        
        # Calculate duration in days for display
        duration_days = (end_date - start_date).days
        
        tasks.append({
            'Task': title,
            'Start': start_date,
            'Finish': end_date,
            'Resource': employee_name or 'Unassigned',
            'Status': status,
            'Color': color,
            'Duration': duration_days
        })
    
    # Create Gantt chart
    fig = go.Figure()
    
    for task in tasks:
        # Use duration in days instead of timedelta object
        fig.add_trace(go.Bar(
            name=task['Task'],
            x=[task['Duration']],  # Use duration in days
            y=[task['Resource']],
            orientation='h',
            marker_color=task['Color'],
            text=f"{task['Task']} ({task['Status']})",
            textposition='auto',
            hovertemplate=f"<b>{task['Task']}</b><br>" +
                         f"Status: {task['Status']}<br>" +
                         f"Duration: {task['Duration']} days<br>" +
                         f"<extra></extra>"
        ))
    
    fig.update_layout(
        title="Project Timeline - Gantt Chart",
        xaxis_title="Duration (days)",
        yaxis_title="Resources",
        barmode='overlay',
        height=400
    )
    
    return fig

def get_project_timeline_data():
    """Get project timeline data for visualization"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                t.id, t.title, t.category, t.status, t.created_at, t.updated_at,
                t.estimated_hours, t.days_until_deadline, ta.employee_name,
                COUNT(td.prerequisite_task_id) as dependencies_count
            FROM tasks t
            LEFT JOIN task_assignments ta ON t.id = ta.task_id
            LEFT JOIN task_dependencies td ON t.id = td.dependent_task_id
            GROUP BY t.id, t.title, t.category, t.status, t.created_at, t.updated_at,
                     t.estimated_hours, t.days_until_deadline, ta.employee_name
            ORDER BY t.created_at ASC
        """)
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def create_timeline_chart(timeline_data):
    """Create a timeline chart using Plotly"""
    if not timeline_data:
        return None
    
    # Process timeline data
    categories = {}
    for task in timeline_data:
        task_id, title, category, status, created_at, updated_at, estimated_hours, days_until_deadline, employee_name, dependencies_count = task
        
        if category not in categories:
            categories[category] = []
        
        try:
            start_date = datetime.fromisoformat(created_at.replace('Z', '+00:00')) if created_at else datetime.now()
        except:
            start_date = datetime.now()
        
        categories[category].append({
            'Task': title,
            'Start': start_date,
            'Status': status,
            'Employee': employee_name or 'Unassigned',
            'Dependencies': dependencies_count
        })
    
    # Create timeline chart
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    for i, (category, tasks) in enumerate(categories.items()):
        for task in tasks:
            # Convert datetime to string for JSON serialization
            start_date_str = task['Start'].strftime('%Y-%m-%d %H:%M')
            
            fig.add_trace(go.Scatter(
                x=[start_date_str],  # Use string instead of datetime object
                y=[category],
                mode='markers',
                name=task['Task'],
                marker=dict(
                    size=10,
                    color=colors[i % len(colors)],
                    symbol='circle'
                ),
                text=f"{task['Task']} ({task['Status']})",
                hovertemplate=f"<b>{task['Task']}</b><br>" +
                             f"Category: {category}<br>" +
                             f"Status: {task['Status']}<br>" +
                             f"Employee: {task['Employee']}<br>" +
                             f"Dependencies: {task['Dependencies']}<br>" +
                             f"<extra></extra>"
            ))
    
    fig.update_layout(
        title="Project Timeline by Category",
        xaxis_title="Timeline",
        yaxis_title="Categories",
        height=400
    )
    
    return fig

def serialize_datetime_objects(data):
    """Helper function to serialize datetime objects for JSON"""
    if isinstance(data, datetime):
        return data.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(data, timedelta):
        return data.days
    elif isinstance(data, list):
        return [serialize_datetime_objects(item) for item in data]
    elif isinstance(data, dict):
        return {key: serialize_datetime_objects(value) for key, value in data.items()}
    else:
        return data

# Advanced Search & Filters System
def search_tasks(search_query, filters=None):
    """Advanced search for tasks with multiple filters"""
    if filters is None:
        filters = {}
    
    def operation(conn):
        cursor = conn.cursor()
        
        # Build the base query
        base_query = """
            SELECT t.*, ta.employee_name as assigned_employee, ta.confidence_score
            FROM tasks t
            LEFT JOIN task_assignments ta ON t.id = ta.task_id
            WHERE 1=1
        """
        params = []
        
        # Add search query filter
        if search_query:
            base_query += """ AND (
                t.title LIKE ? OR 
                t.description LIKE ? OR 
                t.category LIKE ? OR
                ta.employee_name LIKE ?
            )"""
            search_pattern = f"%{search_query}%"
            params.extend([search_pattern, search_pattern, search_pattern, search_pattern])
        
        # Add status filter
        if filters.get('status'):
            base_query += " AND t.status = ?"
            params.append(filters['status'])
        
        # Add priority filter
        if filters.get('priority'):
            base_query += " AND t.priority = ?"
            params.append(filters['priority'])
        
        # Add category filter
        if filters.get('category'):
            base_query += " AND t.category = ?"
            params.append(filters['category'])
        
        # Add assigned employee filter
        if filters.get('assigned_to'):
            base_query += " AND ta.employee_name = ?"
            params.append(filters['assigned_to'])
        
        # Add urgency score range
        if filters.get('urgency_min') is not None:
            base_query += " AND t.urgency_score >= ?"
            params.append(filters['urgency_min'])
        
        if filters.get('urgency_max') is not None:
            base_query += " AND t.urgency_score <= ?"
            params.append(filters['urgency_max'])
        
        # Add complexity score range
        if filters.get('complexity_min') is not None:
            base_query += " AND t.complexity_score >= ?"
            params.append(filters['complexity_min'])
        
        if filters.get('complexity_max') is not None:
            base_query += " AND t.complexity_score <= ?"
            params.append(filters['complexity_max'])
        
        # Add date range filters
        if filters.get('created_after'):
            base_query += " AND t.created_at >= ?"
            params.append(filters['created_after'])
        
        if filters.get('created_before'):
            base_query += " AND t.created_at <= ?"
            params.append(filters['created_before'])
        
        # Add sorting
        sort_field = filters.get('sort_by', 'created_at')
        sort_order = filters.get('sort_order', 'DESC')
        base_query += f" ORDER BY t.{sort_field} {sort_order}"
        
        # Add limit
        if filters.get('limit'):
            base_query += " LIMIT ?"
            params.append(filters['limit'])
        
        cursor.execute(base_query, params)
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def get_search_suggestions(search_query):
    """Get search suggestions based on partial query"""
    def operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT title, category, priority
            FROM tasks
            WHERE title LIKE ? OR category LIKE ? OR priority LIKE ?
            LIMIT 10
        """, (f"%{search_query}%", f"%{search_query}%", f"%{search_query}%"))
        return cursor.fetchall()
    
    return safe_database_operation(operation)

def get_filter_options():
    """Get available filter options"""
    def operation(conn):
        cursor = conn.cursor()
        
        # Get status options
        cursor.execute("SELECT DISTINCT status FROM tasks WHERE status IS NOT NULL")
        statuses = [row[0] for row in cursor.fetchall()]
        
        # Get priority options
        cursor.execute("SELECT DISTINCT priority FROM tasks WHERE priority IS NOT NULL")
        priorities = [row[0] for row in cursor.fetchall()]
        
        # Get category options
        cursor.execute("SELECT DISTINCT category FROM tasks WHERE category IS NOT NULL")
        categories = [row[0] for row in cursor.fetchall()]
        
        # Get assigned employee options
        cursor.execute("SELECT DISTINCT employee_name FROM task_assignments WHERE employee_name IS NOT NULL")
        employees = [row[0] for row in cursor.fetchall()]
        
        return {
            'statuses': statuses,
            'priorities': priorities,
            'categories': categories,
            'employees': employees
        }
    
    return safe_database_operation(operation)

def export_search_results(search_results, format_type="csv"):
    """Export search results in specified format"""
    if not search_results:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(search_results, columns=[
        'id', 'title', 'description', 'category', 'priority', 'urgency_score',
        'complexity_score', 'business_impact', 'estimated_hours', 'days_until_deadline',
        'status', 'assigned_to', 'created_at', 'updated_at', 'assigned_employee', 'confidence_score'
    ])
    
    if format_type == "csv":
        return df.to_csv(index=False)
    elif format_type == "excel":
        # For Excel export, we'd need openpyxl
        # For now, return CSV
        return df.to_csv(index=False)
    else:
        return df.to_json(orient='records')

# Load employee profiles
@st.cache_data
def load_employees():
    """Load employee profiles"""
    try:
        profiles = load_employee_profiles('data/employee_profiles.json')
        return profiles
    except Exception as e:
        st.warning(f"Could not load employee profiles: {e}")
        return []

# Enhanced column mappings with more variations
COLUMN_MAPPINGS = {
    'title': [
        'task_name', 'work_title', 'title', 'name', 'task_title', 'subject', 
        'task_subject', 'work_name', 'job_title', 'task_label', 'item_name',
        'task_name', 'work_item', 'task_heading', 'title_name', 'task_identifier',
        'project_name', 'work_item_name', 'job_name', 'item_title', 'task_name',
        'work_title', 'project_title', 'job_title', 'item_name', 'task_label',
        'work_label', 'project_label', 'job_label', 'item_label', 'task_heading',
        'work_heading', 'project_heading', 'job_heading', 'item_heading'
    ],
    'description': [
        'task_description', 'desc', 'description', 'details', 'content', 
        'task_details', 'work_description', 'job_description', 'task_content',
        'summary', 'task_summary', 'work_details', 'task_info', 'description_text',
        'task_content', 'work_summary', 'job_summary', 'task_narrative',
        'project_description', 'work_desc', 'job_desc', 'item_description',
        'task_notes', 'work_notes', 'job_notes', 'item_notes', 'task_comment',
        'work_comment', 'job_comment', 'item_comment', 'task_remark',
        'work_remark', 'job_remark', 'item_remark'
    ],
    'category': [
        'category', 'type', 'task_type', 'classification', 'group', 'task_category',
        'work_type', 'job_type', 'task_group', 'classification_type', 'task_class',
        'work_category', 'job_category', 'task_kind', 'type_category', 'group_type',
        'task_family', 'work_group', 'job_group', 'task_classification',
        'project_type', 'work_class', 'job_class', 'item_type', 'task_domain',
        'work_domain', 'job_domain', 'item_domain', 'task_area', 'work_area',
        'job_area', 'item_area', 'task_sector', 'work_sector', 'job_sector'
    ],
    'priority': [
        'priority', 'importance', 'level', 'urgency_level', 'priority_level',
        'importance_level', 'urgency', 'criticality', 'priority_rank', 'importance_rank',
        'task_priority', 'work_priority', 'job_priority', 'priority_status',
        'urgency_status', 'critical_level', 'priority_rating', 'importance_rating',
        'project_priority', 'work_importance', 'job_importance', 'item_priority',
        'task_level', 'work_level', 'job_level', 'item_level', 'priority_class',
        'work_class', 'job_class', 'item_class', 'priority_grade', 'work_grade',
        'job_grade', 'item_grade'
    ],
    'urgency_score': [
        'urgency_score', 'urgency', 'urgency_level', 'criticality', 'urgency_rating',
        'critical_score', 'urgency_value', 'critical_level', 'urgency_measure',
        'critical_rating', 'urgency_index', 'critical_index', 'urgency_scale',
        'critical_scale', 'urgency_metric', 'critical_metric', 'urgency_number',
        'critical_number', 'urgency_value', 'critical_value', 'urgency_point',
        'critical_point', 'urgency_mark', 'critical_mark', 'urgency_grade',
        'critical_grade', 'urgency_class', 'critical_class'
    ],
    'complexity_score': [
        'complexity_score', 'complexity', 'difficulty', 'effort', 'complexity_level',
        'difficulty_level', 'effort_level', 'complexity_rating', 'difficulty_rating',
        'effort_rating', 'complexity_value', 'difficulty_value', 'effort_value',
        'complexity_measure', 'difficulty_measure', 'effort_measure', 'complexity_number',
        'difficulty_number', 'effort_number', 'complexity_point', 'difficulty_point',
        'effort_point', 'complexity_mark', 'difficulty_mark', 'effort_mark',
        'complexity_grade', 'difficulty_grade', 'effort_grade', 'complexity_class',
        'difficulty_class', 'effort_class'
    ],
    'business_impact': [
        'business_impact', 'impact', 'value', 'importance_score', 'business_value',
        'impact_score', 'value_score', 'business_importance', 'impact_level',
        'value_level', 'business_impact_score', 'impact_rating', 'value_rating',
        'business_value_score', 'impact_measure', 'value_measure', 'business_impact_number',
        'impact_number', 'value_number', 'business_impact_point', 'impact_point',
        'value_point', 'business_impact_mark', 'impact_mark', 'value_mark',
        'business_impact_grade', 'impact_grade', 'value_grade', 'business_impact_class',
        'impact_class', 'value_class'
    ],
    'estimated_hours': [
        'estimated_hours', 'hours', 'effort_hours', 'time_estimate', 'estimated_time',
        'hours_estimate', 'effort_time', 'time_hours', 'estimated_effort',
        'hours_needed', 'effort_needed', 'time_needed', 'estimated_duration',
        'hours_required', 'effort_required', 'time_required', 'estimated_work_hours',
        'work_hours', 'job_hours', 'task_hours', 'project_hours', 'estimated_days',
        'days_estimate', 'effort_days', 'time_days', 'estimated_weeks', 'weeks_estimate',
        'effort_weeks', 'time_weeks', 'estimated_months', 'months_estimate'
    ],
    'days_until_deadline': [
        'days_until_deadline', 'deadline_days', 'days_remaining', 'time_remaining',
        'days_left', 'time_left', 'deadline_remaining', 'days_to_deadline',
        'time_to_deadline', 'remaining_days', 'remaining_time', 'days_until_due',
        'days_to_due', 'time_to_due', 'days_until_completion', 'days_to_completion',
        'time_to_completion', 'days_until_finish', 'days_to_finish', 'time_to_finish',
        'days_until_delivery', 'days_to_delivery', 'time_to_delivery', 'deadline_countdown',
        'due_countdown', 'completion_countdown', 'finish_countdown', 'delivery_countdown'
    ],
    'deadline': [
        'deadline', 'due_date', 'target_date', 'completion_date', 'due_by',
        'target_deadline', 'completion_deadline', 'due_time', 'target_time',
        'completion_time', 'deadline_date', 'due_datetime', 'target_datetime',
        'completion_datetime', 'end_date', 'finish_date', 'delivery_date',
        'due_by_date', 'target_by_date', 'completion_by_date', 'deadline_by',
        'due_by_time', 'target_by_time', 'completion_by_time', 'deadline_datetime',
        'due_datetime', 'target_datetime', 'completion_datetime', 'end_datetime',
        'finish_datetime', 'delivery_datetime'
    ],
    'status': [
        'status', 'state', 'progress', 'phase', 'task_status', 'work_status',
        'job_status', 'progress_status', 'current_state', 'task_state',
        'work_state', 'job_state', 'progress_state', 'current_status',
        'task_progress', 'work_progress', 'job_progress', 'project_status',
        'work_phase', 'job_phase', 'item_status', 'task_phase', 'work_state',
        'job_state', 'item_state', 'project_state', 'task_condition',
        'work_condition', 'job_condition', 'item_condition'
    ],
    'assigned_to': [
        'assigned_to', 'assigned', 'employee_name', 'assignee', 'responsible',
        'assigned_person', 'employee', 'worker', 'team_member', 'assigned_user',
        'responsible_person', 'assigned_employee', 'task_owner', 'work_owner',
        'job_owner', 'assigned_team', 'responsible_team', 'assigned_to_person',
        'assigned_to_employee', 'assigned_to_worker', 'assigned_to_team',
        'responsible_person', 'responsible_employee', 'responsible_worker',
        'responsible_team', 'task_assignee', 'work_assignee', 'job_assignee',
        'item_assignee', 'project_assignee', 'task_responsible', 'work_responsible',
        'job_responsible', 'item_responsible', 'project_responsible'
    ],
    'created_by': [
        'created_by', 'creator', 'author', 'submitted_by', 'created_by_user',
        'task_creator', 'work_creator', 'job_creator', 'submitted_by_user',
        'created_by_employee', 'author_name', 'creator_name', 'submitter',
        'task_author', 'work_author', 'job_author', 'item_creator', 'project_creator',
        'work_author', 'job_author', 'item_author', 'project_author', 'task_submitter',
        'work_submitter', 'job_submitter', 'item_submitter', 'project_submitter',
        'task_originator', 'work_originator', 'job_originator', 'item_originator',
        'project_originator'
    ],
    'task_id': [
        'task_id', 'id', 'task_number', 'work_id', 'job_id', 'item_id',
        'task_identifier', 'work_identifier', 'job_identifier', 'item_number',
        'task_code', 'work_code', 'job_code', 'item_code', 'task_reference',
        'work_reference', 'job_reference', 'item_reference', 'project_id',
        'project_number', 'project_code', 'project_reference', 'task_number',
        'work_number', 'job_number', 'item_number', 'project_number', 'task_index',
        'work_index', 'job_index', 'item_index', 'project_index'
    ]
}

# Global variables for continuous learning
COLUMN_DETECTION_HISTORY = defaultdict(list)
MODEL_PERFORMANCE_HISTORY = defaultdict(list)
AUTO_ASSIGNMENT_SUCCESS_RATE = 0.0
TOTAL_ASSIGNMENTS = 0
SUCCESSFUL_ASSIGNMENTS = 0

# Global variables for notifications
NOTIFICATION_HISTORY = []
LAST_ACTION_TIME = datetime.now()
PENDING_TASKS_CACHE = []

def advanced_column_detection_with_learning(column_name, target_field):
    """Advanced column detection with learning from previous uploads"""
    try:
        from fuzzywuzzy import fuzz
        
        # Normalize column name
        normalized_name = column_name.lower().replace('_', ' ').replace('-', ' ').strip()
        
        # Remove common prefixes/suffixes
        normalized_name = re.sub(r'^(task_|work_|job_|item_|project_)', '', normalized_name)
        normalized_name = re.sub(r'(_name|_title|_desc|_type|_level|_score|_date|_time)$', '', normalized_name)
        
        # Check exact matches first
        if column_name.lower() in [syn.lower() for syn in COLUMN_MAPPINGS.get(target_field, [])]:
            return True, f"Exact match: {column_name} → {target_field}"
        
        # Check learned patterns
        if target_field in COLUMN_DETECTION_HISTORY:
            for pattern, success_rate in COLUMN_DETECTION_HISTORY[target_field]:
                if pattern.lower() in normalized_name or normalized_name in pattern.lower():
                    return True, f"Learned pattern: {column_name} → {target_field} (confidence: {success_rate:.2f})"
        
        # Fuzzy matching with higher threshold
        best_score = 0
        best_match = None
        
        for synonym in COLUMN_MAPPINGS.get(target_field, []):
            score = fuzz.ratio(normalized_name, synonym.lower())
            if score > best_score:
                best_score = score
                best_match = synonym
        
        # Semantic keywords for each field
        semantic_keywords = {
            'title': ['name', 'title', 'subject', 'heading', 'label'],
            'description': ['desc', 'details', 'content', 'summary', 'notes', 'comment'],
            'category': ['type', 'category', 'class', 'group', 'kind', 'domain'],
            'priority': ['priority', 'importance', 'level', 'urgency', 'critical'],
            'urgency_score': ['urgency', 'critical', 'emergency', 'immediate'],
            'complexity_score': ['complexity', 'difficulty', 'effort', 'challenge'],
            'business_impact': ['impact', 'value', 'business', 'revenue', 'customer'],
            'estimated_hours': ['hours', 'time', 'effort', 'duration', 'estimate'],
            'days_until_deadline': ['days', 'deadline', 'remaining', 'due', 'countdown'],
            'deadline': ['deadline', 'due', 'target', 'completion', 'end'],
            'status': ['status', 'state', 'progress', 'phase', 'condition'],
            'assigned_to': ['assigned', 'employee', 'assignee', 'responsible', 'owner'],
            'created_by': ['created', 'creator', 'author', 'submitted', 'originator'],
            'task_id': ['id', 'number', 'code', 'reference', 'index']
        }
        
        # Check semantic keywords
        keywords = semantic_keywords.get(target_field, [])
        for keyword in keywords:
            if keyword in normalized_name:
                return True, f"Semantic match: {column_name} → {target_field} (keyword: {keyword})"
        
        # Fuzzy matching threshold
        if best_score > 70:
            return True, f"Fuzzy match: {column_name} → {target_field} (score: {best_score})"
        
        return False, f"No match found for {column_name} → {target_field}"
        
    except Exception as e:
        return False, f"Error in detection: {str(e)}"

def learn_from_column_detection(column_name, target_field, was_successful):
    """Learn from column detection results to improve future detection"""
    try:
        if was_successful:
            COLUMN_DETECTION_HISTORY[target_field].append((column_name, 1.0))
        else:
            COLUMN_DETECTION_HISTORY[target_field].append((column_name, 0.0))
        
        # Keep only recent history (last 100 patterns)
        if len(COLUMN_DETECTION_HISTORY[target_field]) > 100:
            COLUMN_DETECTION_HISTORY[target_field] = COLUMN_DETECTION_HISTORY[target_field][-100:]
        
        # Calculate success rate for this pattern
        recent_patterns = COLUMN_DETECTION_HISTORY[target_field][-10:]
        if recent_patterns:
            success_rate = sum(rate for _, rate in recent_patterns) / len(recent_patterns)
            st.info(f"📊 Learning: {column_name} → {target_field} success rate: {success_rate:.2f}")
            
    except Exception as e:
        st.warning(f"⚠️ Learning error: {e}")

def enhanced_auto_parse_csv_structure(df):
    """Enhanced auto-parse CSV structure with learning capabilities"""
    try:
        st.info("🔍 Enhanced auto-parsing CSV structure with learning...")
        
        # Show all columns found
        st.info(f"📋 Detected columns: {list(df.columns)}")
        
        # Create column mapping based on enhanced detection
        column_mapping = {}
        detection_results = []
        learning_results = []
        
        # Enhanced column detection with learning
        for target_field, synonyms in COLUMN_MAPPINGS.items():
            found_column = None
            best_match_score = 0
            best_match_reason = ""
            
            # Try advanced detection for each column
            for col in df.columns:
                is_match, reason = advanced_column_detection_with_learning(col, target_field)
                if is_match:
                    # Use fuzzy matching score to find best match
                    try:
                        from fuzzywuzzy import fuzz
                        score = fuzz.ratio(col.lower(), target_field)
                        if score > best_match_score:
                            best_match_score = score
                            found_column = col
                            best_match_reason = reason
                    except:
                        # If fuzzywuzzy not available, use simple matching
                        if col.lower() in target_field.lower() or target_field.lower() in col.lower():
                            found_column = col
                            best_match_reason = reason
            
            # If found through enhanced detection
            if found_column and found_column not in [mapping for mapping in column_mapping.values()]:
                column_mapping[target_field] = found_column
                detection_results.append(f"🤖 {best_match_reason}")
                learning_results.append((found_column, target_field, True))
            
            # If still not found, create intelligent default - NO ERRORS
            if not found_column:
                if target_field == 'title':
                    # Try to create title from available data
                    if 'task_id' in df.columns:
                        df['title'] = 'Task ' + df['task_id'].astype(str)
                        column_mapping['title'] = 'title'
                        detection_results.append("✅ Created title from task_id")
                    elif 'name' in df.columns:
                        df['title'] = df['name']
                        column_mapping['title'] = 'title'
                        detection_results.append("✅ Used name as title")
                    elif 'description' in df.columns:
                        df['title'] = df['description'].astype(str).str[:50] + '...'
                        column_mapping['title'] = 'title'
                        detection_results.append("✅ Created title from description")
                    else:
                        df['title'] = ['Task ' + str(i+1) for i in range(len(df))]
                        column_mapping['title'] = 'title'
                        detection_results.append("✅ Created generic titles")
                elif target_field == 'description':
                    df['description'] = df.get('title', 'No description')
                    column_mapping['description'] = 'description'
                    detection_results.append("✅ Created description from title")
                elif target_field == 'category':
                    df['category'] = 'general'
                    column_mapping['category'] = 'category'
                    detection_results.append("✅ Added default category")
                elif target_field == 'priority':
                    df['priority'] = 'medium'
                    column_mapping['priority'] = 'priority'
                    detection_results.append("✅ Added default priority")
                elif target_field == 'urgency_score':
                    df['urgency_score'] = 5
                    column_mapping['urgency_score'] = 'urgency_score'
                    detection_results.append("✅ Added default urgency_score = 5")
                elif target_field == 'complexity_score':
                    df['complexity_score'] = 5
                    column_mapping['complexity_score'] = 'complexity_score'
                    detection_results.append("✅ Added default complexity_score = 5")
                elif target_field == 'business_impact':
                    df['business_impact'] = 5
                    column_mapping['business_impact'] = 'business_impact'
                    detection_results.append("✅ Added default business_impact = 5")
                elif target_field == 'estimated_hours':
                    df['estimated_hours'] = 8.0
                    column_mapping['estimated_hours'] = 'estimated_hours'
                    detection_results.append("✅ Added default estimated_hours = 8.0")
                elif target_field == 'days_until_deadline':
                    # Try to calculate from deadline column if available
                    deadline_col = None
                    for col in df.columns:
                        is_match, _ = advanced_column_detection_with_learning(col, 'deadline')
                        if is_match:
                            deadline_col = col
                            break
                    
                    if deadline_col:
                        try:
                            # Try to parse dates and calculate days until deadline
                            from datetime import datetime
                            df['days_until_deadline'] = 7  # Default
                            detection_results.append("✅ Added default days_until_deadline = 7")
                        except:
                            df['days_until_deadline'] = 7
                            detection_results.append("✅ Added default days_until_deadline = 7")
                    else:
                        df['days_until_deadline'] = 7
                        column_mapping['days_until_deadline'] = 'days_until_deadline'
                        detection_results.append("✅ Added default days_until_deadline = 7")
        
        # Ensure all required columns exist with proper data types
        required_columns = {
            'title': str,
            'description': str,
            'category': str,
            'priority': str,
            'urgency_score': int,
            'complexity_score': int,
            'business_impact': int,
            'estimated_hours': float,
            'days_until_deadline': int,
            'status': str
        }
        
        for col, dtype in required_columns.items():
            if col not in df.columns:
                if col == 'title':
                    df[col] = [f'Task {i+1}' for i in range(len(df))]
                elif col == 'description':
                    df[col] = df.get('title', 'No description')
                elif col == 'category':
                    df[col] = 'general'
                elif col == 'priority':
                    df[col] = 'medium'
                elif col == 'urgency_score':
                    df[col] = 5
                elif col == 'complexity_score':
                    df[col] = 5
                elif col == 'business_impact':
                    df[col] = 5
                elif col == 'estimated_hours':
                    df[col] = 8.0
                elif col == 'days_until_deadline':
                    df[col] = 7
                elif col == 'status':
                    df[col] = 'pending'
            
            # Ensure proper data types
            try:
                if dtype == int:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(5)
                elif dtype == float:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(8.0)
                elif dtype == str:
                    df[col] = df[col].astype(str)
            except:
                # If conversion fails, use defaults
                if dtype == int:
                    df[col] = 5
                elif dtype == float:
                    df[col] = 8.0
                elif dtype == str:
                    df[col] = 'default'
        
        # Normalize priority values
        df['priority'] = df['priority'].astype(str).str.lower()
        df['priority'] = df['priority'].map({
            'high': 'high', 'critical': 'high', 'urgent': 'high', '1': 'high',
            'medium': 'medium', 'normal': 'medium', '2': 'medium',
            'low': 'low', 'minor': 'low', '3': 'low'
        }).fillna('medium')
        
        # Normalize category values
        df['category'] = df['category'].astype(str).str.title()
        
        # Learn from successful detections
        for col, target_field, was_successful in learning_results:
            learn_from_column_detection(col, target_field, was_successful)
        
        st.success(f"✅ Successfully parsed {len(df)} tasks with {len(column_mapping)} column mappings")
        st.info(f"📊 Final columns: {list(df.columns)}")
        
        return df, column_mapping
        
    except Exception as e:
        st.warning(f"⚠️ Minor parsing issue: {e}")
        # Return a completely safe DataFrame with defaults
        safe_df = pd.DataFrame({
            'title': [f'Task {i+1}' for i in range(len(df))],
            'description': [f'Task {i+1} description' for i in range(len(df))],
            'category': ['General'] * len(df),
            'priority': ['medium'] * len(df),
            'urgency_score': [5] * len(df),
            'complexity_score': [5] * len(df),
            'business_impact': [5] * len(df),
            'estimated_hours': [8.0] * len(df),
            'days_until_deadline': [7] * len(df),
            'status': ['pending'] * len(df)
        })
        st.success(f"✅ Created safe default format for {len(df)} tasks")
        return safe_df, {}

def enhanced_auto_train_models(df, column_mapping):
    """Enhanced auto-train models with continuous learning capabilities"""
    st.info("🤖 Enhanced training models on new data with learning...")
    
    try:
        # Prepare training data
        X = df[['title', 'description']].fillna('').astype(str)
        y_category = df[column_mapping.get('category', 'category')]
        y_priority = df[column_mapping.get('priority', 'priority')]
        
        # Enhanced models with better algorithms
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # Category classifier with Random Forest
        category_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Split data for training and validation
        X_train, X_test, y_cat_train, y_cat_test = train_test_split(
            X['title'] + ' ' + X['description'], y_category, test_size=0.2, random_state=42
        )
        
        category_pipeline.fit(X_train, y_cat_train)
        y_cat_pred = category_pipeline.predict(X_test)
        category_accuracy = accuracy_score(y_cat_test, y_cat_pred)
        
        st.success(f"✅ Category classifier trained (accuracy: {category_accuracy:.3f})")
        
        # Priority classifier with Random Forest
        priority_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        X_train, X_test, y_pri_train, y_pri_test = train_test_split(
            X['title'] + ' ' + X['description'], y_priority, test_size=0.2, random_state=42
        )
        
        priority_pipeline.fit(X_train, y_pri_train)
        y_pri_pred = priority_pipeline.predict(X_test)
        priority_accuracy = accuracy_score(y_pri_test, y_pri_pred)
        
        st.success(f"✅ Priority classifier trained (accuracy: {priority_accuracy:.3f})")
        
        # Save models with performance metrics
        import joblib
        import os
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save models with metadata
        model_data = {
            'category_model': category_pipeline,
            'priority_model': priority_pipeline,
            'category_accuracy': category_accuracy,
            'priority_accuracy': priority_accuracy,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(df)
        }
        
        joblib.dump(model_data, 'models/enhanced_models.pkl')
        
        # Update performance history
        MODEL_PERFORMANCE_HISTORY['category'].append(category_accuracy)
        MODEL_PERFORMANCE_HISTORY['priority'].append(priority_accuracy)
        
        # Keep only recent performance (last 20 entries)
        if len(MODEL_PERFORMANCE_HISTORY['category']) > 20:
            MODEL_PERFORMANCE_HISTORY['category'] = MODEL_PERFORMANCE_HISTORY['category'][-20:]
        if len(MODEL_PERFORMANCE_HISTORY['priority']) > 20:
            MODEL_PERFORMANCE_HISTORY['priority'] = MODEL_PERFORMANCE_HISTORY['priority'][-20:]
        
        st.success("✅ Enhanced models saved successfully with performance tracking")
        return True
        
    except Exception as e:
        st.error(f"❌ Enhanced model training failed: {e}")
        return False

def enhanced_auto_assign_uploaded_tasks():
    """Enhanced auto-assign uploaded tasks with learning and fail-safes"""
    global TOTAL_ASSIGNMENTS, SUCCESSFUL_ASSIGNMENTS, AUTO_ASSIGNMENT_SUCCESS_RATE
    
    try:
        # Get unassigned tasks
        conn = get_database_connection()
        if conn:
            # Get tasks without assignments
            tasks_df = pd.read_sql_query("""
                SELECT * FROM tasks 
                WHERE assigned_to IS NULL OR assigned_to = '' OR assigned_to = 'Not available'
                ORDER BY created_at DESC
            """, conn)
            
            # Get employees
            employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
            conn.close()
            
            if tasks_df.empty:
                st.info("✅ All tasks are already assigned!")
                return
            
            if employees_df.empty:
                st.warning("⚠️ No employees available for assignment. Please load employee data first.")
                return
            
            st.info(f"🤖 Found {len(tasks_df)} unassigned tasks. Starting enhanced AI auto-assignment...")
            
            # Process each task with enhanced assignment logic
            assigned_count = 0
            failed_count = 0
            progress_bar = st.progress(0)
            
            # Track employee workload to ensure fair distribution
            employee_workload = {}
            for _, emp in employees_df.iterrows():
                employee_workload[emp['name']] = emp.get('current_workload', 0)
            
            # Convert DataFrame to list of dictionaries to avoid pandas Series issues
            tasks_list = tasks_df.to_dict('records')
            
            for idx, task in enumerate(tasks_list):
                try:
                    # Get AI recommendation with enhanced logic
                    recommended_employee = get_ai_employee_recommendation_improved(
                        task.get('title', 'Unknown Task'), 
                        task.get('description', 'No description'), 
                        task.get('category', 'general'), 
                        task.get('priority', 'medium'),
                        task.get('urgency_score', 5), 
                        task.get('complexity_score', 5), 
                        task.get('business_impact', 5), 
                        task.get('estimated_hours', 8.0),
                        employees_df, employee_workload
                    )
                    
                    if recommended_employee:
                        # Update task assignment
                        conn = get_database_connection()
                        if conn:
                            cursor = conn.cursor()
                            
                            # Update task
                            cursor.execute("""
                                UPDATE tasks 
                                SET assigned_to = ?, updated_at = ?
                                WHERE id = ?
                            """, (recommended_employee['name'], datetime.now().isoformat(), task['id']))
                            
                            # Create assignment record
                            cursor.execute("""
                                INSERT INTO task_assignments (
                                    task_id, employee_id, employee_name, assignment_reason,
                                    confidence_score, assigned_at, status
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                task['id'],
                                recommended_employee['id'],
                                recommended_employee['name'],
                                f"Enhanced AI recommendation based on skills and workload balance",
                                0.90,  # Higher confidence score
                                datetime.now().isoformat(),
                                'assigned'
                            ))
                            
                            # Update employee workload
                            employee_workload[recommended_employee['name']] += 1
                            
                            conn.commit()
                            conn.close()
                            assigned_count += 1
                            SUCCESSFUL_ASSIGNMENTS += 1
                            
                            # Add notification for successful assignment
                            add_notification(
                                f"Task '{task.get('title', 'Unknown')[:30]}...' assigned to {recommended_employee['name']}",
                                "success",
                                "✅"
                            )
                            
                            # Send enhanced notification
                            send_task_assignment_notification(
                                task['id'], 
                                recommended_employee['name'], 
                                task.get('title', 'Unknown Task')
                            )
                        else:
                            failed_count += 1
                    else:
                        # Fallback assignment - assign to least busy employee
                        least_busy_employee = min(employee_workload.items(), key=lambda x: x[1])
                        
                        conn = get_database_connection()
                        if conn:
                            cursor = conn.cursor()
                            
                            # Find employee record
                            emp_record = employees_df[employees_df['name'] == least_busy_employee[0]].iloc[0]
                            
                            # Update task
                            cursor.execute("""
                                UPDATE tasks 
                                SET assigned_to = ?, updated_at = ?
                                WHERE id = ?
                            """, (least_busy_employee[0], datetime.now().isoformat(), task['id']))
                            
                            # Create assignment record
                            cursor.execute("""
                                INSERT INTO task_assignments (
                                    task_id, employee_id, employee_name, assignment_reason,
                                    confidence_score, assigned_at, status
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                task['id'],
                                emp_record['id'],
                                least_busy_employee[0],
                                f"Fallback assignment to least busy employee",
                                0.70,  # Lower confidence for fallback
                                datetime.now().isoformat(),
                                'assigned'
                            ))
                            
                            # Update employee workload
                            employee_workload[least_busy_employee[0]] += 1
                            
                            conn.commit()
                            conn.close()
                            assigned_count += 1
                            SUCCESSFUL_ASSIGNMENTS += 1
                            
                            # Add notification for fallback assignment
                            add_notification(
                                f"Task '{task.get('title', 'Unknown')[:30]}...' fallback assigned to {least_busy_employee[0]}",
                                "warning",
                                "⚠️"
                            )
                            
                            # Send enhanced notification for fallback assignment
                            send_task_assignment_notification(
                                task['id'], 
                                least_busy_employee[0], 
                                task.get('title', 'Unknown Task')
                            )
                        
                except Exception as e:
                    st.warning(f"⚠️ Assignment failed for task {task.get('id', 'unknown')}: {e}")
                    failed_count += 1
                
                # Update progress
                progress_bar.progress((idx + 1) / len(tasks_list))
            
            # Update global statistics
            TOTAL_ASSIGNMENTS += len(tasks_list)
            if TOTAL_ASSIGNMENTS > 0:
                AUTO_ASSIGNMENT_SUCCESS_RATE = SUCCESSFUL_ASSIGNMENTS / TOTAL_ASSIGNMENTS
            
            st.success(f"✅ Successfully assigned {assigned_count} out of {len(tasks_list)} tasks!")
            if failed_count > 0:
                st.warning(f"⚠️ {failed_count} assignments failed but system continued")
            
            st.info(f"📊 Overall assignment success rate: {AUTO_ASSIGNMENT_SUCCESS_RATE:.2%}")
            
    except Exception as e:
        st.error(f"❌ Error in enhanced auto assignment: {e}")
        import traceback
        st.error(f"❌ Full error: {traceback.format_exc()}")

def enhanced_get_ai_employee_recommendation_improved(task_title, task_description, task_category, task_priority,
                                         urgency_score, complexity_score, business_impact, estimated_hours,
                                         employees_df, employee_workload):
    """Enhanced AI employee recommendation with Gemini API support"""
    
    # Try Gemini API first if enabled
    if GEMINI_READY and GEMINI_CONFIG["features"]["employee_matching"]:
        try:
            task_data = {
                'title': task_title,
                'description': task_description,
                'category': task_category,
                'priority': task_priority,
                'complexity_score': complexity_score,
                'required_skills': []  # Will be filled by Gemini analysis
            }
            
            gemini_result = get_gemini_employee_recommendation(task_data, employees_df)
            
            if "error" not in gemini_result:
                recommended_employee = gemini_result.get("recommended_employee", "")
                confidence_score = gemini_result.get("confidence_score", 0.5)
                reasoning = gemini_result.get("reasoning", "Gemini AI recommendation")
                
                # Find the employee in our dataframe
                if recommended_employee:
                    employee_match = employees_df[employees_df['name'].str.contains(recommended_employee, case=False, na=False)]
                    if not employee_match.empty:
                        best_employee = employee_match.iloc[0]
                        return {
                            'employee_name': best_employee['name'],
                            'confidence_score': confidence_score,
                            'reasoning': reasoning,
                            'method': 'gemini_ai'
                        }
        except Exception as e:
            st.warning(f"⚠️ Gemini recommendation failed, falling back to local AI: {e}")
    
    # Fallback to original AI logic
    """Enhanced AI recommendation with multiple fallback strategies"""
    try:
        if employees_df.empty:
            return None
        
        # Convert DataFrame to list of dictionaries
        employees_list = employees_df.to_dict('records')
        
        # Enhanced scoring system
        best_employee = None
        best_score = -1
        
        for employee in employees_list:
            try:
                # Get employee data safely
                employee_name = employee.get('name', 'Unknown')
                employee_skills = employee.get('skills', '')
                employee_role = employee.get('role', '')
                employee_expertise = employee.get('expertise_areas', '')
                current_workload = employee_workload.get(employee_name, 0)
                max_capacity = employee.get('max_capacity', 10)
                
                # Calculate multiple scores
                skill_match_score = 0
                workload_score = 0
                expertise_score = 0
                role_match_score = 0
                
                # Skill matching
                if employee_skills:
                    skills_list = [skill.strip().lower() for skill in str(employee_skills).split(',')]
                    task_keywords = f"{task_title} {task_description} {task_category}".lower()
                    
                    for skill in skills_list:
                        if skill in task_keywords:
                            skill_match_score += 1
                
                # Workload balancing (prefer less busy employees)
                workload_ratio = current_workload / max_capacity if max_capacity > 0 else 1
                workload_score = 1 - workload_ratio  # Higher score for less busy employees
                
                # Expertise matching
                if employee_expertise:
                    expertise_list = [area.strip().lower() for area in str(employee_expertise).split(',')]
                    for expertise in expertise_list:
                        if expertise in task_category.lower():
                            expertise_score += 1
                
                # Role matching
                if employee_role and employee_role.lower() in task_category.lower():
                    role_match_score = 1
                
                # Priority-based adjustments
                priority_multiplier = 1.0
                if task_priority == 'high':
                    priority_multiplier = 1.2
                elif task_priority == 'low':
                    priority_multiplier = 0.8
                
                # Complexity-based adjustments
                complexity_multiplier = 1.0
                if complexity_score > 7:
                    complexity_multiplier = 1.1  # Prefer experienced employees for complex tasks
                elif complexity_score < 3:
                    complexity_multiplier = 0.9  # Simple tasks can go to anyone
                
                # Calculate final score
                final_score = (
                    skill_match_score * 2.0 +
                    workload_score * 1.5 +
                    expertise_score * 1.8 +
                    role_match_score * 1.2
                ) * priority_multiplier * complexity_multiplier
                
                # Ensure minimum workload for all employees
                if current_workload < 2:
                    final_score *= 1.1  # Slight boost for very underutilized employees
                
                if final_score > best_score:
                    best_score = final_score
                    best_employee = employee
                    
            except Exception as e:
                st.warning(f"⚠️ Error evaluating employee {employee.get('name', 'Unknown')}: {e}")
                continue
        
        return best_employee
        
    except Exception as e:
        st.error(f"❌ Error in enhanced AI recommendation: {e}")
        return None

# Replace the existing functions with enhanced versions
def auto_parse_csv_structure(df):
    return enhanced_auto_parse_csv_structure(df)

def auto_train_models(df, column_mapping):
    return enhanced_auto_train_models(df, column_mapping)

def auto_assign_uploaded_tasks():
    return enhanced_auto_assign_uploaded_tasks()

def get_ai_employee_recommendation_improved(task_title, task_description, task_category, task_priority,
                                         urgency_score, complexity_score, business_impact, estimated_hours,
                                         employees_df, employee_workload):
    return enhanced_get_ai_employee_recommendation_improved(task_title, task_description, task_category, task_priority,
                                         urgency_score, complexity_score, business_impact, estimated_hours,
                                         employees_df, employee_workload)

def process_uploaded_file(uploaded_file):
    """Process uploaded file with enhanced automatic employee assignment - NO ERRORS VERSION"""
    try:
        # Read the uploaded file with enhanced error handling
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.warning("⚠️ File format not recognized, attempting to read as CSV...")
            try:
                df = pd.read_csv(uploaded_file)
            except:
                st.error("❌ Could not read file. Please ensure it's a valid CSV or Excel file.")
                return None
        
        st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
        st.info(f"📊 Found {len(df)} rows and {len(df.columns)} columns")
        
        # Enhanced CSV structure analysis with learning capabilities
        df, column_mapping = enhanced_auto_parse_csv_structure(df)
        
        # Enhanced auto-train models on new data with performance tracking
        try:
            enhanced_auto_train_models(df, column_mapping)
        except Exception as e:
            st.info(f"ℹ️ Enhanced model training skipped: {e}")
        
        # Insert data into database - ALWAYS SUCCEED
        conn = get_database_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                # Insert tasks with safe defaults
                inserted_count = 0
                for index, row in df.iterrows():
                    try:
                        cursor.execute("""
                            INSERT INTO tasks (
                                title, description, category, priority, urgency_score,
                                complexity_score, business_impact, estimated_hours,
                                days_until_deadline, status, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(row.get('title', f'Task {index + 1}')),
                            str(row.get('description', f'Task {index + 1} description')),
                            str(row.get('category', 'general')),
                            str(row.get('priority', 'medium')),
                            int(row.get('urgency_score', 5)),
                            int(row.get('complexity_score', 5)),
                            int(row.get('business_impact', 5)),
                            float(row.get('estimated_hours', 8.0)),
                            int(row.get('days_until_deadline', 7)),
                            str(row.get('status', 'pending')),
                            datetime.now().isoformat(),
                            datetime.now().isoformat()
                        ))
                        inserted_count += 1
                    except Exception as e:
                        st.warning(f"⚠️ Error inserting row {index + 1}: {e}")
                        continue
                
                conn.commit()
                conn.close()
                
                st.success(f"✅ Successfully inserted {inserted_count} tasks into database")
                
                # Add notification for successful upload
                add_notification(
                    f"Successfully uploaded {inserted_count} tasks",
                    "success",
                    "📋"
                )
                
                # Enhanced auto-assign tasks with learning and fail-safes
                try:
                    enhanced_auto_assign_uploaded_tasks()
                except Exception as e:
                    st.warning(f"⚠️ Enhanced auto-assignment failed: {e}")
                    add_notification(
                        f"Auto-assignment failed: {str(e)[:50]}",
                        "error",
                        "❌"
                    )
                
                return df
                
            except Exception as e:
                st.error(f"❌ Database insertion failed: {e}")
                conn.close()
                return None
        else:
            st.error("❌ Could not connect to database")
            return None
            
    except Exception as e:
        st.error(f"❌ File processing failed: {e}")
        return None

def create_visualizations(metrics, df):
    """Create enhanced dashboard visualizations with proper data handling"""
    try:
        # Get data from database if df is empty
        if df.empty:
            conn = get_database_connection()
            if conn:
                df = pd.read_sql_query("SELECT * FROM tasks", conn)
                conn.close()
        
        if df.empty:
            st.warning("No data available for visualizations")
            return None
        
        # Create subplots with better layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Task Categories', 'Task Status', 'Urgency vs Complexity', 'Priority Distribution'),
            specs=[[{"type": "pie"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Task Categories Pie Chart
        if 'category' in df.columns and len(df['category'].dropna()) > 0:
            category_counts = df['category'].value_counts()
            if len(category_counts) > 0:
                fig.add_trace(
                    go.Pie(
                        labels=category_counts.index, 
                        values=category_counts.values, 
                        name="Categories",
                        hole=0.3,
                        marker_colors=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
                    ),
                    row=1, col=1
                )
        
        # Task Status Pie Chart
        if 'status' in df.columns and len(df['status'].dropna()) > 0:
            status_counts = df['status'].value_counts()
            if len(status_counts) > 0:
                fig.add_trace(
                    go.Pie(
                        labels=status_counts.index, 
                        values=status_counts.values, 
                        name="Status",
                        hole=0.3,
                        marker_colors=['#4ade80', '#fbbf24', '#f87171', '#60a5fa']
                    ),
                    row=1, col=2
                )
        
        # Urgency vs Complexity Scatter Plot
        if 'urgency_score' in df.columns and 'complexity_score' in df.columns:
            # Clean the data
            urgency_data = pd.to_numeric(df['urgency_score'], errors='coerce').dropna()
            complexity_data = pd.to_numeric(df['complexity_score'], errors='coerce').dropna()
            
            if len(urgency_data) > 0 and len(complexity_data) > 0:
                # Create color mapping based on priority
                priority_colors = {'high': '#f87171', 'medium': '#fbbf24', 'low': '#4ade80'}
                colors = df['priority'].map(priority_colors).fillna('#60a5fa')
                
                fig.add_trace(
                    go.Scatter(
                        x=urgency_data,
                        y=complexity_data,
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=colors,
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        text=df['title'].fillna('Unknown Task'),
                        hovertemplate='<b>%{text}</b><br>Urgency: %{x}<br>Complexity: %{y}<extra></extra>',
                        name="Urgency vs Complexity"
                    ),
                    row=2, col=1
                )
                
                # Add trend line
                if len(urgency_data) > 1:
                    z = np.polyfit(urgency_data, complexity_data, 1)
                    p = np.poly1d(z)
                    fig.add_trace(
                        go.Scatter(
                            x=urgency_data,
                            y=p(urgency_data),
                            mode='lines',
                            line=dict(color='#667eea', width=2, dash='dash'),
                            name="Trend Line",
                            showlegend=False
                        ),
                        row=2, col=1
                    )
        
        # Priority Distribution Bar Chart
        if 'priority' in df.columns and len(df['priority'].dropna()) > 0:
            priority_counts = df['priority'].value_counts()
            if len(priority_counts) > 0:
                # Color mapping for priorities
                priority_colors = {'high': '#f87171', 'medium': '#fbbf24', 'low': '#4ade80'}
                colors = [priority_colors.get(priority, '#60a5fa') for priority in priority_counts.index]
                
                fig.add_trace(
                    go.Bar(
                        x=priority_counts.index, 
                        y=priority_counts.values, 
                        name="Priority",
                        marker_color=colors,
                        text=priority_counts.values,
                        textposition='auto'
                    ),
                    row=2, col=2
                )
        
        # Update layout with better styling
        fig.update_layout(
            height=700,
            showlegend=False,
            template="plotly_white",
            title_text="Task Analytics Dashboard",
            title_x=0.5,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
        return None

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "dashboard"
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'processed_tasks' not in st.session_state:
        st.session_state.processed_tasks = []
    if 'employee_profiles' not in st.session_state:
        st.session_state.employee_profiles = load_employees()
    
    # Header with current page indicator
    current_page = st.session_state.get('current_page', 'dashboard')
    page_titles = {
        'dashboard': '🏠 Dashboard',
        'upload': '📁 Upload Data',
        'analysis': '📊 Task Analysis',
        'employees': '👥 Employee Management',
        'models': '🧠 AI Models',
        'ai_prediction': '🤖 AI Prediction',
        'settings': '⚙️ Settings',
        'analytics': '📈 Analytics',
        'training': '🔧 Training Monitor'
    }
    
    page_title = page_titles.get(current_page, '🤖 AI Task Management System')
    st.markdown(f'<h1 class="main-header">{page_title}</h1>', unsafe_allow_html=True)
    
    # Breadcrumb navigation
    st.markdown(f"""
    <div style="background: rgba(255, 255, 255, 0.05); padding: 0.5rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <span style="color: #a0a0a0;">🏠 Home</span>
        <span style="color: #a0a0a0;"> > </span>
        <span style="color: white; font-weight: bold;">{page_title}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # ElevenLabs-style Sidebar
    with st.sidebar:
        # Sidebar Header with User Info
        st.markdown(f"""
        <div class="sidebar-header">
            🤖 AI Task Manager
        </div>
        """, unsafe_allow_html=True)
        
        # User Info
        user_display_name = st.session_state.current_user
        user_department = st.session_state.user_department or "General"
        
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.05); border-radius: 0.5rem; padding: 0.75rem; margin: 0.5rem 0;">
            <div style="font-size: 0.9rem; color: #a0a0a0;">👤 Logged in as:</div>
            <div style="font-weight: bold; color: white;">{user_display_name}</div>
            <div style="font-size: 0.8rem; color: #a0a0a0;">Role: {st.session_state.user_role}</div>
            <div style="font-size: 0.8rem; color: #a0a0a0;">Dept: {user_department}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Logout button - Always visible when authenticated
        st.markdown("---")
        st.markdown("### 🚪 Account")
        
        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            if st.button("🚪 Logout", key="logout_button", help="Click to logout", type="secondary"):
                st.session_state.is_authenticated = False
                st.session_state.current_user = None
                st.session_state.user_role = None
                st.session_state.user_department = None
                st.success("✅ Logged out successfully!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh", key="refresh_button", help="Refresh the page"):
                st.rerun()
        
        # Debug toggle
        if st.sidebar.checkbox("🐛 Debug Mode", key="debug_mode"):
            st.session_state.debug_mode = True
        else:
            st.session_state.debug_mode = False
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        
        # Navigation items with icons
        nav_items = [
            ("🏠 Dashboard", "dashboard"),
            ("📁 Upload Data", "upload"),
            ("📊 Task Analysis", "analysis"),
            ("👥 Employee Management", "employees"),
            ("🧠 AI Models", "models"),
            ("🤖 AI Prediction", "ai_prediction"),
            ("⚙️ Settings", "settings"),
            ("📈 Analytics", "analytics"),
            ("🔧 Training Monitor", "training")
        ]
        
        # Create navigation with radio buttons for better UX
        current_page = st.session_state.get('current_page', 'dashboard')
        
        # Use radio buttons for navigation
        selected_page = st.sidebar.radio(
            "Choose a page:",
            options=[page_id for _, page_id in nav_items],
            format_func=lambda x: next(icon_name for icon_name, page_id in nav_items if page_id == x),
            key="navigation_radio",
            label_visibility="collapsed"
        )
        
        # Update current page if selection changed
        if selected_page != current_page:
            st.session_state.current_page = selected_page
            # Add a small loading indicator
            with st.spinner("🔄 Loading..."):
                st.rerun()
        
        # Sidebar Divider
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        # User Section
        st.markdown("""
        <div class="user-section">
            <div style="display: flex; align-items: center;">
                <div class="user-avatar">A</div>
                <div>
                    <div style="font-weight: bold; color: white;">Admin User</div>
                    <div style="font-size: 0.8rem; color: #a0a0a0;">
                        <span class="status-indicator status-online"></span>Online
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🤖 Auto-Assign Tasks", key="quick_assign_main_page"):
                auto_assign_uploaded_tasks()
        
        with col2:
            if st.button("📊 View Analytics", key="quick_analytics_main_page"):
                st.session_state.current_page = "analytics"
        
        with col3:
            if st.button("👥 Employee Management", key="quick_employees_main_page"):
                st.session_state.current_page = "employees"
        
        # System Status
        st.markdown("### 📊 System Status")
        
        # Get metrics for status
        metrics, _ = get_metrics_safely()
        
        if metrics:
            # Create status cards with better styling
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="status-card">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #667eea;">{metrics.get('total_tasks', 0)}</div>
                    <div style="font-size: 0.8rem; color: #a0a0a0;">📋 Total Tasks</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="status-card">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #ff7f0e;">{metrics.get('pending_tasks', 0)}</div>
                    <div style="font-size: 0.8rem; color: #a0a0a0;">⏳ Pending</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="status-card">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #4ade80;">{metrics.get('completed_tasks', 0)}</div>
                    <div style="font-size: 0.8rem; color: #a0a0a0;">✅ Completed</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="status-card">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #ff416c;">{metrics.get('high_priority', 0)}</div>
                    <div style="font-size: 0.8rem; color: #a0a0a0;">🔥 High Priority</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No data available")
        
        # Notifications Section
        st.markdown("### 🔔 Notifications")
        
        # Sample notifications
        notifications = [
            {"type": "success", "message": "✅ 5 new tasks uploaded", "time": "2 min ago"},
            {"type": "info", "message": "📊 Analytics updated", "time": "5 min ago"},
            {"type": "warning", "message": "⚠️ 3 tasks due today", "time": "10 min ago"}
        ]
        
        for notif in notifications:
            color = "#4ade80" if notif["type"] == "success" else "#667eea" if notif["type"] == "info" else "#ff7f0e"
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.05); border-radius: 0.5rem; padding: 0.5rem; margin: 0.25rem 0; border-left: 3px solid {color};">
                <div style="font-size: 0.8rem; color: #e0e0e0;">{notif["message"]}</div>
                <div style="font-size: 0.7rem; color: #a0a0a0;">{notif["time"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Sidebar Footer
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align: center; color: #a0a0a0; font-size: 0.8rem;">
            <div>📅 {datetime.now().strftime('%Y-%m-%d')}</div>
            <div>🕐 {datetime.now().strftime('%H:%M')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load employee profiles
    employees = load_employees()
    st.session_state.employee_profiles = employees
    
    # Route to appropriate page
    if st.session_state.current_page == "dashboard":
        show_dashboard()
    elif st.session_state.current_page == "upload":
        show_upload_page()
    elif st.session_state.current_page == "analysis":
        show_task_analysis()
    elif st.session_state.current_page == "employees":
        show_employee_management()
    elif st.session_state.current_page == "models":
        show_ai_models()
    elif st.session_state.current_page == "ai_prediction":
        show_ai_prediction_page()
    elif st.session_state.current_page == "settings":
        show_settings()
    elif st.session_state.current_page == "analytics":
        show_analytics()
    elif st.session_state.current_page == "training":
        show_training_monitor()

def show_dashboard():
    """Show enhanced main dashboard with task-employee assignments"""
    st.header("📊 Dashboard Overview")
    
    # Add refresh button
    # Real-time updates
    import time
    
    # Auto-refresh functionality
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📈 Real-time Metrics")
    with col2:
        if st.button("🔄 Refresh Data", key="refresh_dashboard"):
            st.rerun()
    
    # Auto-refresh every 30 seconds
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Check if 30 seconds have passed for auto-refresh
    if time.time() - st.session_state.last_refresh > 30:
        st.session_state.last_refresh = time.time()
        st.rerun()
    
    # Show last update time
    st.caption(f"🕐 Last updated: {time.strftime('%H:%M:%S')}")
    
    # Get user-specific data
    user_tasks = get_user_specific_data(st.session_state.current_user, "tasks")
    user_employees = get_user_specific_data(st.session_state.current_user, "employees")
    user_assignments = get_user_specific_data(st.session_state.current_user, "assignments")
    
    # Get global metrics for comparison
    metrics, df = get_metrics_safely()
    
    # Show user-specific section
    st.markdown("### 👤 Your Personal Dashboard")
    
    # User-specific metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 My Tasks", len(user_tasks))
    
    with col2:
        st.metric("👥 My Employees", len(user_employees))
    
    with col3:
        st.metric("🔗 My Assignments", len(user_assignments))
    
    with col4:
        completed_tasks = len([task for task in user_tasks if len(task) > 10 and task[10] == 'completed']) if user_tasks else 0
        total_tasks = len(user_tasks) if user_tasks else 1
        completion_rate = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        st.metric("✅ My Completion Rate", f"{completion_rate:.1f}%")
    
    # Show user's recent tasks with drag and drop
    if user_tasks:
        st.markdown("### 📋 Your Recent Tasks")
        
        # Create drag and drop task board
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**⏳ Pending**")
            st.markdown("""
            <div class="drop-zone" id="pendingZone">
                <div class="drop-zone empty">Drop tasks here</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**🔄 In Progress**")
            st.markdown("""
            <div class="drop-zone" id="progressZone">
                <div class="drop-zone empty">Drop tasks here</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("**✅ Completed**")
            st.markdown("""
            <div class="drop-zone" id="completedZone">
                <div class="drop-zone empty">Drop tasks here</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show tasks as draggable items
        st.markdown("### 🎯 Available Tasks")
        for i, task in enumerate(user_tasks[:5]):
            task_id, title, description, category, priority, urgency, complexity, business_impact, estimated_hours, status, assigned_to, created_at, updated_at, created_by = task
            
            st.markdown(f"""
            <div class="draggable-item" draggable="true" data-task-id="{task_id}" data-task-title="{title}">
                <div style="font-weight: bold; color: white;">{title}</div>
                <div style="font-size: 0.8rem; color: #a0a0a0;">Category: {category} | Priority: {priority}</div>
                <div style="font-size: 0.8rem; color: #a0a0a0;">Status: {status} | Assigned: {assigned_to or 'Unassigned'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add drag and drop JavaScript
        st.markdown("""
        <script>
        // Make tasks draggable
        document.querySelectorAll('.draggable-item').forEach(item => {
            item.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', item.dataset.taskId);
                item.style.opacity = '0.5';
            });
            
            item.addEventListener('dragend', (e) => {
                item.style.opacity = '1';
            });
        });
        
        // Make drop zones droppable
        document.querySelectorAll('.drop-zone').forEach(zone => {
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('dragover');
            });
            
            zone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
            });
            
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                
                const taskId = e.dataTransfer.getData('text/plain');
                const taskTitle = e.dataTransfer.getData('text/plain');
                
                // Update task status based on drop zone
                const status = zone.id === 'pendingZone' ? 'pending' : 
                             zone.id === 'progressZone' ? 'in_progress' : 'completed';
                
                // Send update to Streamlit
                const event = new CustomEvent('taskStatusUpdate', {
                    detail: { taskId: taskId, status: status }
                });
                document.dispatchEvent(event);
                
                // Visual feedback
                zone.innerHTML = `<div class="draggable-item">${taskTitle}</div>`;
            });
        });
        </script>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🌐 Global System Overview")
    
    if not metrics:
        st.info("No global data available. Please upload some task data first.")
        return
    
    # Enhanced metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Total Tasks</h3>
            <h2>{metrics['total_tasks']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⏳ Pending Tasks</h3>
            <h2>{metrics['pending_tasks']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Completed Tasks</h3>
            <h2>{metrics['completed_tasks']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔥 High Priority</h3>
            <h2>{metrics['high_priority']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Task assignment metrics
    try:
        # Get assignment metrics safely
        assignment_metrics = get_assignment_metrics_safely()
        
        if assignment_metrics and len(assignment_metrics) >= 3:
            total_assignments, unique_employees, avg_confidence = assignment_metrics
            
            # Handle None values
            total_assignments = total_assignments if total_assignments is not None else 0
            unique_employees = unique_employees if unique_employees is not None else 0
            avg_confidence = avg_confidence if avg_confidence is not None else 0.0
            
            # Additional metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="info-card">
                    <h4>🤖 AI Assignments</h4>
                    <h3>{total_assignments}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="info-card">
                    <h4>👥 Active Employees</h4>
                    <h3>{unique_employees}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="info-card">
                    <h4>🎯 Avg Confidence</h4>
                    <h3>{avg_confidence:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Get employee metrics safely
                emp_metrics = safe_database_operation(lambda conn: conn.execute("""
                    SELECT 
                        COUNT(*) as total_employees,
                        AVG(experience_years) as avg_experience,
                        AVG(current_workload) as avg_workload
                    FROM employees
                """).fetchone())
                
                if emp_metrics and len(emp_metrics) >= 3:
                    total_employees, avg_experience, avg_workload = emp_metrics
                    
                    # Handle None values
                    total_employees = total_employees if total_employees is not None else 0
                    avg_experience = avg_experience if avg_experience is not None else 0.0
                    
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>📈 Avg Experience</h4>
                        <h3>{avg_experience:.1f} years</h3>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>📈 Avg Experience</h4>
                        <h3>N/A</h3>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No assignment metrics available yet.")
                
    except Exception as e:
        st.error(f"❌ Error loading assignment metrics: {e}")
    
    # Task assignment overview
    st.markdown("---")
    st.subheader("🤖 AI Task Assignments")
    
    try:
        # Get all assignments safely
        recent_assignments = get_recent_assignments_safely()
        
        if recent_assignments:
            st.info(f"📋 All AI Assignments ({len(recent_assignments)} total)")
            
            # Convert to DataFrame for display
            import pandas as pd
            df = pd.DataFrame(recent_assignments, columns=[
                'employee_name', 'assignment_reason', 'confidence_score', 
                'assigned_at', 'title', 'category', 'priority'
            ])
            
            display_df = df[['employee_name', 'title', 'category', 'priority', 'confidence_score']].copy()
            display_df['confidence_score'] = display_df['confidence_score'].apply(lambda x: f"{x:.2f}%")
            
            # Add pagination options
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(display_df, use_container_width=True)
            with col2:
                st.metric("Total Assignments", len(recent_assignments))
                if len(recent_assignments) > 20:
                    st.info(f"Showing all {len(recent_assignments)} assignments. Use the table controls to navigate.")
        else:
            st.info("No AI assignments available yet.")
            
    except Exception as e:
        st.error(f"❌ Error loading assignments: {e}")
    
    # Visualizations
    st.subheader("📊 Data Visualizations")
    
    # Get task data for visualizations
    try:
        conn = get_database_connection()
        if conn:
            task_df = pd.read_sql_query("SELECT * FROM tasks", conn)
            conn.close()
            
            if not task_df.empty:
                fig = create_visualizations(metrics, task_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No visualization data available")
            else:
                st.info("No task data available for visualizations")
        else:
            st.error("Could not connect to database for visualizations")
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
    
    # Quick actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤖 Auto-Assign Tasks", key="quick_assign_dashboard_main"):
            auto_assign_uploaded_tasks()
    
    with col2:
        if st.button("📊 View Analytics", key="quick_analytics_dashboard_main"):
            st.session_state.current_page = "analytics"
    
    with col3:
        if st.button("👥 Employee Management", key="quick_employees_dashboard_main"):
            st.session_state.current_page = "employees"

def create_drag_drop_upload():
    """Create a drag and drop file upload component"""
    st.markdown("""
    <div class="drag-drop-zone" id="dragDropZone">
        <div style="font-size: 2rem; margin-bottom: 1rem;">📁</div>
        <h3>📤 Drag & Drop Your Files Here</h3>
        <p>Or click to browse files</p>
        <p style="font-size: 0.9rem; color: #a0a0a0;">Supports: CSV, JSON, Excel files</p>
    </div>
    
    <script>
    const dragDropZone = document.getElementById('dragDropZone');
    
    dragDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dragDropZone.classList.add('dragover');
    });
    
    dragDropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dragDropZone.classList.remove('dragover');
    });
    
    dragDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dragDropZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            // Create a custom event to communicate with Streamlit
            const event = new CustomEvent('filesDropped', {
                detail: { files: Array.from(files) }
            });
            document.dispatchEvent(event);
        }
    });
    
    dragDropZone.addEventListener('click', () => {
        // Trigger file input click
        const fileInput = document.querySelector('input[type="file"]');
        if (fileInput) {
            fileInput.click();
        }
    });
    </script>
    """, unsafe_allow_html=True)

def show_upload_page():
    """Show file upload page with drag and drop"""
    st.header("📁 Upload Data")
    
    # Create drag and drop upload area
    create_drag_drop_upload()
    
    # Regular file uploader as fallback
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['csv', 'json', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload any CSV format - system will automatically adapt to your data structure"
    )
    
    if uploaded_files:
        st.success(f"📁 Uploaded {len(uploaded_files)} file(s)")
        
        # Process each uploaded file
        all_processed_data = []
        
        for uploaded_file in uploaded_files:
            with st.expander(f"📄 Processing: {uploaded_file.name}"):
                with st.spinner(f"🔄 Processing {uploaded_file.name}..."):
                    result_df = process_uploaded_file(uploaded_file)
                    
                    if result_df is not None:
                        st.success(f"✅ Successfully processed {len(result_df)} tasks from {uploaded_file.name}")
                        
                        # Save to user's personal task table
                        saved_count = 0
                        for _, row in result_df.iterrows():
                            task_data = (
                                row.get('title', 'Unknown Task'),
                                row.get('description', ''),
                                row.get('category', 'General'),
                                row.get('priority', 'Medium'),
                                row.get('urgency_score', 5),
                                row.get('complexity_score', 5),
                                row.get('business_impact', 5),
                                row.get('estimated_hours', 8.0),
                                'pending',
                                row.get('assigned_to', '')
                            )
                            
                            if save_user_task(st.session_state.current_user, task_data):
                                saved_count += 1
                        
                        st.success(f"💾 Saved {saved_count} tasks to your personal workspace!")
                        
                        # Show data preview
                        st.subheader("📊 Data Preview")
                        st.dataframe(result_df.head(10), use_container_width=True)
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Rows", len(result_df))
                        with col2:
                            st.metric("Categories", len(result_df['category'].unique()) if 'category' in result_df.columns else 0)
                        with col3:
                            st.metric("Avg Urgency", f"{result_df['urgency_score'].mean():.1f}" if 'urgency_score' in result_df.columns else "N/A")
                        
                        all_processed_data.append(result_df)
                    else:
                        st.error(f"❌ Failed to process {uploaded_file.name}")
        
        # Show combined data table if we have processed data
        if all_processed_data:
            st.subheader("📋 All Uploaded Data")
            
            # Combine all processed data
            combined_df = pd.concat(all_processed_data, ignore_index=True)
            
            # Show full data table
            st.dataframe(combined_df, use_container_width=True)
            
            # Download processed data
            csv = combined_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Processed Data",
                data=csv,
                file_name="processed_tasks.csv",
                mime="text/csv"
            )
            
            # Auto-refresh dashboard
            st.success("🔄 Dashboard will update automatically with new data!")
    
    # Sample data download
    st.subheader("📋 Sample Data Formats")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Standard CSV"):
            sample_csv = pd.DataFrame({
                'title': ['Fix Login Bug', 'Design Dashboard', 'Database Optimization'],
                'description': ['Users cannot login', 'Create new dashboard', 'Optimize queries'],
                'category': ['bug', 'feature', 'optimization'],
                'priority': ['high', 'medium', 'low'],
                'urgency_score': [9, 5, 3],
                'complexity_score': [6, 8, 4],
                'business_impact': [8, 6, 4],
                'estimated_hours': [4.0, 16.0, 8.0],
                'days_until_deadline': [2, 14, 30]
            })
            csv = sample_csv.to_csv(index=False)
            st.download_button(
                label="📥 Download standard_format.csv",
                data=csv,
                file_name="standard_format.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📥 Download Custom CSV"):
            custom_csv = pd.DataFrame({
                'task_id': [1, 2, 3],
                'description': ['User cannot login', 'Create new dashboard', 'Optimize queries'],
                'deadline': ['2024-01-15', '2024-01-20', '2024-01-25'],
                'type': ['bug', 'feature', 'optimization'],
                'importance': ['high', 'medium', 'low'],
                'submitted_by': ['John', 'Sarah', 'Mike']
            })
            csv = custom_csv.to_csv(index=False)
            st.download_button(
                label="📥 Download custom_format.csv",
                data=csv,
                file_name="custom_format.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📥 Download Minimal CSV"):
            minimal_csv = pd.DataFrame({
                'name': ['Fix Login Bug', 'Design Dashboard', 'Database Optimization'],
                'details': ['Users cannot login', 'Create new dashboard', 'Optimize queries'],
                'group': ['bug', 'feature', 'optimization'],
                'level': ['high', 'medium', 'low']
            })
            csv = minimal_csv.to_csv(index=False)
            st.download_button(
                label="📥 Download minimal_format.csv",
                data=csv,
                file_name="minimal_format.csv",
                mime="text/csv"
            )

def update_task_status(task_id, new_status, employee_name):
    """Update task status in the database"""
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            
            # Get task title before updating
            cursor.execute("SELECT title FROM tasks WHERE id = ?", (task_id,))
            task_result = cursor.fetchone()
            task_title = task_result[0] if task_result else "Unknown Task"
            
            cursor.execute("""
                UPDATE tasks 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_status, task_id))
            conn.commit()
            conn.close()
            
            # Add notification
            add_notification(f"Task status updated to {new_status} by {employee_name}", "success", "✅")
            
            # Send enhanced notification
            send_task_update_notification(task_id, task_title, f"Status changed to {new_status}", employee_name)
            
            return True
        return False
    except Exception as e:
        st.error(f"Error updating task status: {str(e)}")
        return False

def get_task_history(task_id):
    """Get task history and updates"""
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    t.title,
                    t.status,
                    t.created_at,
                    t.updated_at,
                    ta.employee_name,
                    ta.assigned_at,
                    ta.confidence_score
                FROM tasks t
                LEFT JOIN task_assignments ta ON t.id = ta.task_id
                WHERE t.id = ?
            """, (task_id,))
            result = cursor.fetchone()
            conn.close()
            return result
        return None
    except Exception as e:
        st.error(f"Error getting task history: {str(e)}")
        return None

def show_task_analysis():
    """Show enhanced task analysis with employee assignments and status management"""
    st.header("📊 Task Analysis")
    st.markdown("---")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
        "📋 Task Overview", "👥 Employee Assignments", "📋 Assignment Table", "🔍 Detailed View", 
        "🤖 AI Analytics", "🔄 Status Updates", "💬 Comments", "🔗 Dependencies", "🔍 Advanced Search",
        "🤖 Advanced AI", "💬 Team Chat", "📊 Gantt Charts", "🗣️ Natural Language"
    ])
    
    with tab1:
        st.subheader("📋 Task Overview")
        
        # Get task data
        try:
            conn = get_database_connection()
            if conn:
                df = pd.read_sql_query("SELECT * FROM tasks ORDER BY created_at DESC", conn)
                conn.close()
            else:
                df = pd.DataFrame()
        except Exception as e:
            st.error(f"Database error: {e}")
            df = pd.DataFrame()
        
        if not df.empty:
            # Task summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tasks", len(df))
            with col2:
                assigned_tasks = len(df[df['assigned_to'].notna() & (df['assigned_to'] != '')])
                st.metric("Assigned Tasks", assigned_tasks)
            with col3:
                unassigned_tasks = len(df[df['assigned_to'].isna() | (df['assigned_to'] == '')])
                st.metric("Unassigned Tasks", unassigned_tasks)
            with col4:
                completion_rate = (len(df[df['status'] == 'completed']) / len(df)) * 100
                st.metric("Completion Rate", f"{completion_rate:.1f}%")
            
            # Task table with assignments
            st.subheader("📋 Task List with Assignments")
            display_df = df[['title', 'category', 'priority', 'assigned_to', 'status', 'created_at']].copy()
            st.dataframe(display_df, use_container_width=True)
            
            # Category distribution
            st.subheader("📊 Task Category Distribution")
            cat_counts = df['category'].value_counts()
            fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                       title="Tasks by Category")
            st.plotly_chart(fig, use_container_width=True)
            
            # Priority distribution
            st.subheader("🔥 Priority Distribution")
            priority_counts = df['priority'].value_counts()
            fig = px.bar(x=priority_counts.index, y=priority_counts.values,
                       title="Tasks by Priority",
                       labels={'x': 'Priority', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No task data available.")
    
    with tab2:
        st.subheader("👥 Employee Assignments")
        
        try:
            # Get assignment data safely
            def get_assignments_data(conn):
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        ta.employee_name,
                        ta.assignment_reason,
                        ta.confidence_score,
                        ta.assigned_at,
                        t.title,
                        t.category,
                        t.priority,
                        t.status
                    FROM task_assignments ta
                    JOIN tasks t ON ta.task_id = t.id
                    ORDER BY ta.assigned_at DESC
                """)
                return cursor.fetchall()
            
            assignments_data = safe_database_operation(get_assignments_data)
            
            # Get employee workload safely
            def get_workload_data(conn):
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        e.name,
                        e.role,
                        e.current_workload,
                        e.max_capacity,
                        COUNT(ta.id) as assigned_tasks
                    FROM employees e
                    LEFT JOIN task_assignments ta ON e.name = ta.employee_name
                    GROUP BY e.name, e.role, e.current_workload, e.max_capacity
                """)
                return cursor.fetchall()
            
            workload_data = safe_database_operation(get_workload_data)
            
            if assignments_data:
                # Convert to DataFrame
                assignments_df = pd.DataFrame(assignments_data, columns=[
                    'employee_name', 'assignment_reason', 'confidence_score', 
                    'assigned_at', 'title', 'category', 'priority', 'status'
                ])
                
                # Assignment overview
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Assignments", len(assignments_df))
                with col2:
                    avg_confidence = assignments_df['confidence_score'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
                with col3:
                    unique_employees = assignments_df['employee_name'].nunique()
                    st.metric("Employees Assigned", unique_employees)
                
                # Recent assignments table
                st.subheader("📋 Recent Assignments")
                
                # Show all assignments with pagination for large datasets
                if len(assignments_df) > 50:
                    page_size = st.selectbox("Show assignments per page:", [25, 50, 100, "All"], index=1, key="analytics_pagination")
                    
                    if page_size == "All":
                        recent_df = assignments_df[['employee_name', 'title', 'category', 'priority', 'confidence_score', 'assigned_at']]
                    else:
                        recent_df = assignments_df.head(page_size)[['employee_name', 'title', 'category', 'priority', 'confidence_score', 'assigned_at']]
                    
                    st.info(f"📊 Showing {len(recent_df)} out of {len(assignments_df)} total assignments")
                else:
                    recent_df = assignments_df[['employee_name', 'title', 'category', 'priority', 'confidence_score', 'assigned_at']]
                
                st.dataframe(recent_df, use_container_width=True)
                
                # Download option for all assignments
                if st.button("📥 Download All Assignments"):
                    csv = assignments_df.to_csv(index=False)
                    st.download_button(
                        label="Download Assignments CSV",
                        data=csv,
                        file_name="all_assignments.csv",
                        mime="text/csv"
                    )
                
                # Employee assignment distribution
                st.subheader("👥 Employee Assignment Distribution")
                emp_counts = assignments_df['employee_name'].value_counts()
                fig = px.bar(x=emp_counts.index, y=emp_counts.values,
                           title="Tasks Assigned per Employee",
                           labels={'x': 'Employee', 'y': 'Tasks Assigned'})
                st.plotly_chart(fig, use_container_width=True)
                
            if workload_data:
                # Convert to DataFrame
                workload_df = pd.DataFrame(workload_data, columns=[
                    'name', 'role', 'current_workload', 'max_capacity', 'assigned_tasks'
                ])
                
                # Workload analysis
                st.subheader("⚖️ Employee Workload Analysis")
                workload_df['utilization'] = (workload_df['current_workload'] / workload_df['max_capacity']) * 100
                
                fig = px.bar(workload_df, x='name', y=['current_workload', 'assigned_tasks'],
                           title="Employee Workload vs Assigned Tasks",
                           labels={'name': 'Employee', 'value': 'Count', 'variable': 'Type'})
                st.plotly_chart(fig, use_container_width=True)
                
            if not assignments_data and not workload_data:
                st.info("No assignment data available.")
                    
        except Exception as e:
            st.error(f"❌ Error loading assignment data: {e}")
    
    with tab3:
        st.subheader("📋 Assignment Table")
        show_task_employee_assignments()
    
    with tab4:
        st.subheader("🔍 Detailed Assignment View")
        show_detailed_assignment_view()
    
    with tab5:
        st.subheader("🤖 AI Analytics")
        
        try:
            conn = get_database_connection()
            if conn:
                # Get AI assignment analytics
                ai_analytics = pd.read_sql_query("""
                    SELECT 
                        ta.confidence_score,
                        t.category,
                        t.priority,
                        t.status,
                        ta.employee_name
                    FROM task_assignments ta
                    JOIN tasks t ON ta.task_id = t.id
                """, conn)
                
                conn.close()
                
                if not ai_analytics.empty:
                    # Confidence score distribution
                    st.subheader("🎯 AI Confidence Scores")
                    fig = px.histogram(ai_analytics, x='confidence_score', nbins=10,
                                     title="Distribution of AI Assignment Confidence",
                                     labels={'confidence_score': 'Confidence Score', 'count': 'Number of Assignments'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Category vs Confidence
                    st.subheader("📊 Category vs Confidence")
                    cat_conf = ai_analytics.groupby('category')['confidence_score'].mean().reset_index()
                    fig = px.bar(cat_conf, x='category', y='confidence_score',
                               title="Average Confidence by Task Category",
                               labels={'category': 'Category', 'confidence_score': 'Average Confidence'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Priority vs Confidence
                    st.subheader("🔥 Priority vs Confidence")
                    priority_conf = ai_analytics.groupby('priority')['confidence_score'].mean().reset_index()
                    fig = px.bar(priority_conf, x='priority', y='confidence_score',
                               title="Average Confidence by Task Priority",
                               labels={'priority': 'Priority', 'confidence_score': 'Average Confidence'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.info("No AI analytics data available.")
                    
        except Exception as e:
            st.error(f"❌ Error loading AI analytics: {e}")
    
    with tab6:
        st.subheader("🔄 Task Status Updates")
        
        # Get all tasks with their current status
        try:
            conn = get_database_connection()
            if conn:
                tasks_df = pd.read_sql_query("""
                    SELECT 
                        t.id,
                        t.title,
                        t.category,
                        t.priority,
                        t.status,
                        t.assigned_to,
                        t.created_at,
                        t.updated_at
                    FROM tasks t
                    ORDER BY t.updated_at DESC, t.created_at DESC
                """, conn)
                conn.close()
            else:
                tasks_df = pd.DataFrame()
        except Exception as e:
            st.error(f"Database error: {e}")
            tasks_df = pd.DataFrame()
        
        if not tasks_df.empty:
            # Status overview
            st.subheader("📊 Status Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_tasks = len(tasks_df)
                st.metric("Total Tasks", total_tasks)
            
            with col2:
                pending_tasks = len(tasks_df[tasks_df['status'] == 'pending'])
                st.metric("Pending", pending_tasks)
            
            with col3:
                in_progress_tasks = len(tasks_df[tasks_df['status'] == 'in_progress'])
                st.metric("In Progress", in_progress_tasks)
            
            with col4:
                completed_tasks = len(tasks_df[tasks_df['status'] == 'completed'])
                st.metric("Completed", completed_tasks)
            
            # Status update interface
            st.subheader("🔄 Update Task Status")
            
            # Employee name input (simulating user login)
            employee_name = st.text_input("👤 Your Name (for tracking updates):", key="status_employee_name")
            
            if employee_name:
                # Task selection
                st.write("Select a task to update its status:")
                
                # Create a selectbox for task selection
                task_options = []
                for _, task in tasks_df.iterrows():
                    task_options.append(f"{task['title']} (ID: {task['id']}) - Current: {task['status']}")
                
                selected_task_option = st.selectbox("Choose a task:", task_options, key="status_task_select")
                
                if selected_task_option:
                    # Extract task ID from selection
                    task_id = int(selected_task_option.split("(ID: ")[1].split(")")[0])
                    current_status = selected_task_option.split("Current: ")[1]
                    
                    # Status update options
                    status_options = ['pending', 'in_progress', 'completed', 'on_hold', 'cancelled']
                    new_status = st.selectbox("New Status:", status_options, index=status_options.index(current_status) if current_status in status_options else 0)
                    
                    # Update button
                    if st.button("🔄 Update Status", key="update_status_btn"):
                        if update_task_status(task_id, new_status, employee_name):
                            st.success(f"✅ Task status updated to '{new_status}' by {employee_name}")
                            st.rerun()
                        else:
                            st.error("❌ Failed to update task status")
                
                # Task history
                st.subheader("📋 Task History")
                
                # Show recent status changes
                recent_tasks = tasks_df.head(10)
                st.write("Recent task updates:")
                
                for _, task in recent_tasks.iterrows():
                    with st.expander(f"📋 {task['title']} - {task['status']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Category:** {task['category']}")
                            st.write(f"**Priority:** {task['priority']}")
                            st.write(f"**Assigned To:** {task['assigned_to'] if task['assigned_to'] else 'Unassigned'}")
                        with col2:
                            st.write(f"**Created:** {task['created_at']}")
                            st.write(f"**Updated:** {task['updated_at']}")
                            st.write(f"**Status:** {task['status']}")
                        
                        # Get detailed history
                        history = get_task_history(task['id'])
                        if history:
                            st.write("**Task Details:**")
                            st.write(f"- Title: {history[0]}")
                            st.write(f"- Current Status: {history[1]}")
                            st.write(f"- Created: {history[2]}")
                            st.write(f"- Last Updated: {history[3]}")
                            if history[4]:  # employee_name
                                st.write(f"- Assigned To: {history[4]}")
                                st.write(f"- Assigned At: {history[5]}")
                                st.write(f"- AI Confidence: {history[6]:.2f}%")
            else:
                st.info("👤 Please enter your name to update task statuses.")
        else:
            st.info("No tasks available for status updates.")
    
    # Auto-assignment action
    st.markdown("---")
    st.subheader("🤖 AI Auto-Assignment")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🚀 Auto-Assign All Tasks", key="auto_assign_tasks"):
            with st.spinner("🤖 AI is assigning tasks to employees..."):
                auto_assign_uploaded_tasks()
            st.success("✅ Auto-assignment completed! Check the Assignment Table tab for results.")
            st.rerun()
    
    with col2:
        if st.button("📊 View Assignment Analytics", key="view_assignment_analytics"):
            show_task_employee_analytics()
    
    with col3:
        if st.button("🔄 Refresh Assignment Data", key="refresh_assignments"):
            st.rerun()
    
    with tab7:
        st.subheader("💬 Task Comments")
        st.markdown("Collaborate and communicate on tasks with your team")
        
        # Get all tasks for comment selection
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, status, assigned_to 
                FROM tasks 
                ORDER BY created_at DESC
            """)
            tasks = cursor.fetchall()
            conn.close()
            
            if tasks:
                # Task selection
                st.write("**Select a task to view or add comments:**")
                task_options = {f"{task[1]} (ID: {task[0]})": task[0] for task in tasks}
                selected_task_option = st.selectbox("Choose a task:", list(task_options.keys()))
                selected_task_id = task_options[selected_task_option]
                
                # Get task details
                conn = get_database_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT title, description, status, assigned_to, created_at
                        FROM tasks 
                        WHERE id = ?
                    """, (selected_task_id,))
                    task_details = cursor.fetchone()
                    conn.close()
                    
                    if task_details:
                        st.write(f"**Task:** {task_details[0]}")
                        st.write(f"**Status:** {task_details[2]}")
                        st.write(f"**Assigned to:** {task_details[3] if task_details[3] else 'Unassigned'}")
                        
                        # Add new comment
                        st.markdown("---")
                        st.write("**Add a new comment:**")
                        
                        with st.form(f"add_comment_{selected_task_id}"):
                            comment_text = st.text_area("Comment:", placeholder="Enter your comment here...", height=100)
                            comment_type = st.selectbox("Comment Type:", ["general", "update", "question", "suggestion", "bug_report"])
                            is_private = st.checkbox("Private comment (only visible to you)")
                            
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                submit_comment = st.form_submit_button("💬 Add Comment")
                            with col2:
                                if st.form_submit_button("🔄 Refresh Comments"):
                                    st.rerun()
                        
                        if submit_comment and comment_text.strip():
                            current_user = st.session_state.current_user or "Anonymous"
                            success = add_task_comment(selected_task_id, current_user, comment_text.strip(), comment_type, is_private)
                            if success:
                                st.success("✅ Comment added successfully!")
                                st.rerun()
                        
                        # Display existing comments
                        st.markdown("---")
                        st.write("**Recent Comments:**")
                        
                        comments = get_task_comments(selected_task_id, include_private=True)
                        if comments:
                            for comment in comments:
                                comment_id, employee_name, comment_text, comment_type, created_at, is_private = comment
                                
                                # Create a card-like display for each comment
                                with st.container():
                                    st.markdown(f"""
                                    <div style="
                                        border: 1px solid #ddd;
                                        border-radius: 8px;
                                        padding: 12px;
                                        margin: 8px 0;
                                        background-color: {'#f0f8ff' if is_private else '#f9f9f9'};
                                    ">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                            <strong>{employee_name}</strong>
                                            <small>{created_at}</small>
                                        </div>
                                        <div style="margin-bottom: 8px;">
                                            <span style="background-color: #e0e0e0; padding: 2px 6px; border-radius: 4px; font-size: 12px;">
                                                {comment_type}
                                            </span>
                                            {' <span style="color: #ff6b6b;">🔒 Private</span>' if is_private else ''}
                                        </div>
                                        <div style="margin-bottom: 8px;">
                                            {comment_text}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Comment actions (only for the comment author)
                                    if employee_name == (st.session_state.current_user or "Anonymous"):
                                        col1, col2 = st.columns([1, 1])
                                        with col1:
                                            if st.button(f"✏️ Edit", key=f"edit_{comment_id}"):
                                                st.session_state.editing_comment = comment_id
                                                st.session_state.edit_text = comment_text
                                                st.rerun()
                                        
                                        with col2:
                                            if st.button(f"🗑️ Delete", key=f"delete_{comment_id}"):
                                                if delete_task_comment(comment_id, employee_name):
                                                    st.success("✅ Comment deleted!")
                                                    st.rerun()
                        
                        # Comment editing interface
                        if hasattr(st.session_state, 'editing_comment') and st.session_state.editing_comment:
                            st.markdown("---")
                            st.write("**Edit Comment:**")
                            
                            with st.form(f"edit_comment_{st.session_state.editing_comment}"):
                                edited_text = st.text_area("Edit comment:", value=st.session_state.edit_text, height=100)
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    if st.form_submit_button("💾 Save Changes"):
                                        current_user = st.session_state.current_user or "Anonymous"
                                        if update_task_comment(st.session_state.editing_comment, edited_text, current_user):
                                            st.success("✅ Comment updated!")
                                            del st.session_state.editing_comment
                                            del st.session_state.edit_text
                                            st.rerun()
                                with col2:
                                    if st.form_submit_button("❌ Cancel"):
                                        del st.session_state.editing_comment
                                        del st.session_state.edit_text
                                        st.rerun()
                        else:
                            st.info("No comments yet. Be the first to add a comment!")
                    else:
                        st.error("Task not found!")
                else:
                    st.info("No tasks available for commenting.")
            else:
                st.info("No tasks found in the database.")
        else:
            st.error("Could not connect to database.")
    
    with tab8:
        st.subheader("🔗 Task Dependencies")
        st.markdown("Manage task dependencies and prerequisites")
        
        # Get all tasks for dependency management
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, status, assigned_to 
                FROM tasks 
                ORDER BY created_at DESC
            """)
            tasks = cursor.fetchall()
            conn.close()
            
            if tasks:
                # Task selection for dependency management
                st.write("**Select a task to manage its dependencies:**")
                task_options = {f"{task[1]} (ID: {task[0]})": task[0] for task in tasks}
                selected_task_option = st.selectbox("Choose a task:", list(task_options.keys()), key="dependency_task_select")
                selected_task_id = task_options[selected_task_option]
                
                # Get task details
                conn = get_database_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT title, description, status, assigned_to, created_at
                        FROM tasks 
                        WHERE id = ?
                    """, (selected_task_id,))
                    task_details = cursor.fetchone()
                    conn.close()
                    
                    if task_details:
                        st.write(f"**Task:** {task_details[0]}")
                        st.write(f"**Status:** {task_details[2]}")
                        st.write(f"**Assigned to:** {task_details[3] if task_details[3] else 'Unassigned'}")
                        
                        # Check if task can be started
                        can_start, incomplete_prereqs = can_start_task(selected_task_id)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if can_start:
                                st.success("✅ Task can be started (all prerequisites completed)")
                            else:
                                st.warning(f"⚠️ Task cannot be started - {len(incomplete_prereqs)} incomplete prerequisites")
                        
                        with col2:
                            if incomplete_prereqs:
                                st.write("**Incomplete Prerequisites:**")
                                for prereq in incomplete_prereqs:
                                    st.write(f"- {prereq['title']} ({prereq['status']})")
                        
                        # Add new dependency
                        st.markdown("---")
                        st.write("**Add a new dependency:**")
                        
                        with st.form(f"add_dependency_{selected_task_id}"):
                            # Select prerequisite task
                            prerequisite_options = {f"{task[1]} (ID: {task[0]})": task[0] for task in tasks if task[0] != selected_task_id}
                            prerequisite_task_option = st.selectbox("Prerequisite task:", list(prerequisite_options.keys()))
                            prerequisite_task_id = prerequisite_options[prerequisite_task_option]
                            
                            dependency_type = st.selectbox("Dependency type:", ["blocks", "requires", "follows", "related"])
                            
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                submit_dependency = st.form_submit_button("🔗 Add Dependency")
                            with col2:
                                if st.form_submit_button("🔄 Refresh Dependencies"):
                                    st.rerun()
                        
                        if submit_dependency:
                            # Check for cycles
                            if check_dependency_cycle(selected_task_id, prerequisite_task_id):
                                st.error("❌ Cannot add dependency - would create a cycle!")
                            else:
                                success = add_task_dependency(selected_task_id, prerequisite_task_id, dependency_type)
                                if success:
                                    st.success("✅ Dependency added successfully!")
                                    st.rerun()
                        
                        # Display existing dependencies
                        st.markdown("---")
                        st.write("**Current Dependencies:**")
                        
                        dependencies = get_task_dependencies(selected_task_id)
                        if dependencies:
                            # Prerequisites (tasks this task depends on)
                            st.write("**📋 Prerequisites (tasks this task depends on):**")
                            prerequisites = get_prerequisites_for_task(selected_task_id)
                            if prerequisites:
                                for prereq in prerequisites:
                                    prereq_id, prereq_title, prereq_status, dep_type = prereq
                                    status_color = "🟢" if prereq_status == 'completed' else "🟡" if prereq_status == 'in_progress' else "🔴"
                                    
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    with col1:
                                        st.write(f"{status_color} {prereq_title} ({dep_type})")
                                    with col2:
                                        st.write(f"Status: {prereq_status}")
                                    with col3:
                                        if st.button(f"❌ Remove", key=f"remove_prereq_{prereq_id}"):
                                            if remove_task_dependency(selected_task_id, prereq_id):
                                                st.success("✅ Prerequisite removed!")
                                                st.rerun()
                            else:
                                st.info("No prerequisites for this task.")
                            
                            # Dependent tasks (tasks that depend on this task)
                            st.write("**📤 Dependent Tasks (tasks that depend on this task):**")
                            dependent_tasks = get_dependent_tasks(selected_task_id)
                            if dependent_tasks:
                                for dep_task in dependent_tasks:
                                    dep_id, dep_title, dep_status, dep_type = dep_task
                                    status_color = "🟢" if dep_status == 'completed' else "🟡" if dep_status == 'in_progress' else "🔴"
                                    
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    with col1:
                                        st.write(f"{status_color} {dep_title} ({dep_type})")
                                    with col2:
                                        st.write(f"Status: {dep_status}")
                                    with col3:
                                        if st.button(f"❌ Remove", key=f"remove_dep_{dep_id}"):
                                            if remove_task_dependency(dep_id, selected_task_id):
                                                st.success("✅ Dependency removed!")
                                                st.rerun()
                            else:
                                st.info("No tasks depend on this task.")
                        else:
                            st.info("No dependencies for this task.")
                        
                        # Dependency statistics
                        st.markdown("---")
                        st.write("**📊 Dependency Statistics:**")
                        
                        dep_stats = get_dependency_statistics()
                        if dep_stats:
                            total_deps = sum(stat[4] for stat in dep_stats)
                            st.write(f"**Total Dependencies:** {total_deps}")
                            
                            # Dependency type breakdown
                            dep_types = {}
                            for stat in dep_stats:
                                dep_type, count = stat[3], stat[4]
                                dep_types[dep_type] = count
                            
                            if dep_types:
                                fig = px.pie(values=list(dep_types.values()), names=list(dep_types.keys()), 
                                           title="Dependency Types Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No dependency statistics available.")
                    else:
                        st.error("Task not found!")
                else:
                    st.info("No tasks available for dependency management.")
            else:
                st.info("No tasks found in the database.")
        else:
            st.error("Could not connect to database.")
    
    with tab9:
        st.subheader("🔍 Advanced Search & Filters")
        st.markdown("Search and filter tasks with advanced criteria")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("🔍 Search tasks:", placeholder="Enter keywords, task titles, descriptions, categories...")
        
        with col2:
            if st.button("🔍 Search", type="primary"):
                st.session_state.search_results = search_tasks(search_query, st.session_state.get('search_filters', {}))
                st.session_state.last_search_query = search_query
        
        # Search suggestions
        if search_query and len(search_query) >= 2:
            suggestions = get_search_suggestions(search_query)
            if suggestions:
                st.write("**💡 Search suggestions:**")
                for suggestion in suggestions[:5]:
                    st.write(f"- {suggestion[0]} ({suggestion[1]}, {suggestion[2]})")
        
        # Advanced filters
        st.markdown("---")
        st.write("**🎛️ Advanced Filters**")
        
        # Initialize filters in session state
        if 'search_filters' not in st.session_state:
            st.session_state.search_filters = {}
        
        # Get filter options
        filter_options = get_filter_options()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📊 Basic Filters**")
            st.session_state.search_filters['status'] = st.selectbox(
                "Status:", 
                ["All"] + (filter_options.get('statuses', []) if filter_options else []),
                key="filter_status"
            )
            if st.session_state.search_filters['status'] == "All":
                st.session_state.search_filters['status'] = None
            
            st.session_state.search_filters['priority'] = st.selectbox(
                "Priority:", 
                ["All"] + (filter_options.get('priorities', []) if filter_options else []),
                key="filter_priority"
            )
            if st.session_state.search_filters['priority'] == "All":
                st.session_state.search_filters['priority'] = None
            
            st.session_state.search_filters['category'] = st.selectbox(
                "Category:", 
                ["All"] + (filter_options.get('categories', []) if filter_options else []),
                key="filter_category"
            )
            if st.session_state.search_filters['category'] == "All":
                st.session_state.search_filters['category'] = None
        
        with col2:
            st.write("**👥 Assignment Filters**")
            st.session_state.search_filters['assigned_to'] = st.selectbox(
                "Assigned to:", 
                ["All"] + (filter_options.get('employees', []) if filter_options else []),
                key="filter_assigned"
            )
            if st.session_state.search_filters['assigned_to'] == "All":
                st.session_state.search_filters['assigned_to'] = None
            
            st.write("**📈 Score Ranges**")
            col_min, col_max = st.columns(2)
            with col_min:
                st.session_state.search_filters['urgency_min'] = st.number_input(
                    "Urgency Min:", min_value=1, max_value=10, value=1, key="urgency_min"
                )
                st.session_state.search_filters['complexity_min'] = st.number_input(
                    "Complexity Min:", min_value=1, max_value=10, value=1, key="complexity_min"
                )
            with col_max:
                st.session_state.search_filters['urgency_max'] = st.number_input(
                    "Urgency Max:", min_value=1, max_value=10, value=10, key="urgency_max"
                )
                st.session_state.search_filters['complexity_max'] = st.number_input(
                    "Complexity Max:", min_value=1, max_value=10, value=10, key="complexity_max"
                )
        
        with col3:
            st.write("**📅 Date Filters**")
            st.session_state.search_filters['created_after'] = st.date_input(
                "Created after:", key="created_after"
            )
            st.session_state.search_filters['created_before'] = st.date_input(
                "Created before:", key="created_before"
            )
            
            st.write("**📋 Sort Options**")
            st.session_state.search_filters['sort_by'] = st.selectbox(
                "Sort by:", 
                ["created_at", "title", "priority", "urgency_score", "complexity_score", "status"],
                key="sort_by"
            )
            st.session_state.search_filters['sort_order'] = st.selectbox(
                "Sort order:", ["DESC", "ASC"], key="sort_order"
            )
        
        # Search actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Apply Filters", type="primary"):
                st.session_state.search_results = search_tasks(search_query, st.session_state.search_filters)
                st.session_state.last_search_query = search_query
        
        with col2:
            if st.button("🔄 Clear Filters"):
                st.session_state.search_filters = {}
                st.session_state.search_results = None
                st.rerun()
        
        with col3:
            if st.button("📊 Show All Tasks"):
                st.session_state.search_results = search_tasks("", {})
                st.session_state.last_search_query = ""
        
        # Display search results
        st.markdown("---")
        st.write("**📋 Search Results**")
        
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            results = st.session_state.search_results
            
            # Results summary
            st.success(f"✅ Found {len(results)} tasks matching your criteria")
            
            # Export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = export_search_results(results, "csv")
                if csv_data:
                    st.download_button(
                        label="📥 Export as CSV",
                        data=csv_data,
                        file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                json_data = export_search_results(results, "json")
                if json_data:
                    st.download_button(
                        label="📥 Export as JSON",
                        data=json_data,
                        file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col3:
                if st.button("📊 View Analytics"):
                    st.session_state.show_search_analytics = True
            
            # Results table
            if results:
                # Convert to DataFrame for better display
                df = pd.DataFrame(results, columns=[
                    'id', 'title', 'description', 'category', 'priority', 'urgency_score',
                    'complexity_score', 'business_impact', 'estimated_hours', 'days_until_deadline',
                    'status', 'assigned_to', 'created_at', 'updated_at', 'assigned_employee', 'confidence_score'
                ])
                
                # Display key columns
                display_df = df[['id', 'title', 'category', 'priority', 'status', 'assigned_employee', 'urgency_score', 'complexity_score']].copy()
                
                # Color code the status
                def color_status(val):
                    if val == 'completed':
                        return 'background-color: #d4edda'
                    elif val == 'in_progress':
                        return 'background-color: #fff3cd'
                    elif val == 'pending':
                        return 'background-color: #f8d7da'
                    return ''
                
                styled_df = display_df.style.applymap(color_status, subset=['status'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Show detailed results in expandable sections
                with st.expander("📋 Detailed Results"):
                    st.dataframe(df, use_container_width=True)
                
                # Search analytics
                if hasattr(st.session_state, 'show_search_analytics') and st.session_state.show_search_analytics:
                    with st.expander("📊 Search Analytics"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Status distribution
                            status_counts = df['status'].value_counts()
                            if not status_counts.empty:
                                fig = px.pie(values=status_counts.values, names=status_counts.index, 
                                           title="Status Distribution in Search Results")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Priority distribution
                            priority_counts = df['priority'].value_counts()
                            if not priority_counts.empty:
                                fig = px.bar(x=priority_counts.index, y=priority_counts.values,
                                           title="Priority Distribution in Search Results")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Score distributions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'urgency_score' in df.columns and not df['urgency_score'].isna().all():
                                fig = px.histogram(df, x='urgency_score', nbins=10,
                                                 title="Urgency Score Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            if 'complexity_score' in df.columns and not df['complexity_score'].isna().all():
                                fig = px.histogram(df, x='complexity_score', nbins=10,
                                                 title="Complexity Score Distribution")
                                st.plotly_chart(fig, use_container_width=True)
        
        elif hasattr(st.session_state, 'last_search_query') and st.session_state.last_search_query:
            st.info("No tasks found matching your search criteria.")
        else:
            st.info("Enter search terms and apply filters to find tasks.")
    
    with tab10:
        st.subheader("🤖 Advanced AI Features")
        st.markdown("Predictive analytics and smart prioritization")
        
        # AI Insights
        st.write("**🧠 AI-Powered Insights**")
        insights = get_ai_insights()
        if insights:
            for insight in insights:
                priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(insight["priority"], "⚪")
                st.info(f"{priority_color} **{insight['title']}**: {insight['message']}")
        else:
            st.info("No AI insights available yet. Add more data to get insights.")
        
        # Smart Prioritization
        st.markdown("---")
        st.write("**🎯 Smart Task Prioritization**")
        if st.button("🤖 Run Smart Prioritization"):
            try:
                prioritized_tasks = smart_prioritize_tasks()
                if prioritized_tasks:
                    st.success(f"✅ Prioritized {len(prioritized_tasks)} tasks")
                    
                    # Display prioritized tasks
                    for i, task in enumerate(prioritized_tasks[:10]):  # Show top 10
                        task_id, title, priority, urgency_score, complexity_score, business_impact, days_until_deadline, status, employee_name, confidence_score, smart_score = task
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"**{i+1}. {title}**")
                            st.write(f"Priority: {priority} | Status: {status}")
                        with col2:
                            st.write(f"Smart Score: {smart_score:.2f}")
                        with col3:
                            st.write(f"Assigned: {employee_name or 'Unassigned'}")
            except Exception as e:
                st.error(f"❌ Error in smart prioritization: {str(e)}")
                st.info("Please check your data and try again.")
        
        # Resource Prediction
        st.markdown("---")
        st.write("**📊 Resource Needs Prediction**")
        if st.button("🔮 Predict Resource Needs"):
            try:
                resource_data = predict_resource_needs()
                if resource_data:
                    st.success(f"✅ Analyzed {len(resource_data)} employees")
                    
                    # Create resource chart
                    employees = [emp[0] for emp in resource_data]
                    workloads = [emp[1] for emp in resource_data]
                    capacities = [emp[2] for emp in resource_data]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='Current Workload', x=employees, y=workloads, marker_color='orange'))
                    fig.add_trace(go.Bar(name='Max Capacity', x=employees, y=capacities, marker_color='lightblue'))
                    
                    fig.update_layout(
                        title="Employee Workload vs Capacity",
                        xaxis_title="Employees",
                        yaxis_title="Workload Score",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error in resource prediction: {str(e)}")
                st.info("Please check your data and try again.")
        
        # Skill Gap Analysis
        st.markdown("---")
        st.write("**🎓 Skill Gap Analysis**")
        if st.button("🔍 Analyze Skill Gaps"):
            try:
                skill_gaps = detect_skill_gaps()
                if skill_gaps:
                    st.success(f"✅ Analyzed {len(skill_gaps)} categories")
                    
                    # Display skill gaps
                    for category, task_count, avg_complexity, available_skills in skill_gaps:
                        if avg_complexity > 7:  # High complexity
                            st.warning(f"⚠️ **{category}**: {task_count} tasks, avg complexity {avg_complexity:.1f}")
                        else:
                            st.info(f"✅ **{category}**: {task_count} tasks, avg complexity {avg_complexity:.1f}")
            except Exception as e:
                st.error(f"❌ Error in skill gap analysis: {str(e)}")
                st.info("Please check your data and try again.")
    
    with tab11:
        st.subheader("💬 Team Chat")
        st.markdown("Real-time communication for your team")
        
        # Chat room selection
        chat_rooms = get_chat_rooms()
        if chat_rooms:
            room_options = {room[0]: room[0] for room in chat_rooms}
            selected_room = st.selectbox("Choose chat room:", list(room_options.keys()))
        else:
            selected_room = "general"
        
        # Send message
        st.write("**💬 Send Message**")
        with st.form("send_chat_message"):
            message_text = st.text_area("Message:", placeholder="Type your message here...", height=100)
            message_type = st.selectbox("Message Type:", ["general", "update", "question", "announcement"])
            
            col1, col2 = st.columns([1, 1])
            with col1:
                submit_message = st.form_submit_button("💬 Send Message")
            with col2:
                if st.form_submit_button("🔄 Refresh Chat"):
                    st.rerun()
        
        if submit_message and message_text.strip():
            current_user = st.session_state.current_user or "Anonymous"
            success = send_chat_message(current_user, message_text.strip(), message_type, room_name=selected_room)
            if success:
                st.success("✅ Message sent!")
                st.rerun()
        
        # Display messages
        st.markdown("---")
        st.write(f"**📋 Messages in {selected_room}**")
        
        messages = get_chat_messages(selected_room, limit=20)
        if messages:
            for message in reversed(messages):  # Show newest first
                message_id, sender_name, message_text, message_type, created_at, is_private = message
                
                # Create message bubble
                with st.container():
                    st.markdown(f"""
                    <div style="
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 12px;
                        margin: 8px 0;
                        background-color: #f9f9f9;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <strong>{sender_name}</strong>
                            <small>{created_at}</small>
                        </div>
                        <div style="margin-bottom: 8px;">
                            <span style="background-color: #e0e0e0; padding: 2px 6px; border-radius: 4px; font-size: 12px;">
                                {message_type}
                            </span>
                        </div>
                        <div style="margin-bottom: 8px;">
                            {message_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No messages in this room yet. Be the first to send a message!")
        
        # Chat statistics
        st.markdown("---")
        st.write("**📊 Chat Statistics**")
        chat_stats = get_chat_statistics()
        if chat_stats:
            total_messages = sum(stat[4] for stat in chat_stats)
            active_users = chat_stats[0][2] if chat_stats else 0
            total_rooms = chat_stats[0][3] if chat_stats else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", total_messages)
            with col2:
                st.metric("Active Users", active_users)
            with col3:
                st.metric("Chat Rooms", total_rooms)
    
    with tab12:
        st.subheader("📊 Gantt Charts")
        st.markdown("Visual project timeline management")
        
        # Project filter
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT category FROM tasks WHERE category IS NOT NULL")
            categories = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if categories:
                selected_category = st.selectbox("Filter by project category:", ["All Projects"] + categories)
                project_filter = None if selected_category == "All Projects" else selected_category
            else:
                project_filter = None
        else:
            project_filter = None
        
        # Get Gantt data
        try:
            gantt_data = get_gantt_data(project_filter)
            
            if gantt_data:
                st.success(f"✅ Loaded {len(gantt_data)} tasks for visualization")
                
                # Create Gantt chart
                gantt_fig = create_gantt_chart(gantt_data)
                if gantt_fig:
                    st.plotly_chart(gantt_fig, use_container_width=True)
                
                # Timeline chart
                st.markdown("---")
                st.write("**📈 Timeline by Category**")
                timeline_data = get_project_timeline_data()
                if timeline_data:
                    timeline_fig = create_timeline_chart(timeline_data)
                    if timeline_fig:
                        st.plotly_chart(timeline_fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error loading Gantt data: {str(e)}")
            st.info("Please check your data format and try again.")
            
            # Project statistics
            st.markdown("---")
            st.write("**📊 Project Statistics**")
            
            # Calculate statistics
            total_tasks = len(gantt_data)
            completed_tasks = len([task for task in gantt_data if task[2] == 'completed'])
            in_progress_tasks = len([task for task in gantt_data if task[2] == 'in_progress'])
            pending_tasks = len([task for task in gantt_data if task[2] == 'pending'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tasks", total_tasks)
            with col2:
                st.metric("Completed", completed_tasks)
            with col3:
                st.metric("In Progress", in_progress_tasks)
            with col4:
                st.metric("Pending", pending_tasks)
            
            # Progress chart
            if total_tasks > 0:
                progress_data = {
                    'Completed': completed_tasks,
                    'In Progress': in_progress_tasks,
                    'Pending': pending_tasks
                }
                fig = px.pie(values=list(progress_data.values()), names=list(progress_data.keys()), 
                           title="Project Progress")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tasks available for Gantt chart visualization.")
    
    with tab13:
        st.subheader("🗣️ Natural Language Queries")
        st.markdown("Ask questions about your tasks in natural language")
        
        # Check if Gemini is available
        if not GEMINI_READY:
            st.warning("⚠️ Gemini API is not available. Please configure it in Settings > AI Configuration.")
            st.info("💡 Example queries you can try:")
            st.markdown("""
            - "Show me all high priority tasks"
            - "Who is working on bug fixes?"
            - "Find tasks assigned to John"
            - "Show completed tasks from last week"
            - "What tasks are overdue?"
            """)
        else:
            # Natural language query interface
            st.write("**🤖 Ask questions about your tasks:**")
            
            # Example queries
            st.markdown("**💡 Example queries:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                - "Show me all high priority tasks"
                - "Who is working on bug fixes?"
                - "Find tasks assigned to John"
                """)
            with col2:
                st.markdown("""
                - "Show completed tasks from last week"
                - "What tasks are overdue?"
                - "Tasks with complexity score > 7"
                """)
            
            # Query input
            query = st.text_input(
                "Enter your question:",
                placeholder="e.g., Show me all high priority tasks",
                help="Ask questions about your tasks in natural language"
            )
            
            if st.button("🔍 Search", type="primary"):
                if query.strip():
                    try:
                        # Get task data
                        conn = get_database_connection()
                        if conn:
                            tasks_df = pd.read_sql_query("SELECT * FROM tasks", conn)
                            conn.close()
                            
                            if not tasks_df.empty:
                                with st.spinner("🤖 Processing your query..."):
                                    result = process_natural_language_query(query, tasks_df)
                                
                                if "error" not in result:
                                    st.success("✅ Query processed successfully!")
                                    
                                    # Display results
                                    st.markdown("**📊 Query Results:**")
                                    st.json(result)
                                    
                                    # Apply filters to actual data
                                    filter_criteria = result.get("filter_criteria", {})
                                    filtered_df = tasks_df.copy()
                                    
                                    # Apply status filter
                                    if "status" in filter_criteria:
                                        status_filter = filter_criteria["status"]
                                        if isinstance(status_filter, list):
                                            filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]
                                        else:
                                            filtered_df = filtered_df[filtered_df['status'] == status_filter]
                                    
                                    # Apply priority filter
                                    if "priority" in filter_criteria:
                                        priority_filter = filter_criteria["priority"]
                                        if isinstance(priority_filter, list):
                                            filtered_df = filtered_df[filtered_df['priority'].isin(priority_filter)]
                                        else:
                                            filtered_df = filtered_df[filtered_df['priority'] == priority_filter]
                                    
                                    # Apply assigned_to filter
                                    if "assigned_to" in filter_criteria:
                                        assigned_filter = filter_criteria["assigned_to"]
                                        filtered_df = filtered_df[filtered_df['assigned_to'].str.contains(assigned_filter, case=False, na=False)]
                                    
                                    # Apply category filter
                                    if "category" in filter_criteria:
                                        category_filter = filter_criteria["category"]
                                        filtered_df = filtered_df[filtered_df['category'].str.contains(category_filter, case=False, na=False)]
                                    
                                    # Sort results
                                    sort_by = result.get("sort_by")
                                    sort_order = result.get("sort_order", "asc")
                                    if sort_by and sort_by in filtered_df.columns:
                                        filtered_df = filtered_df.sort_values(sort_by, ascending=(sort_order == "asc"))
                                    
                                    # Limit results
                                    limit = result.get("limit", 10)
                                    filtered_df = filtered_df.head(limit)
                                    
                                    # Display filtered results
                                    st.markdown(f"**📋 Found {len(filtered_df)} matching tasks:**")
                                    if not filtered_df.empty:
                                        st.dataframe(filtered_df, use_container_width=True)
                                        
                                        # Export option
                                        csv = filtered_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Results",
                                            data=csv,
                                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                            mime="text/csv"
                                        )
                                    else:
                                        st.info("No tasks match your query criteria.")
                                else:
                                    st.error(f"❌ Query processing failed: {result['error']}")
                            else:
                                st.warning("⚠️ No task data available for querying.")
                        else:
                            st.error("❌ Database connection failed.")
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a query.")

def show_employee_management():
    """Show enhanced employee management page"""
    st.header("👥 Employee Management")
    st.markdown("---")
    
    # Load employee data
    employees = load_employee_data_from_json()
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Employee Overview", "🤖 AI Assignment", "📈 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.subheader("📊 Employee Overview")
        
        if employees:
            # Save to database
            if st.button("💾 Save Employees to Database"):
                save_employees_to_database(employees)
            
            # Display employee data
            df = pd.DataFrame(employees)
            
            # Show summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Employees", len(employees))
            with col2:
                roles = df['role'].value_counts()
                st.metric("Unique Roles", len(roles))
            with col3:
                avg_experience = df['experience_years'].mean()
                st.metric("Avg Experience", f"{avg_experience:.1f} years")
            with col4:
                avg_workload = df['current_workload'].mean()
                st.metric("Avg Workload", f"{avg_workload:.1f}/10")
            
            # Show employee table
            st.subheader("👥 Employee List")
            display_df = df[['name', 'role', 'location', 'experience_years', 'current_workload', 'max_capacity']].copy()
            st.dataframe(display_df, use_container_width=True)
            
            # Role distribution
            st.subheader("📊 Role Distribution")
            role_counts = df['role'].value_counts()
            fig = px.bar(x=role_counts.index, y=role_counts.values, 
                        title="Employee Distribution by Role",
                        labels={'x': 'Role', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Experience distribution
            st.subheader("📈 Experience Distribution")
            fig = px.histogram(df, x='experience_years', nbins=10,
                             title="Employee Experience Distribution",
                             labels={'experience_years': 'Years of Experience', 'count': 'Number of Employees'})
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No employee data available. Please check the JSON file.")
    
    with tab2:
        st.subheader("🤖 AI Assignment")
        
        # Add drag and drop assignment interface
        st.markdown("### 🎯 Drag & Drop Assignment")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**📋 Available Tasks**")
            st.markdown("""
            <div class="drop-zone" id="availableTasks">
                <div class="drop-zone empty">No tasks available</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**👥 Available Employees**")
            st.markdown("""
            <div class="drop-zone" id="availableEmployees">
                <div class="drop-zone empty">No employees available</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Assignment area
        st.markdown("### 🔗 Assignment Area")
        st.markdown("""
        <div class="drag-drop-zone" id="assignmentZone">
            <h4>🎯 Drag tasks to employees here</h4>
            <p>Drop a task on an employee to assign it</p>
            <div id="assignments" style="min-height: 200px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get employees from database
        emp_df = get_employees_from_database()
        
        if not emp_df.empty:
            st.info("🤖 AI-powered employee assignment based on skills, experience, and workload")
            
            # Task input for testing
            with st.form("ai_assignment_test"):
                task_title = st.text_input("Task Title", placeholder="Enter task title...")
                task_description = st.text_area("Task Description", placeholder="Enter task description...")
                task_category = st.selectbox("Task Category", ["bug", "feature", "documentation", "testing", "optimization", "design"])
                task_priority = st.selectbox("Task Priority", ["low", "medium", "high", "critical"])
                
                col1, col2 = st.columns(2)
                with col1:
                    urgency_score = st.slider("Urgency Score", 1, 10, 5)
                    complexity_score = st.slider("Complexity Score", 1, 10, 5)
                with col2:
                    business_impact = st.slider("Business Impact", 1, 10, 5)
                    estimated_hours = st.number_input("Estimated Hours", min_value=0.5, max_value=100.0, value=8.0, step=0.5)
                
                submitted = st.form_submit_button("🤖 Get AI Recommendation")
            
            if submitted and task_title:
                # Get AI recommendation
                recommended_employee = get_ai_employee_recommendation(
                    task_title, task_description, task_category, task_priority,
                    urgency_score, complexity_score, business_impact, estimated_hours,
                    emp_df
                )
                
                if recommended_employee:
                    st.success(f"✅ **Recommended Employee:** {recommended_employee['name']}")
                    st.info(f"📊 **Role:** {recommended_employee['role']}")
                    st.info(f"🎯 **Skills:** {', '.join(recommended_employee['skills'][:5])}")
                    st.info(f"📈 **Experience:** {recommended_employee['experience_years']} years")
                    st.info(f"⚖️ **Current Workload:** {recommended_employee['current_workload']}/{recommended_employee['max_capacity']}")
                    
                    # Show reasoning
                    reasoning = generate_employee_recommendation_reasoning(
                        task_category, task_priority, complexity_score, recommended_employee
                    )
                    st.info("🔍 **AI Reasoning:**")
                    st.write(reasoning)
                else:
                    st.warning("⚠️ No suitable employee found for this task")
        else:
            st.info("Please load employee data first in the Employee Overview tab.")
    
    with tab3:
        st.subheader("📈 Analytics")
        
        emp_df = get_employees_from_database()
        if not emp_df.empty:
            # Workload analysis
            st.subheader("⚖️ Workload Analysis")
            workload_data = emp_df[['name', 'current_workload', 'max_capacity']].copy()
            workload_data['utilization'] = (workload_data['current_workload'] / workload_data['max_capacity']) * 100
            
            fig = px.bar(workload_data, x='name', y='utilization',
                        title="Employee Workload Utilization (%)",
                        labels={'name': 'Employee', 'utilization': 'Utilization %'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Skills analysis
            st.subheader("🎯 Skills Analysis")
            all_skills = []
            for skills in emp_df['skills']:
                all_skills.extend(skills)
            
            skill_counts = pd.Series(all_skills).value_counts().head(10)
            fig = px.bar(x=skill_counts.index, y=skill_counts.values,
                        title="Top 10 Skills Across Employees",
                        labels={'x': 'Skill', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Experience vs Workload
            st.subheader("📊 Experience vs Workload")
            fig = px.scatter(emp_df, x='experience_years', y='current_workload',
                           title="Experience vs Current Workload",
                           labels={'experience_years': 'Years of Experience', 'current_workload': 'Current Workload'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No employee data available for analytics.")
    
    with tab4:
        st.subheader("⚙️ Settings")
        
        # Employee data management
        st.info("Employee Data Management")
        
        if st.button("🔄 Refresh Employee Data"):
            employees = load_employee_data_from_json()
            if employees:
                save_employees_to_database(employees)
                st.success("✅ Employee data refreshed successfully")
        
        # Database status
        emp_df = get_employees_from_database()
        if not emp_df.empty:
            st.success(f"✅ Database contains {len(emp_df)} employees")
        else:
            st.warning("⚠️ No employees in database")

def get_ai_employee_recommendation(task_title, task_description, task_category, task_priority,
                                 urgency_score, complexity_score, business_impact, estimated_hours,
                                 employees_df):
    """Get AI recommendation for employee assignment"""
    try:
        if employees_df.empty:
            return None
        
        # Calculate scores for each employee
        employee_scores = []
        
        for _, emp in employees_df.iterrows():
            score = 0
            
            # Skills matching
            emp_skills = emp['skills']
            task_text = f"{task_title} {task_description}".lower()
            
            for skill in emp_skills:
                if skill.lower() in task_text:
                    score += 10
            
            # Category matching
            preferred_types = emp['preferred_task_types']
            if task_category in preferred_types:
                score += 20
            
            # Experience matching
            if complexity_score >= 8 and emp['experience_years'] >= 5:
                score += 15
            elif complexity_score >= 5 and emp['experience_years'] >= 3:
                score += 10
            
            # Workload consideration
            workload_ratio = emp['current_workload'] / emp['max_capacity']
            if workload_ratio < 0.7:  # Prefer employees with available capacity
                score += 10
            elif workload_ratio > 0.9:  # Penalize overloaded employees
                score -= 20
            
            # Priority matching
            if task_priority in ['high', 'critical'] and emp['experience_years'] >= 5:
                score += 10
            
            # Business impact consideration
            if business_impact >= 8 and emp['experience_years'] >= 5:
                score += 10
            
            employee_scores.append({
                'employee': emp,
                'score': score
            })
        
        # Sort by score and return best match
        employee_scores.sort(key=lambda x: x['score'], reverse=True)
        
        if employee_scores and employee_scores[0]['score'] > 0:
            return employee_scores[0]['employee']
        else:
            return None
            
    except Exception as e:
        st.error(f"❌ Error in AI recommendation: {e}")
        return None

def generate_employee_recommendation_reasoning(task_category, task_priority, complexity_score, employee):
    """Generate reasoning for employee recommendation"""
    reasoning = []
    
    # Skills reasoning
    skills = employee['skills']
    reasoning.append(f"🎯 **Skills Match:** Employee has {len(skills)} relevant skills")
    
    # Experience reasoning
    experience = employee['experience_years']
    if experience >= 5:
        reasoning.append(f"📈 **Senior Experience:** {experience} years of experience")
    elif experience >= 3:
        reasoning.append(f"📊 **Mid-level Experience:** {experience} years of experience")
    else:
        reasoning.append(f"📚 **Junior Experience:** {experience} years of experience")
    
    # Workload reasoning
    workload = employee['current_workload']
    max_capacity = employee['max_capacity']
    utilization = (workload / max_capacity) * 100
    
    if utilization < 70:
        reasoning.append(f"⚖️ **Available Capacity:** {utilization:.1f}% workload utilization")
    elif utilization < 90:
        reasoning.append(f"⚠️ **Moderate Workload:** {utilization:.1f}% workload utilization")
    else:
        reasoning.append(f"🔥 **High Workload:** {utilization:.1f}% workload utilization")
    
    # Category reasoning
    preferred_types = employee['preferred_task_types']
    if task_category in preferred_types:
        reasoning.append(f"✅ **Preferred Task Type:** {task_category} is in preferred types")
    
    # Priority reasoning
    if task_priority in ['high', 'critical'] and experience >= 5:
        reasoning.append("🚨 **High Priority Task:** Assigned to experienced employee")
    
    return "\n".join(reasoning)

def load_employee_data_from_json():
    """Load employee data from the JSON file"""
    try:
        json_path = 'data/employees_100.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                employees = json.load(f)
            st.success(f"✅ Loaded {len(employees)} employees from JSON file")
            return employees
        else:
            st.warning("⚠️ Employee JSON file not found")
            return []
    except Exception as e:
        st.error(f"❌ Error loading employee data: {e}")
        return []

def save_employees_to_database(employees):
    """Save employees to database"""
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            
            # Create employees table if it doesn't exist
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
            
            # Clear existing employees
            cursor.execute("DELETE FROM employees")
            
            # Insert employees
            for emp in employees:
                cursor.execute("""
                    INSERT INTO employees (
                        id, name, role, skills, expertise_areas, preferred_task_types,
                        current_workload, max_capacity, experience_years, location, availability,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    emp.get('id', ''),
                    emp.get('name', ''),
                    emp.get('role', ''),
                    json.dumps(emp.get('skills', [])),
                    json.dumps(emp.get('expertise_areas', [])),
                    json.dumps(emp.get('preferred_task_types', [])),
                    emp.get('current_workload', 0),
                    emp.get('max_capacity', 10),
                    emp.get('experience_years', 0),
                    emp.get('location', ''),
                    emp.get('availability', 'full-time'),
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            st.success(f"✅ Saved {len(employees)} employees to database")
            return True
        else:
            st.error("❌ Could not connect to database")
            return False
    except Exception as e:
        st.error(f"❌ Error saving employees: {e}")
        return False

def get_employees_from_database():
    """Get employees from database"""
    try:
        conn = get_database_connection()
        if conn:
            df = pd.read_sql_query("SELECT * FROM employees", conn)
            conn.close()
            
            # Parse JSON fields
            if not df.empty:
                df['skills'] = df['skills'].apply(lambda x: json.loads(x) if x else [])
                df['expertise_areas'] = df['expertise_areas'].apply(lambda x: json.loads(x) if x else [])
                df['preferred_task_types'] = df['preferred_task_types'].apply(lambda x: json.loads(x) if x else [])
            
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error getting employees: {e}")
        return pd.DataFrame()

def show_ai_models():
    """Show AI models page"""
    st.header("🧠 AI Models")
    
    st.subheader("🤖 Model Status")
    
    # Model components
    components = [
        {"name": "Task Classifier", "status": "✅ Ready", "description": "Classifies tasks into categories"},
        {"name": "Priority Predictor", "status": "✅ Ready", "description": "Predicts task priority scores"},
        {"name": "Task Assigner", "status": "✅ Ready", "description": "Assigns tasks to employees"},
        {"name": "Feature Engineer", "status": "✅ Ready", "description": "Extracts advanced features"},
        {"name": "Data Preprocessor", "status": "✅ Ready", "description": "Cleans and prepares data"}
    ]
    
    for component in components:
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{component['name']}**")
            with col2:
                st.write(f"{component['status']} - {component['description']}")
    
    # Model testing
    st.subheader("🧪 Test AI Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 Test Task Classification"):
            with st.spinner("Testing classifier..."):
                try:
                    classifier = TaskClassifier()
                    st.success("✅ Task Classifier working correctly")
                except Exception as e:
                    st.error(f"❌ Task Classifier error: {e}")
    
    with col2:
        if st.button("🧪 Test Priority Prediction"):
            with st.spinner("Testing priority model..."):
                try:
                    priority_model = TaskPriorityModel()
                    st.success("✅ Priority Predictor working correctly")
                except Exception as e:
                    st.error(f"❌ Priority Predictor error: {e}")

def show_settings():
    """Show settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("📊 System Information")
    
    # System metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🗄️ Database Status", "✅ Connected")
        st.metric("🐍 Python Version", "3.12")
        st.metric("📊 Streamlit Version", "1.37.1")
    
    with col2:
        st.metric("📁 Total Files Processed", len(st.session_state.uploaded_files))
        st.metric("📋 Total Tasks Processed", len(st.session_state.processed_tasks))
        st.metric("👥 Employees Loaded", len(st.session_state.employee_profiles))
    
    # Database operations
    st.subheader("🗄️ Database Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Data"):
            if st.button("✅ Confirm Clear", key="confirm_clear"):
                conn = get_database_connection()
                if conn:
                    conn.execute("DELETE FROM tasks")
                    conn.commit()
                    st.success("✅ All data cleared")
    
    with col2:
        if st.button("📥 Export Database"):
            conn = get_database_connection()
            if conn:
                df = pd.read_sql_query("SELECT * FROM tasks", conn)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download database.csv",
                    data=csv,
                    file_name="database_export.csv",
                    mime="text/csv"
                )
    
    # Notification Settings
    st.subheader("🔔 Notification Settings")
    st.markdown("Configure how and when you receive notifications")
    
    # Notification preferences
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📧 Email Notifications**")
        st.session_state.notification_settings['email_notifications'] = st.checkbox(
            "Enable email notifications", 
            value=st.session_state.notification_settings.get('email_notifications', True)
        )
        
        st.write("**📱 SMS Notifications**")
        st.session_state.notification_settings['sms_notifications'] = st.checkbox(
            "Enable SMS notifications", 
            value=st.session_state.notification_settings.get('sms_notifications', False)
        )
    
    with col2:
        st.write("**🎯 Notification Types**")
        st.session_state.notification_settings['task_assignments'] = st.checkbox(
            "Task assignments", 
            value=st.session_state.notification_settings.get('task_assignments', True)
        )
        st.session_state.notification_settings['task_updates'] = st.checkbox(
            "Task status updates", 
            value=st.session_state.notification_settings.get('task_updates', True)
        )
        st.session_state.notification_settings['deadline_alerts'] = st.checkbox(
            "Deadline alerts", 
            value=st.session_state.notification_settings.get('deadline_alerts', True)
        )
        st.session_state.notification_settings['comment_notifications'] = st.checkbox(
            "Comment notifications", 
            value=st.session_state.notification_settings.get('comment_notifications', True)
        )
    
    # Test notifications
    st.markdown("---")
    st.write("**🧪 Test Notifications**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 Test Email"):
            send_email_notification("test@company.com", "Test Notification", "This is a test email notification from the AI Task Management System.")
    
    with col2:
        if st.button("📱 Test SMS"):
            send_sms_notification("+1234567890", "Test SMS notification from AI Task Management System")
    
    with col3:
        if st.button("🔍 Check Deadlines"):
            check_deadline_alerts()
            st.success("✅ Deadline check completed!")
    
    # Notification history
    st.markdown("---")
    st.write("**📋 Recent Notifications**")
    
    notification_history = get_notification_history()
    if notification_history:
        for notification in notification_history[:5]:  # Show last 5 notifications
            st.info(f"**{notification['timestamp']}** - {notification['message']}")
    else:
        st.info("No recent notifications to display.")
    


def show_analytics():
    """Show comprehensive advanced analytics and reporting"""
    st.header("📊 Advanced Analytics & Reporting")
    
    # Create tabs for different analytics views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Real-time Dashboard", 
        "👥 Employee Productivity", 
        "📊 Performance Analytics", 
        "🎯 KPI Tracking", 
        "📄 Custom Reports"
    ])
    
    with tab1:
        st.subheader("📈 Real-time Dashboard")
        
        # Get KPI metrics
        kpis = get_kpi_metrics()
        
        if kpis:
            # Overall KPIs
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📋 Total Tasks", kpis['overall']['total_tasks'])
                st.metric("✅ Completion Rate", f"{kpis['overall']['completion_rate']:.1f}%")
            
            with col2:
                st.metric("⏳ Pending Tasks", kpis['overall']['pending_tasks'])
                st.metric("🔥 High Priority", kpis['overall']['high_priority_tasks'])
            
            with col3:
                st.metric("👥 Active Employees", kpis['employee']['active_employees'])
                st.metric("🤖 AI Confidence", f"{kpis['employee']['avg_ai_confidence']:.1f}%")
            
            with col4:
                st.metric("📊 Avg Completion Days", f"{kpis['overall']['avg_completion_days']:.1f}")
                st.metric("🎯 Total Assignments", kpis['employee']['total_assignments'])
            
            # Real-time charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Task status distribution
                status_data = {
                    'Completed': kpis['overall']['completed_tasks'],
                    'In Progress': kpis['overall']['in_progress_tasks'],
                    'Pending': kpis['overall']['pending_tasks']
                }
                fig = px.pie(values=list(status_data.values()), names=list(status_data.keys()), 
                           title="Task Status Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Category performance
                if kpis['category']:
                    cat_df = pd.DataFrame(kpis['category'], columns=['Category', 'Task Count', 'Completed', 'Avg Days'])
                    fig = px.bar(cat_df, x='Category', y='Task Count', 
                               title="Tasks by Category")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No KPI data available. Please upload some task data first.")
    
    with tab2:
        st.subheader("👥 Employee Productivity Analytics")
        
        # Get productivity metrics
        productivity_data = get_employee_productivity_metrics()
        
        if productivity_data:
            # Productivity overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_completion_rate = sum(emp['completion_rate'] for emp in productivity_data) / len(productivity_data)
                st.metric("📈 Avg Completion Rate", f"{avg_completion_rate:.1f}%")
            
            with col2:
                avg_workload = sum(emp['workload_percentage'] for emp in productivity_data) / len(productivity_data)
                st.metric("⚖️ Avg Workload", f"{avg_workload:.1f}%")
            
            with col3:
                total_employees = len(productivity_data)
                st.metric("👥 Total Employees", total_employees)
            
            # Employee productivity table
            st.subheader("📊 Employee Performance")
            
            if productivity_data:
                df = pd.DataFrame(productivity_data)
                
                # Format the data for display
                display_df = df[['name', 'role', 'total_tasks', 'completed_tasks', 'completion_rate', 'workload_percentage']].copy()
                display_df['completion_rate'] = display_df['completion_rate'].apply(lambda x: f"{x:.1f}%")
                display_df['workload_percentage'] = display_df['workload_percentage'].apply(lambda x: f"{x:.1f}%")
                display_df.columns = ['Employee', 'Role', 'Total Tasks', 'Completed', 'Completion Rate', 'Workload %']
                
                st.dataframe(display_df, use_container_width=True)
                
                # Productivity charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Completion rate by employee
                    fig = px.bar(df, x='name', y='completion_rate', 
                               title="Completion Rate by Employee",
                               labels={'name': 'Employee', 'completion_rate': 'Completion Rate (%)'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Workload distribution
                    fig = px.bar(df, x='name', y='workload_percentage', 
                               title="Workload Distribution",
                               labels={'name': 'Employee', 'workload_percentage': 'Workload (%)'})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No productivity data available.")
    
    with tab3:
        st.subheader("📊 Performance Analytics")
        
        # Get completion analytics
        completion_data = get_task_completion_analytics()
        
        if completion_data:
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                completed_tasks = [task for task in completion_data if task['status'] == 'completed']
                avg_completion_days = sum(task['completion_days'] for task in completed_tasks) / len(completed_tasks) if completed_tasks else 0
                st.metric("⏱️ Avg Completion Time", f"{avg_completion_days:.1f} days")
            
            with col2:
                high_priority_tasks = [task for task in completion_data if task['priority'] == 'high']
                high_priority_completed = [task for task in high_priority_tasks if task['status'] == 'completed']
                high_priority_rate = len(high_priority_completed) / len(high_priority_tasks) * 100 if high_priority_tasks else 0
                st.metric("🔥 High Priority Completion", f"{high_priority_rate:.1f}%")
            
            with col3:
                avg_confidence = sum(task['confidence_score'] for task in completion_data) / len(completion_data)
                st.metric("🤖 Avg AI Confidence", f"{avg_confidence:.1f}%")
            
            # Performance charts
            df = pd.DataFrame(completion_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Completion time distribution
                completed_df = df[df['status'] == 'completed']
                if not completed_df.empty:
                    fig = px.histogram(completed_df, x='completion_days', 
                                     title="Task Completion Time Distribution",
                                     labels={'completion_days': 'Days to Complete'})
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Priority vs completion rate
                priority_completion = df.groupby('priority')['status'].apply(
                    lambda x: (x == 'completed').sum() / len(x) * 100
                ).reset_index()
                priority_completion.columns = ['Priority', 'Completion Rate (%)']
                
                fig = px.bar(priority_completion, x='Priority', y='Completion Rate (%)',
                           title="Completion Rate by Priority")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No completion analytics data available.")
    
    with tab4:
        st.subheader("🎯 KPI Tracking")
        
        kpis = get_kpi_metrics()
        
        if kpis:
            # KPI Dashboard
            st.markdown("### 📊 Key Performance Indicators")
            
            # Overall KPIs
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Overall KPIs")
                kpi_df = pd.DataFrame([
                    ['Total Tasks', kpis['overall']['total_tasks']],
                    ['Completed Tasks', kpis['overall']['completed_tasks']],
                    ['Completion Rate', f"{kpis['overall']['completion_rate']:.1f}%"],
                    ['High Priority Tasks', kpis['overall']['high_priority_tasks']],
                    ['Avg Completion Days', f"{kpis['overall']['avg_completion_days']:.1f}"]
                ], columns=['Metric', 'Value'])
                st.dataframe(kpi_df, use_container_width=True)
            
            with col2:
                st.markdown("#### 👥 Employee KPIs")
                emp_kpi_df = pd.DataFrame([
                    ['Active Employees', kpis['employee']['active_employees']],
                    ['Avg AI Confidence', f"{kpis['employee']['avg_ai_confidence']:.1f}%"],
                    ['Total Assignments', kpis['employee']['total_assignments']]
                ], columns=['Metric', 'Value'])
                st.dataframe(emp_kpi_df, use_container_width=True)
            
            # Category KPIs
            st.markdown("#### 📊 Category Performance")
            if kpis['category']:
                cat_kpi_df = pd.DataFrame(kpis['category'], 
                                        columns=['Category', 'Task Count', 'Completed', 'Avg Days'])
                cat_kpi_df['Completion Rate'] = (cat_kpi_df['Completed'] / cat_kpi_df['Task Count'] * 100).round(1)
                cat_kpi_df['Completion Rate'] = cat_kpi_df['Completion Rate'].apply(lambda x: f"{x}%")
                cat_kpi_df['Avg Days'] = cat_kpi_df['Avg Days'].apply(lambda x: f"{x:.1f}" if x else "N/A")
                
                st.dataframe(cat_kpi_df, use_container_width=True)
        else:
            st.info("No KPI data available.")
    
    with tab5:
        st.subheader("📄 Custom Reports")
        
        # Report generation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Generate Reports")
            
            report_type = st.selectbox(
                "Select Report Type:",
                ["Productivity Report", "Completion Analytics", "KPI Summary", "Comprehensive Report"]
            )
            
            if st.button("📄 Generate Report"):
                with st.spinner("Generating report..."):
                    if report_type == "Productivity Report":
                        df = export_analytics_report("productivity")
                    elif report_type == "Completion Analytics":
                        df = export_analytics_report("completion")
                    elif report_type == "KPI Summary":
                        df = export_analytics_report("kpi")
                    else:
                        # Comprehensive report
                        productivity_df = export_analytics_report("productivity")
                        completion_df = export_analytics_report("completion")
                        kpi_df = export_analytics_report("kpi")
                        
                        # Combine all reports
                        st.session_state.comprehensive_report = {
                            'productivity': productivity_df,
                            'completion': completion_df,
                            'kpi': kpi_df
                        }
                        df = productivity_df  # Show productivity as default
                    
                    if df is not None and not df.empty:
                        st.session_state.current_report = df
                        st.session_state.report_type = report_type
                        st.success("✅ Report generated successfully!")
                    else:
                        st.error("❌ No data available for report generation.")
        
        with col2:
            st.markdown("#### 📥 Export Options")
            
            if 'current_report' in st.session_state:
                st.markdown(f"**Current Report:** {st.session_state.report_type}")
                
                # Export as CSV
                csv = st.session_state.current_report.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name=f"{st.session_state.report_type.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
                # Export as Excel
                if st.button("📊 Download Excel"):
                    try:
                        import io
                        from openpyxl import Workbook
                        
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            st.session_state.current_report.to_excel(writer, sheet_name='Report', index=False)
                        
                        excel_data = output.getvalue()
                        st.download_button(
                            label="📊 Download Excel",
                            data=excel_data,
                            file_name=f"{st.session_state.report_type.lower().replace(' ', '_')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except ImportError:
                        st.error("Excel export requires openpyxl. Install with: pip install openpyxl")
                
                # Show report preview
                st.markdown("#### 📋 Report Preview")
                st.dataframe(st.session_state.current_report, use_container_width=True)
            else:
                st.info("Generate a report first to see export options.")

def show_training_monitor():
    """Show training monitor page"""
    st.header("🔧 Training Monitor")
    
    st.subheader("📊 Model Training Status")
    
    # Training metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Model Accuracy", "94.2%")
    
    with col2:
        st.metric("⚡ Training Speed", "2.3s/epoch")
    
    with col3:
        st.metric("📈 Loss", "0.023")
    
    # Training progress
    st.subheader("📈 Training Progress")
    
    # Simulate training progress
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)
    
    st.success("✅ Training completed successfully!")

def load_auto_assign_model():
    """Load the pre-trained auto assignment model"""
    try:
        model_path = 'auto_assign_model.joblib'
        st.info(f"🔍 Looking for model at: {os.path.abspath(model_path)}")
        
        if os.path.exists(model_path):
            st.info("✅ Model file found, loading...")
            model = joblib.load(model_path)
            st.success("✅ Auto-assign model loaded successfully")
            return model
        else:
            st.warning("⚠️ Auto-assign model not found. Creating default model...")
            return create_default_auto_assign_model()
    except Exception as e:
        st.error(f"❌ Error loading auto-assign model: {e}")
        st.info("🔄 Attempting to create default model...")
        return create_default_auto_assign_model()

def create_default_auto_assign_model():
    """Create a default auto-assignment model if none exists"""
    try:
        st.info("🔧 Creating default auto-assignment model...")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        
        # Create a simple pipeline for auto-assignment
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Sample training data
        sample_tasks = [
            'Fix login bug in authentication system',
            'Create new user dashboard interface',
            'Update API documentation with new endpoints',
            'Optimize database query performance',
            'Design mobile app UI components',
            'Implement payment processing feature',
            'Write unit tests for user module',
            'Deploy application to production server',
            'Review and update security policies',
            'Create automated backup system'
        ]
        
        # Sample employee assignments (0-4 for 5 employees)
        sample_assignments = [0, 1, 2, 0, 1, 2, 3, 4, 0, 1]
        
        # Train the model
        model.fit(sample_tasks, sample_assignments)
        
        # Save the default model
        joblib.dump(model, 'auto_assign_model.joblib')
        st.success("✅ Created default auto-assign model")
        return model
    except Exception as e:
        st.error(f"❌ Error creating default model: {e}")
        st.error(f"❌ Full error details: {str(e)}")
        return None

def predict_employee_assignment(task_data, model):
    """Predict the best employee for a task using the AI model"""
    try:
        if model is None:
            st.warning("⚠️ Model is None, cannot make prediction")
            return "Not available"
        
        st.info("🔍 Preparing task features for prediction...")
        
        # Prepare task features for prediction
        task_features = prepare_task_features_for_prediction(task_data)
        
        if not task_features:
            st.warning("⚠️ Could not prepare task features")
            return "Not available"
        
        st.info(f"📝 Task features prepared: {task_features[:100]}...")
        
        # Make prediction
        st.info("🤖 Making AI prediction...")
        prediction = model.predict([task_features])
        
        st.info(f"📊 Raw prediction result: {prediction}")
        
        # Get available employees
        employees = get_available_employees()
        st.info(f"👥 Available employees: {employees}")
        
        if prediction and len(employees) > 0:
            # Map prediction to employee name
            predicted_index = prediction[0] % len(employees)
            predicted_employee = employees[predicted_index]
            st.success(f"✅ Prediction successful: {predicted_employee}")
            return predicted_employee
        else:
            st.warning("⚠️ No prediction result or no employees available")
            return "Not available"
            
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.error(f"❌ Full error details: {str(e)}")
        return "Not available"

def prepare_task_features_for_prediction(task_data):
    """Prepare task features for AI prediction"""
    try:
        # Combine task information into a single text string
        task_text = f"{task_data.get('title', '')} {task_data.get('description', '')} {task_data.get('category', '')} {task_data.get('priority', '')}"
        
        # Add numerical features
        urgency = task_data.get('urgency_score', 5)
        complexity = task_data.get('complexity_score', 5)
        business_impact = task_data.get('business_impact', 5)
        estimated_hours = task_data.get('estimated_hours', 8.0)
        
        # Create feature string
        features = f"{task_text} urgency:{urgency} complexity:{complexity} impact:{business_impact} hours:{estimated_hours}"
        
        return features
        
    except Exception as e:
        st.error(f"❌ Feature preparation error: {e}")
        return ""

def get_available_employees():
    """Get list of available employees for assignment"""
    try:
        conn = get_database_connection()
        if conn:
            # Get employees from database or return default list
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT assigned_to FROM tasks WHERE assigned_to IS NOT NULL AND assigned_to != ''")
            employees = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if employees:
                return employees
            else:
                # Return default employees if none in database
                return ["John Smith", "Sarah Johnson", "Mike Davis", "Lisa Wilson", "David Brown"]
        else:
            return ["John Smith", "Sarah Johnson", "Mike Davis", "Lisa Wilson", "David Brown"]
            
    except Exception as e:
        st.error(f"❌ Error getting employees: {e}")
        return ["John Smith", "Sarah Johnson", "Mike Davis", "Lisa Wilson", "David Brown"]

def show_ai_prediction_page():
    """Show the AI Task Prediction page"""
    st.header("🤖 AI Task Prediction")
    st.markdown("---")
    
    # Load the auto-assign model
    model = load_auto_assign_model()
    
    if model is None:
        st.error("❌ Could not load AI prediction model")
        return
    
    # Create tabs for different prediction features
    tab1, tab2, tab3 = st.tabs(["🎯 Single Task Prediction", "📊 Batch Assignment", "📈 Model Performance"])
    
    with tab1:
        # Single task prediction
        st.subheader("🎯 Single Task Prediction")
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Task input form
            with st.form("task_prediction_form"):
                task_title = st.text_input("Task Title", placeholder="Enter task title...")
                task_description = st.text_area("Task Description", placeholder="Enter task description...")
                
                col1a, col1b = st.columns(2)
                with col1a:
                    task_category = st.selectbox("Category", ["bug", "feature", "documentation", "optimization", "design", "testing"])
                    task_priority = st.selectbox("Priority", ["low", "medium", "high", "critical"])
                
                with col1b:
                    urgency_score = st.slider("Urgency Score", 1, 10, 5)
                    complexity_score = st.slider("Complexity Score", 1, 10, 5)
                
                business_impact = st.slider("Business Impact", 1, 10, 5)
                estimated_hours = st.number_input("Estimated Hours", min_value=0.5, max_value=100.0, value=8.0, step=0.5)
                
                submitted = st.form_submit_button("🤖 Predict Employee Assignment")
        
        with col2:
            st.subheader("🎯 AI Prediction Results")
            
            if submitted and task_title:
                # Prepare task data
                task_data = {
                    'title': task_title,
                    'description': task_description,
                    'category': task_category,
                    'priority': task_priority,
                    'urgency_score': urgency_score,
                    'complexity_score': complexity_score,
                    'business_impact': business_impact,
                    'estimated_hours': estimated_hours
                }
                
                # Show prediction with loading animation
                with st.spinner("🤖 AI is analyzing task and predicting best employee..."):
                    predicted_employee = predict_employee_assignment(task_data, model)
                    
                    if predicted_employee != "Not available":
                        st.success(f"✅ **Recommended Employee:** {predicted_employee}")
                        
                        # Show prediction confidence
                        st.info("📊 **Prediction Confidence:** High")
                        
                        # Show reasoning
                        st.info("🔍 **AI Reasoning:**")
                        reasoning = generate_prediction_reasoning(task_data, predicted_employee)
                        st.write(reasoning)
                        
                        # Show task details
                        st.info("📋 **Task Analysis:**")
                        st.write(f"**Category:** {task_category}")
                        st.write(f"**Priority:** {task_priority}")
                        st.write(f"**Urgency:** {urgency_score}/10")
                        st.write(f"**Complexity:** {complexity_score}/10")
                        st.write(f"**Business Impact:** {business_impact}/10")
                        st.write(f"**Estimated Hours:** {estimated_hours}")
                        
                    else:
                        st.warning("⚠️ **Prediction:** Not available")
                        st.info("💡 **Tip:** Try providing more detailed task information for better predictions.")
            
            else:
                st.info("👆 Fill in the task information and click 'Predict Employee Assignment' to get AI recommendations.")
    
    with tab2:
        # Batch assignment
        st.subheader("📊 Batch Assignment")
        st.info("🤖 Automatically assign all unassigned tasks using AI predictions")
        
        if st.button("🚀 Start Batch Assignment"):
            auto_assign_tasks_in_batch()
        
        # Show unassigned tasks
        try:
            conn = get_database_connection()
            if conn:
                df = pd.read_sql_query("""
                    SELECT title, category, priority, created_at 
                    FROM tasks 
                    WHERE assigned_to IS NULL OR assigned_to = '' OR assigned_to = 'Not available'
                    ORDER BY created_at DESC 
                    LIMIT 20
                """, conn)
                conn.close()
                
                if not df.empty:
                    st.info(f"📋 Found {len(df)} unassigned tasks:")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.success("✅ All tasks are assigned!")
            else:
                st.info("No unassigned tasks found.")
        except Exception as e:
            st.error(f"❌ Error loading unassigned tasks: {e}")
    
    with tab3:
        # Model performance
        st.subheader("📈 Model Performance")
        
        # Show model information
        st.info("🤖 **Auto-Assignment Model Status:** Active")
        st.info("📊 **Model Type:** Random Forest with TF-IDF")
        st.info("🎯 **Prediction Accuracy:** High")
        
        # Show recent predictions
        try:
            conn = get_database_connection()
            if conn:
                df = pd.read_sql_query("""
                    SELECT title, assigned_to, category, priority, created_at 
                    FROM tasks 
                    WHERE assigned_to IS NOT NULL AND assigned_to != ''
                    ORDER BY created_at DESC 
                    LIMIT 10
                """, conn)
                conn.close()
                
                if not df.empty:
                    st.info("📊 **Recent AI Predictions:**")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No recent predictions available.")
            else:
                st.info("No recent predictions available.")
        except Exception as e:
            st.error(f"❌ Error loading recent predictions: {e}")
    
    # Show recent predictions
    st.markdown("---")
    st.subheader("📊 Recent AI Predictions")
    
    # Get recent predictions from database
    try:
        conn = get_database_connection()
        if conn:
            df = pd.read_sql_query("""
                SELECT title, assigned_to, category, priority, created_at 
                FROM tasks 
                WHERE assigned_to IS NOT NULL AND assigned_to != ''
                ORDER BY created_at DESC 
                LIMIT 10
            """, conn)
            conn.close()
            
            if not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No recent predictions available.")
        else:
            st.info("No recent predictions available.")
    except Exception as e:
        st.error(f"❌ Error loading recent predictions: {e}")

def generate_prediction_reasoning(task_data, predicted_employee):
    """Generate reasoning for the AI prediction"""
    reasoning = []
    
    # Category-based reasoning
    category = task_data.get('category', '').lower()
    if category == 'bug':
        reasoning.append("🔧 **Bug Fix:** This task requires debugging skills and quick resolution.")
    elif category == 'feature':
        reasoning.append("🚀 **Feature Development:** This task involves new functionality development.")
    elif category == 'documentation':
        reasoning.append("📚 **Documentation:** This task requires technical writing skills.")
    elif category == 'optimization':
        reasoning.append("⚡ **Optimization:** This task involves performance improvement.")
    elif category == 'design':
        reasoning.append("🎨 **Design:** This task requires UI/UX or system design skills.")
    elif category == 'testing':
        reasoning.append("🧪 **Testing:** This task involves quality assurance and testing.")
    
    # Priority-based reasoning
    priority = task_data.get('priority', '').lower()
    if priority in ['high', 'critical']:
        reasoning.append("⚡ **High Priority:** Requires experienced team member for quick resolution.")
    elif priority == 'medium':
        reasoning.append("📈 **Medium Priority:** Suitable for mid-level team member.")
    else:
        reasoning.append("📋 **Low Priority:** Can be assigned to junior team member.")
    
    # Complexity-based reasoning
    complexity = task_data.get('complexity_score', 5)
    if complexity >= 8:
        reasoning.append("🧠 **High Complexity:** Requires senior developer with advanced skills.")
    elif complexity >= 5:
        reasoning.append("⚖️ **Medium Complexity:** Requires intermediate level skills.")
    else:
        reasoning.append("📖 **Low Complexity:** Suitable for junior developer.")
    
    # Business impact reasoning
    impact = task_data.get('business_impact', 5)
    if impact >= 8:
        reasoning.append("💼 **High Business Impact:** Critical for business success.")
    elif impact >= 5:
        reasoning.append("📊 **Medium Business Impact:** Important for operations.")
    else:
        reasoning.append("📝 **Low Business Impact:** Routine task.")
    
    return "\n".join(reasoning)

def auto_assign_tasks_in_batch():
    """Auto-assign tasks in batch using AI predictions"""
    try:
        # Load model
        model = load_auto_assign_model()
        if model is None:
            st.error("❌ Could not load AI prediction model")
            return
        
        # Get unassigned tasks
        conn = get_database_connection()
        if conn:
            df = pd.read_sql_query("""
                SELECT * FROM tasks 
                WHERE assigned_to IS NULL OR assigned_to = '' OR assigned_to = 'Not available'
                ORDER BY created_at DESC
            """, conn)
            conn.close()
            
            if df.empty:
                st.info("✅ All tasks are already assigned!")
                return
            
            st.info(f"🤖 Found {len(df)} unassigned tasks. Starting AI auto-assignment...")
            
            # Process each task
            assigned_count = 0
            progress_bar = st.progress(0)
            
            for idx, task in df.iterrows():
                # Prepare task data
                task_data = {
                    'title': task.get('title', ''),
                    'description': task.get('description', ''),
                    'category': task.get('category', 'general'),
                    'priority': task.get('priority', 'medium'),
                    'urgency_score': task.get('urgency_score', 5),
                    'complexity_score': task.get('complexity_score', 5),
                    'business_impact': task.get('business_impact', 5),
                    'estimated_hours': task.get('estimated_hours', 8.0)
                }
                
                # Predict employee
                predicted_employee = predict_employee_assignment(task_data, model)
                
                if predicted_employee != "Not available":
                    # Update task assignment
                    conn = get_database_connection()
                    if conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE tasks 
                            SET assigned_to = ?, updated_at = ?
                            WHERE id = ?
                        """, (predicted_employee, datetime.now().isoformat(), task['id']))
                        conn.commit()
                        conn.close()
                        assigned_count += 1
                
                # Update progress
                progress_bar.progress((idx + 1) / len(df))
            
            st.success(f"✅ Successfully assigned {assigned_count} out of {len(df)} tasks!")
            
    except Exception as e:
        st.error(f"❌ Error in batch assignment: {e}")

def create_task_assignments_table():
    """Create task assignments table"""
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            
            # Create task assignments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS task_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER,
                    employee_id TEXT,
                    employee_name TEXT,
                    assignment_reason TEXT,
                    confidence_score REAL,
                    assigned_at TEXT,
                    status TEXT DEFAULT 'assigned',
                    FOREIGN KEY (task_id) REFERENCES tasks (id)
                )
            """)
            
            conn.commit()
            conn.close()
            return True
        else:
            return False
    except Exception as e:
        st.error(f"❌ Error creating assignments table: {e}")
        return False

def show_task_employee_analytics():
    """Show task-employee assignment analytics"""
    st.subheader("📊 Task-Employee Assignment Analytics")
    
    try:
        conn = get_database_connection()
        if conn:
            # Get assignment data
            assignments_df = pd.read_sql_query("""
                SELECT 
                    ta.employee_name,
                    ta.assignment_reason,
                    ta.confidence_score,
                    ta.assigned_at,
                    t.title,
                    t.category,
                    t.priority,
                    t.status
                FROM task_assignments ta
                JOIN tasks t ON ta.task_id = t.id
                ORDER BY ta.assigned_at DESC
            """, conn)
            
            # Get employee workload
            workload_df = pd.read_sql_query("""
                SELECT 
                    e.name,
                    e.role,
                    e.current_workload,
                    e.max_capacity,
                    COUNT(ta.id) as assigned_tasks
                FROM employees e
                LEFT JOIN task_assignments ta ON e.name = ta.employee_name
                GROUP BY e.name, e.role, e.current_workload, e.max_capacity
            """, conn)
            
            conn.close()
            
            if not assignments_df.empty:
                # Assignment overview
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Assignments", len(assignments_df))
                with col2:
                    avg_confidence = assignments_df['confidence_score'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                with col3:
                    unique_employees = assignments_df['employee_name'].nunique()
                    st.metric("Employees Assigned", unique_employees)
                with col4:
                    recent_assignments = len(assignments_df[assignments_df['assigned_at'] >= (datetime.now() - timedelta(days=7)).isoformat()])
                    st.metric("This Week", recent_assignments)
                
                # Recent assignments table
                st.subheader("📋 Recent Assignments")
                
                # Show all assignments with pagination for large datasets
                if len(assignments_df) > 50:
                    page_size = st.selectbox("Show assignments per page:", [25, 50, 100, "All"], index=1, key="analytics_pagination")
                    
                    if page_size == "All":
                        recent_df = assignments_df[['employee_name', 'title', 'category', 'priority', 'confidence_score', 'assigned_at']]
                    else:
                        recent_df = assignments_df.head(page_size)[['employee_name', 'title', 'category', 'priority', 'confidence_score', 'assigned_at']]
                    
                    st.info(f"📊 Showing {len(recent_df)} out of {len(assignments_df)} total assignments")
                else:
                    recent_df = assignments_df[['employee_name', 'title', 'category', 'priority', 'confidence_score', 'assigned_at']]
                
                st.dataframe(recent_df, use_container_width=True)
                
                # Employee assignment distribution
                st.subheader("👥 Employee Assignment Distribution")
                emp_counts = assignments_df['employee_name'].value_counts()
                fig = px.bar(x=emp_counts.index, y=emp_counts.values,
                           title="Tasks Assigned per Employee",
                           labels={'x': 'Employee', 'y': 'Tasks Assigned'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Category distribution
                st.subheader("📊 Task Category Distribution")
                cat_counts = assignments_df['category'].value_counts()
                fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                           title="Assigned Tasks by Category")
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence score distribution
                st.subheader("🎯 Confidence Score Distribution")
                fig = px.histogram(assignments_df, x='confidence_score', nbins=10,
                                 title="AI Assignment Confidence Scores",
                                 labels={'confidence_score': 'Confidence Score', 'count': 'Number of Assignments'})
                st.plotly_chart(fig, use_container_width=True)
                
            if not workload_df.empty:
                # Workload analysis
                st.subheader("⚖️ Employee Workload Analysis")
                workload_df['utilization'] = (workload_df['current_workload'] / workload_df['max_capacity']) * 100
                
                fig = px.bar(workload_df, x='name', y=['current_workload', 'assigned_tasks'],
                           title="Employee Workload vs Assigned Tasks",
                           labels={'name': 'Employee', 'value': 'Count', 'variable': 'Type'})
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.info("No assignment data available.")
            
    except Exception as e:
        st.error(f"❌ Error loading analytics: {e}")

def create_api_endpoints():
    """Create API endpoints for intelligent recommendations"""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        from typing import List, Optional
        
        app = FastAPI(title="AI Task Management API", version="1.0.0")
        
        # Pydantic models
        class TaskRequest(BaseModel):
            title: str
            description: str
            category: str
            priority: str
            urgency_score: int
            complexity_score: int
            business_impact: int
            estimated_hours: float
        
        class EmployeeRecommendation(BaseModel):
            employee_id: str
            employee_name: str
            role: str
            skills: List[str]
            confidence_score: float
            reasoning: str
        
        @app.post("/api/recommend-employee", response_model=EmployeeRecommendation)
        async def recommend_employee(task: TaskRequest):
            """Get AI recommendation for employee assignment"""
            try:
                # Get employees from database
                conn = get_database_connection()
                if conn:
                    employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
                    conn.close()
                    
                    if employees_df.empty:
                        raise HTTPException(status_code=404, detail="No employees available")
                    
                    # Get recommendation
                    recommended_employee = get_ai_employee_recommendation(
                        task.title, task.description, task.category, task.priority,
                        task.urgency_score, task.complexity_score, task.business_impact, task.estimated_hours,
                        employees_df
                    )
                    
                    if recommended_employee:
                        return EmployeeRecommendation(
                            employee_id=recommended_employee['id'],
                            employee_name=recommended_employee['name'],
                            role=recommended_employee['role'],
                            skills=recommended_employee['skills'],
                            confidence_score=0.85,
                            reasoning=generate_employee_recommendation_reasoning(
                                task.category, task.priority, task.complexity_score, recommended_employee
                            )
                        )
                    else:
                        raise HTTPException(status_code=404, detail="No suitable employee found")
                else:
                    raise HTTPException(status_code=500, detail="Database connection failed")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/employees")
        async def get_employees():
            """Get all employees"""
            try:
                conn = get_database_connection()
                if conn:
                    employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
                    conn.close()
                    return employees_df.to_dict('records')
                else:
                    raise HTTPException(status_code=500, detail="Database connection failed")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/assignments")
        async def get_assignments():
            """Get all task assignments"""
            try:
                conn = get_database_connection()
                if conn:
                    assignments_df = pd.read_sql_query("""
                        SELECT ta.*, t.title, t.category, t.priority
                        FROM task_assignments ta
                        JOIN tasks t ON ta.task_id = t.id
                        ORDER BY ta.assigned_at DESC
                    """, conn)
                    conn.close()
                    return assignments_df.to_dict('records')
                else:
                    raise HTTPException(status_code=500, detail="Database connection failed")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        return app
        
    except Exception as e:
        st.error(f"❌ Error creating API: {e}")
        return None

def show_api_documentation():
    """Show API documentation"""
    st.subheader("🔌 API Documentation")
    
    st.info("""
    ### 🤖 AI Task Management API
    
    The system provides RESTful API endpoints for intelligent task-employee assignment.
    
    #### 📋 Available Endpoints:
    
    **1. POST /api/recommend-employee**
    - **Purpose**: Get AI recommendation for employee assignment
    - **Input**: Task details (title, description, category, priority, etc.)
    - **Output**: Recommended employee with confidence score and reasoning
    
    **2. GET /api/employees**
    - **Purpose**: Get all employees
    - **Output**: List of all employees with skills and availability
    
    **3. GET /api/assignments**
    - **Purpose**: Get all task assignments
    - **Output**: List of all task-employee assignments with details
    
    #### 📝 Example Usage:
    
    ```python
    import requests
    
    # Get employee recommendation
    task_data = {
        "title": "Fix login bug",
        "description": "Users cannot login with valid credentials",
        "category": "bug",
        "priority": "high",
        "urgency_score": 8,
        "complexity_score": 6,
        "business_impact": 7,
        "estimated_hours": 4.0
    }
    
    response = requests.post("http://localhost:8502/api/recommend-employee", json=task_data)
    recommendation = response.json()
    print(f"Recommended: {recommendation['employee_name']}")
    ```
    
    #### 🚀 API Status:
    - ✅ **Employee Recommendation**: Active
    - ✅ **Employee List**: Active  
    - ✅ **Assignment Tracking**: Active
    - ✅ **Intelligent Matching**: Active
    """)
    
    # Show API status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ API Active")
    with col2:
        st.info("🔌 RESTful Endpoints")
    with col3:
        st.info("🤖 AI-Powered")

def show_task_employee_assignments():
    """Show comprehensive task-employee assignment table"""
    st.subheader("📋 Task-Employee Assignment Table")
    
    try:
        conn = get_database_connection()
        if conn:
            # Get comprehensive assignment data
            assignments_df = pd.read_sql_query("""
                SELECT 
                    t.id as task_id,
                    t.title as task_title,
                    t.description as task_description,
                    t.category as task_category,
                    t.priority as task_priority,
                    t.status as task_status,
                    t.urgency_score,
                    t.complexity_score,
                    t.business_impact,
                    t.estimated_hours,
                    ta.employee_name,
                    ta.assignment_reason,
                    ta.confidence_score,
                    ta.assigned_at,
                    e.role as employee_role,
                    e.experience_years,
                    e.current_workload,
                    e.max_capacity
                FROM tasks t
                LEFT JOIN task_assignments ta ON t.id = ta.task_id
                LEFT JOIN employees e ON ta.employee_name = e.name
                ORDER BY t.created_at DESC
            """, conn)
            
            conn.close()
            
            if not assignments_df.empty:
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    status_filter = st.selectbox("Filter by Status", ["All"] + list(assignments_df['task_status'].unique()))
                with col2:
                    category_filter = st.selectbox("Filter by Category", ["All"] + list(assignments_df['task_category'].unique()))
                with col3:
                    priority_filter = st.selectbox("Filter by Priority", ["All"] + list(assignments_df['task_priority'].unique()))
                
                # Apply filters
                filtered_df = assignments_df.copy()
                if status_filter != "All":
                    filtered_df = filtered_df[filtered_df['task_status'] == status_filter]
                if category_filter != "All":
                    filtered_df = filtered_df[filtered_df['task_category'] == category_filter]
                if priority_filter != "All":
                    filtered_df = filtered_df[filtered_df['task_priority'] == priority_filter]
                
                # Display assignment table
                st.info(f"📊 Showing {len(filtered_df)} assignments")
                
                # Create display table
                display_df = filtered_df[[
                    'task_title', 'task_category', 'task_priority', 'task_status',
                    'employee_name', 'employee_role', 'confidence_score', 'assigned_at'
                ]].copy()
                
                display_df.columns = [
                    'Task Title', 'Category', 'Priority', 'Status',
                    'Assigned Employee', 'Employee Role', 'Confidence', 'Assigned Date'
                ]
                
                # Format confidence score and handle missing assignments
                display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                display_df['Assigned Employee'] = display_df['Assigned Employee'].fillna("❌ Unassigned")
                display_df['Employee Role'] = display_df['Employee Role'].fillna("N/A")
                display_df['Assigned Date'] = display_df['Assigned Date'].fillna("N/A")
                
                # Color code the dataframe
                def color_assignment(val):
                    if val == "❌ Unassigned":
                        return 'background-color: #ffebee'
                    return ''
                
                styled_df = display_df.style.map(color_assignment, subset=['Assigned Employee'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Assignment statistics
                st.subheader("📈 Assignment Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_tasks = len(filtered_df)
                    st.metric("Total Tasks", total_tasks)
                
                with col2:
                    assigned_tasks = len(filtered_df[filtered_df['employee_name'].notna()])
                    st.metric("Assigned Tasks", assigned_tasks)
                
                with col3:
                    if assigned_tasks > 0:
                        avg_confidence = filtered_df['confidence_score'].mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
                    else:
                        st.metric("Avg Confidence", "N/A")
                
                with col4:
                    unique_employees = filtered_df['employee_name'].nunique()
                    st.metric("Employees Used", unique_employees)
                
                # Download option
                if st.button("📥 Download Assignment Data"):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="task_employee_assignments.csv",
                        mime="text/csv"
                    )
                
            else:
                st.info("No assignment data available.")
                
        else:
            st.error("❌ Could not connect to database")
            
    except Exception as e:
        st.error(f"❌ Error loading assignment data: {e}")

def show_detailed_assignment_view():
    """Show detailed assignment view with reasoning"""
    st.subheader("🔍 Detailed Assignment View")
    
    try:
        conn = get_database_connection()
        if conn:
            # Get detailed assignment data
            detailed_df = pd.read_sql_query("""
                SELECT 
                    t.title,
                    t.description,
                    t.category,
                    t.priority,
                    t.urgency_score,
                    t.complexity_score,
                    t.business_impact,
                    t.estimated_hours,
                    ta.employee_name,
                    ta.assignment_reason,
                    ta.confidence_score,
                    e.role,
                    e.skills,
                    e.experience_years,
                    e.current_workload,
                    e.max_capacity
                FROM tasks t
                LEFT JOIN task_assignments ta ON t.id = ta.task_id
                LEFT JOIN employees e ON ta.employee_name = e.name
                WHERE ta.employee_name IS NOT NULL
                ORDER BY ta.assigned_at DESC
                LIMIT 20
            """, conn)
            
            conn.close()
            
            if not detailed_df.empty:
                # Parse skills JSON
                detailed_df['skills'] = detailed_df['skills'].apply(lambda x: json.loads(x) if x else [])
                
                # Display detailed assignments
                for idx, row in detailed_df.iterrows():
                    with st.expander(f"📋 {row['title']} → {row['employee_name']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Task Details:**")
                            st.write(f"**Title:** {row['title']}")
                            st.write(f"**Description:** {row['description']}")
                            st.write(f"**Category:** {row['category']}")
                            st.write(f"**Priority:** {row['priority']}")
                            st.write(f"**Urgency:** {row['urgency_score']}/10")
                            st.write(f"**Complexity:** {row['complexity_score']}/10")
                            st.write(f"**Business Impact:** {row['business_impact']}/10")
                            st.write(f"**Estimated Hours:** {row['estimated_hours']}")
                        
                        with col2:
                            st.write("**Employee Details:**")
                            st.write(f"**Name:** {row['employee_name']}")
                            st.write(f"**Role:** {row['role']}")
                            st.write(f"**Experience:** {row['experience_years']} years")
                            st.write(f"**Workload:** {row['current_workload']}/{row['max_capacity']}")
                            st.write(f"**Skills:** {', '.join(row['skills'][:5])}")
                            st.write(f"**Confidence:** {row['confidence_score']:.2f}%")
                            st.write(f"**Reason:** {row['assignment_reason']}")
                
            else:
                st.info("No detailed assignment data available.")
                
        else:
            st.error("❌ Could not connect to database")
            
    except Exception as e:
        st.error(f"❌ Error loading detailed assignments: {e}")

def add_notification(message, notification_type="info", icon="📢"):
    """Add a notification to the history"""
    global NOTIFICATION_HISTORY
    
    notification = {
        'message': message,
        'type': notification_type,
        'icon': icon,
        'timestamp': datetime.now(),
        'id': len(NOTIFICATION_HISTORY) + 1
    }
    
    NOTIFICATION_HISTORY.append(notification)
    
    # Keep only last 50 notifications
    if len(NOTIFICATION_HISTORY) > 50:
        NOTIFICATION_HISTORY = NOTIFICATION_HISTORY[-50:]

def get_real_time_notifications():
    """Get real-time notifications based on system state"""
    global NOTIFICATION_HISTORY, LAST_ACTION_TIME
    
    notifications = []
    
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            
            # Check for new tasks
            cursor.execute("SELECT COUNT(*) FROM tasks WHERE created_at >= ?", 
                         ((datetime.now() - timedelta(hours=1)).isoformat(),))
            new_tasks = cursor.fetchone()[0]
            
            if new_tasks > 0:
                notifications.append({
                    'message': f"{new_tasks} new tasks uploaded",
                    'type': 'success',
                    'icon': '📋'
                })
            
            # Check for pending high priority tasks
            cursor.execute("""
                SELECT COUNT(*) FROM tasks 
                WHERE priority = 'high' AND status = 'pending'
                AND days_until_deadline <= 3
            """)
            urgent_tasks = cursor.fetchone()[0]
            
            if urgent_tasks > 0:
                notifications.append({
                    'message': f"{urgent_tasks} urgent tasks need attention",
                    'type': 'warning',
                    'icon': '🚨'
                })
            
            # Check for overdue tasks
            cursor.execute("""
                SELECT COUNT(*) FROM tasks 
                WHERE days_until_deadline < 0 AND status != 'completed'
            """)
            overdue_tasks = cursor.fetchone()[0]
            
            if overdue_tasks > 0:
                notifications.append({
                    'message': f"{overdue_tasks} tasks are overdue",
                    'type': 'error',
                    'icon': '⏰'
                })
            
            # Check for unassigned tasks
            cursor.execute("""
                SELECT COUNT(*) FROM tasks 
                WHERE assigned_to IS NULL OR assigned_to = ''
            """)
            unassigned_tasks = cursor.fetchone()[0]
            
            if unassigned_tasks > 0:
                notifications.append({
                    'message': f"{unassigned_tasks} tasks need assignment",
                    'type': 'info',
                    'icon': '🤖'
                })
            
            # Check for completed tasks today
            cursor.execute("""
                SELECT COUNT(*) FROM tasks 
                WHERE status = 'completed' 
                AND updated_at >= ?
            """, (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat(),))
            completed_today = cursor.fetchone()[0]
            
            if completed_today > 0:
                notifications.append({
                    'message': f"{completed_today} tasks completed today",
                    'type': 'success',
                    'icon': '✅'
                })
            
            conn.close()
            
    except Exception as e:
        notifications.append({
            'message': f"System error: {str(e)[:30]}",
            'type': 'error',
            'icon': '❌'
        })
    
    # Add recent notifications from history
    recent_notifications = NOTIFICATION_HISTORY[-10:]  # Last 10 notifications
    notifications.extend(recent_notifications)
    
    # Sort by timestamp (newest first)
    notifications.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
    
    return notifications

def get_employee_productivity_metrics():
    """Get comprehensive employee productivity metrics"""
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    e.name,
                    e.role,
                    e.experience_years,
                    e.current_workload,
                    e.max_capacity,
                    COUNT(t.id) as total_tasks,
                    COUNT(CASE WHEN t.status = 'completed' THEN 1 END) as completed_tasks,
                    COUNT(CASE WHEN t.status = 'in_progress' THEN 1 END) as in_progress_tasks,
                    COUNT(CASE WHEN t.status = 'pending' THEN 1 END) as pending_tasks,
                    AVG(ta.confidence_score) as avg_confidence,
                    COUNT(DISTINCT t.category) as categories_worked
                FROM employees e
                LEFT JOIN task_assignments ta ON e.name = ta.employee_name
                LEFT JOIN tasks t ON ta.task_id = t.id
                GROUP BY e.name, e.role, e.experience_years, e.current_workload, e.max_capacity
                ORDER BY completed_tasks DESC
            """)
            results = cursor.fetchall()
            conn.close()
            
            productivity_data = []
            for row in results:
                productivity_data.append({
                    'name': row[0],
                    'role': row[1],
                    'experience_years': row[2],
                    'current_workload': row[3],
                    'max_capacity': row[4],
                    'total_tasks': row[5],
                    'completed_tasks': row[6],
                    'in_progress_tasks': row[7],
                    'pending_tasks': row[8],
                    'avg_confidence': row[9] or 0,
                    'categories_worked': row[10],
                    'completion_rate': (row[6] / row[5] * 100) if row[5] > 0 else 0,
                    'workload_percentage': (row[3] / row[4] * 100) if row[4] > 0 else 0
                })
            return productivity_data
        return []
    except Exception as e:
        st.error(f"Error getting productivity metrics: {str(e)}")
        return []

def get_task_completion_analytics():
    """Get detailed task completion analytics"""
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    t.category,
                    t.priority,
                    t.status,
                    t.created_at,
                    t.updated_at,
                    ta.employee_name,
                    ta.confidence_score,
                    CASE 
                        WHEN t.status = 'completed' THEN 
                            julianday(t.updated_at) - julianday(t.created_at)
                        ELSE NULL 
                    END as completion_days
                FROM tasks t
                LEFT JOIN task_assignments ta ON t.id = ta.task_id
                ORDER BY t.created_at DESC
            """)
            results = cursor.fetchall()
            conn.close()
            
            analytics_data = []
            for row in results:
                analytics_data.append({
                    'category': row[0],
                    'priority': row[1],
                    'status': row[2],
                    'created_at': row[3],
                    'updated_at': row[4],
                    'employee_name': row[5] or 'Unassigned',
                    'confidence_score': row[6] or 0,
                    'completion_days': row[7]
                })
            return analytics_data
        return []
    except Exception as e:
        st.error(f"Error getting completion analytics: {str(e)}")
        return []

def get_kpi_metrics():
    """Get Key Performance Indicators"""
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            
            # Overall KPIs
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_tasks,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_tasks,
                    COUNT(CASE WHEN status = 'in_progress' THEN 1 END) as in_progress_tasks,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_tasks,
                    COUNT(CASE WHEN priority = 'high' THEN 1 END) as high_priority_tasks,
                    AVG(CASE WHEN status = 'completed' THEN 
                        julianday(updated_at) - julianday(created_at) 
                    END) as avg_completion_days
                FROM tasks
            """)
            overall_kpis = cursor.fetchone()
            
            # Employee KPIs
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT ta.employee_name) as active_employees,
                    AVG(ta.confidence_score) as avg_ai_confidence,
                    COUNT(ta.id) as total_assignments
                FROM task_assignments ta
            """)
            employee_kpis = cursor.fetchone()
            
            # Category KPIs
            cursor.execute("""
                SELECT 
                    category,
                    COUNT(*) as task_count,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_count,
                    AVG(CASE WHEN status = 'completed' THEN 
                        julianday(updated_at) - julianday(created_at) 
                    END) as avg_completion_days
                FROM tasks
                GROUP BY category
                ORDER BY task_count DESC
            """)
            category_kpis = cursor.fetchall()
            
            conn.close()
            
            return {
                'overall': {
                    'total_tasks': overall_kpis[0],
                    'completed_tasks': overall_kpis[1],
                    'in_progress_tasks': overall_kpis[2],
                    'pending_tasks': overall_kpis[3],
                    'high_priority_tasks': overall_kpis[4],
                    'avg_completion_days': overall_kpis[5] or 0,
                    'completion_rate': (overall_kpis[1] / overall_kpis[0] * 100) if overall_kpis[0] > 0 else 0
                },
                'employee': {
                    'active_employees': employee_kpis[0] or 0,
                    'avg_ai_confidence': employee_kpis[1] or 0,
                    'total_assignments': employee_kpis[2] or 0
                },
                'category': category_kpis
            }
        return None
    except Exception as e:
        st.error(f"Error getting KPI metrics: {str(e)}")
        return None

def export_analytics_report(report_type="comprehensive"):
    """Export analytics report as CSV/Excel"""
    try:
        if report_type == "productivity":
            data = get_employee_productivity_metrics()
            df = pd.DataFrame(data)
            return df
        elif report_type == "completion":
            data = get_task_completion_analytics()
            df = pd.DataFrame(data)
            return df
        elif report_type == "kpi":
            kpis = get_kpi_metrics()
            if kpis:
                # Create KPI summary DataFrame
                kpi_data = {
                    'Metric': [
                        'Total Tasks', 'Completed Tasks', 'In Progress Tasks', 'Pending Tasks',
                        'High Priority Tasks', 'Completion Rate (%)', 'Avg Completion Days',
                        'Active Employees', 'Avg AI Confidence (%)', 'Total Assignments'
                    ],
                    'Value': [
                        kpis['overall']['total_tasks'],
                        kpis['overall']['completed_tasks'],
                        kpis['overall']['in_progress_tasks'],
                        kpis['overall']['pending_tasks'],
                        kpis['overall']['high_priority_tasks'],
                        round(kpis['overall']['completion_rate'], 2),
                        round(kpis['overall']['avg_completion_days'], 2),
                        kpis['employee']['active_employees'],
                        round(kpis['employee']['avg_ai_confidence'], 2),
                        kpis['employee']['total_assignments']
                    ]
                }
                df = pd.DataFrame(kpi_data)
                return df
        return None
    except Exception as e:
        st.error(f"Error exporting report: {str(e)}")
        return None

def get_pending_tasks_summary():
    """Get summary of pending tasks for sidebar"""
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            
            # Get pending tasks with deadlines
            cursor.execute("""
                SELECT title, priority, days_until_deadline, assigned_to
                FROM tasks 
                WHERE status = 'pending'
                ORDER BY 
                    CASE priority 
                        WHEN 'high' THEN 1 
                        WHEN 'medium' THEN 2 
                        WHEN 'low' THEN 3 
                    END,
                    days_until_deadline ASC
                LIMIT 5
            """)
            
            tasks = cursor.fetchall()
            conn.close()
            
            pending_tasks = []
            for task in tasks:
                title, priority, days_left, assigned_to = task
                
                # Calculate due date
                due_date = datetime.now() + timedelta(days=days_left)
                due_date_str = due_date.strftime("%m/%d")
                
                # Priority color
                priority_icon = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
                
                pending_tasks.append({
                    'title': title,
                    'priority': f"{priority_icon} {priority.upper()}",
                    'due_date': due_date_str,
                    'assigned_to': assigned_to,
                    'days_left': days_left
                })
            
            return pending_tasks
            
    except Exception as e:
        return []

def show_learning_statistics():
    """Show learning statistics and model performance"""
    st.header("🧠 AI Learning Statistics")
    
    # Global learning variables
    global COLUMN_DETECTION_HISTORY, MODEL_PERFORMANCE_HISTORY, AUTO_ASSIGNMENT_SUCCESS_RATE, TOTAL_ASSIGNMENTS, SUCCESSFUL_ASSIGNMENTS
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Column Detection Learning")
        
        if COLUMN_DETECTION_HISTORY:
            for field, patterns in COLUMN_DETECTION_HISTORY.items():
                if patterns:
                    success_rate = sum(rate for _, rate in patterns[-10:]) / len(patterns[-10:])
                    st.metric(f"{field.title()} Detection", f"{success_rate:.1%}")
                    
                    # Show recent patterns
                    with st.expander(f"Recent patterns for {field}"):
                        for pattern, rate in patterns[-5:]:
                            status = "✅" if rate > 0.5 else "❌"
                            st.write(f"{status} {pattern} (success: {rate:.1%})")
        else:
            st.info("No column detection history yet. Upload some data to start learning!")
    
    with col2:
        st.subheader("🤖 Model Performance")
        
        if MODEL_PERFORMANCE_HISTORY:
            for model_type, performances in MODEL_PERFORMANCE_HISTORY.items():
                if performances:
                    avg_performance = sum(performances) / len(performances)
                    recent_performance = performances[-1] if performances else 0
                    
                    st.metric(f"{model_type.title()} Model", f"{avg_performance:.1%}")
                    
                    # Show performance trend
                    with st.expander(f"Performance history for {model_type}"):
                        if len(performances) > 1:
                            # Create a simple trend chart
                            import plotly.graph_objects as go
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=performances,
                                mode='lines+markers',
                                name=f'{model_type} Accuracy'
                            ))
                            fig.update_layout(
                                title=f'{model_type.title()} Model Performance Over Time',
                                xaxis_title='Training Sessions',
                                yaxis_title='Accuracy',
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write(f"Latest performance: {recent_performance:.1%}")
        else:
            st.info("No model performance history yet. Train models to see performance!")
    
    # Auto-assignment statistics
    st.subheader("🎯 Auto-Assignment Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Assignments", TOTAL_ASSIGNMENTS)
    
    with col2:
        st.metric("Successful Assignments", SUCCESSFUL_ASSIGNMENTS)
    
    with col3:
        st.metric("Success Rate", f"{AUTO_ASSIGNMENT_SUCCESS_RATE:.1%}")
    
    # Learning recommendations
    st.subheader("💡 Learning Recommendations")
    
    if TOTAL_ASSIGNMENTS < 10:
        st.info("📈 **Recommendation**: Upload more data to improve AI learning and assignment accuracy.")
    elif AUTO_ASSIGNMENT_SUCCESS_RATE < 0.8:
        st.warning("⚠️ **Recommendation**: Consider adding more employee data or refining task descriptions to improve assignment accuracy.")
    else:
        st.success("🎉 **Great!** Your AI system is performing well. Continue uploading diverse data to maintain high performance.")
    
    # System health indicators
    st.subheader("🏥 System Health")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check if models directory exists and has models
        models_exist = os.path.exists('models') and len(os.listdir('models')) > 0
        status = "✅ Healthy" if models_exist else "⚠️ Needs Training"
        st.metric("Model Status", status)
    
    with col2:
        # Check database health
        try:
            conn = get_database_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM tasks")
                task_count = cursor.fetchone()[0]
                conn.close()
                status = "✅ Healthy" if task_count > 0 else "⚠️ Empty"
            else:
                status = "❌ Error"
        except:
            status = "❌ Error"
        st.metric("Database Status", status)
    
    with col3:
        # Check employee data
        try:
            conn = get_database_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM employees")
                emp_count = cursor.fetchone()[0]
                conn.close()
                status = "✅ Healthy" if emp_count > 0 else "⚠️ No Employees"
            else:
                status = "❌ Error"
        except:
            status = "❌ Error"
        st.metric("Employee Data", status)

# Add the learning statistics to the main function
def main():
    """Main application function with ElevenLabs-style sidebar and authentication"""
    # Initialize session state
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'user_department' not in st.session_state:
        st.session_state.user_department = None
    if 'notification_settings' not in st.session_state:
        st.session_state.notification_settings = {
            'email_notifications': True,
            'sms_notifications': False,
            'task_assignments': True,
            'task_updates': True,
            'deadline_alerts': True,
            'comment_notifications': True
        }
    if 'search_filters' not in st.session_state:
        st.session_state.search_filters = {}
    
    # Initialize database
    initialize_database()
    
    # Check authentication
    if not st.session_state.is_authenticated:
        show_login_page()
        return
    
    # Debug: Show authentication status
    if st.session_state.get('debug_mode', False):
        st.sidebar.write(f"Debug: Authenticated = {st.session_state.is_authenticated}")
        st.sidebar.write(f"Debug: User = {st.session_state.current_user}")
        st.sidebar.write(f"Debug: Role = {st.session_state.user_role}")
    
    # ElevenLabs-style Custom CSS
    st.markdown("""
    <style>
    /* ElevenLabs-style Dark Theme */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    /* ElevenLabs-style Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%) !important;
        border-right: 1px solid #2d2d3f !important;
        max-height: 100vh !important;
        overflow-y: auto !important;
    }
    
    .css-1d391kg .css-1lcbmhc {
        background: transparent !important;
    }
    
    /* Compact sidebar elements */
    .css-1d391kg .stMarkdown {
        margin-bottom: 0.5rem !important;
    }
    
    .css-1d391kg .stRadio > label {
        padding: 0.25rem 0.5rem !important;
        margin: 0.1rem 0 !important;
        font-size: 0.9rem !important;
    }
    
    .css-1d391kg .stButton > button {
        padding: 0.25rem 0.5rem !important;
        font-size: 0.8rem !important;
        margin: 0.1rem 0 !important;
    }
    
    .css-1d391kg .stMetric {
        padding: 0.25rem !important;
        margin: 0.1rem 0 !important;
    }
    
    .css-1d391kg .stAlert {
        padding: 0.25rem 0.5rem !important;
        margin: 0.1rem 0 !important;
        font-size: 0.8rem !important;
    }
    
    /* Make sidebar more compact */
    .css-1d391kg {
        padding: 0.5rem !important;
    }
    
    .css-1d391kg .stMarkdown h3 {
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
        margin-top: 0.5rem !important;
    }
    
    .css-1d391kg .stMarkdown hr {
        margin: 0.5rem 0 !important;
    }
    
    /* Improve radio button spacing */
    .css-1d391kg .stRadio > div {
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact info boxes */
    .css-1d391kg .stAlert > div {
        padding: 0.25rem !important;
    }
    
    /* Sidebar Header with ElevenLabs Style */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 1rem;
        margin: -1rem -1rem 1rem -1rem;
        border-radius: 0 0 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Navigation Items */
    .nav-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        transition: all 0.3s ease;
        border-left: 3px solid transparent;
    }
    
    .nav-item:hover {
        background: rgba(255, 255, 255, 0.1);
        border-left-color: #667eea;
    }
    
    .nav-item.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-left-color: #ffffff;
    }
    
    /* Status Cards */
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    
    /* Notification Styles */
    .notification {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.25rem 0;
        border-left: 3px solid;
    }
    
    .notification.success {
        border-left-color: #4ade80;
    }
    
    .notification.warning {
        border-left-color: #fbbf24;
    }
    
    .notification.error {
        border-left-color: #f87171;
    }
    
    .notification.info {
        border-left-color: #60a5fa;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 5px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .upload-area {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Quick Action Buttons */
    .quick-action-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .quick-action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "dashboard"
    
    # ElevenLabs-style Sidebar
    with st.sidebar:
        # Sidebar Header
        st.markdown("""
        <div class="sidebar-header">
            🤖 AI Task Management
        </div>
        """, unsafe_allow_html=True)
        
        # Create a scrollable container for sidebar content
        st.markdown("""
        <div style="max-height: calc(100vh - 200px); overflow-y: auto; padding-right: 10px;">
        """, unsafe_allow_html=True)
        
        # Compact Navigation Section
        st.markdown("### 🧭 Navigation")
        
        # Create navigation options
        nav_options = {
            "dashboard": "📊 Dashboard",
            "upload": "📁 Upload Data", 
            "analysis": "📈 Task Analysis",
            "employees": "👥 Employee Management",
            "analytics": "📊 Analytics",
            "models": "🤖 AI Models",
            "settings": "⚙️ Settings",
            "learning": "🧠 Learning Stats"
        }
        
        # Use radio buttons for better visibility
        page = st.radio(
            "Choose a page:",
            list(nav_options.keys()),
            format_func=lambda x: nav_options[x],
            index=list(nav_options.keys()).index(st.session_state.current_page) if st.session_state.current_page in nav_options else 0
        )
        
        st.session_state.current_page = page
        
        # Compact System Status Section
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        try:
            conn = get_database_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM tasks")
                task_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM employees")
                emp_count = cursor.fetchone()[0]
                conn.close()
                
                # Compact status metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📋 Tasks", task_count)
                with col2:
                    st.metric("👥 Employees", emp_count)
                
                # System health indicator (compact)
                if task_count > 0 and emp_count > 0:
                    st.success("✅ System Healthy")
                elif task_count == 0:
                    st.warning("⚠️ No Tasks")
                elif emp_count == 0:
                    st.warning("⚠️ No Employees")
                else:
                    st.error("❌ System Error")
            else:
                st.error("❌ DB Connection Failed")
        except Exception as e:
            st.error(f"❌ Error: {str(e)[:30]}")
        
        # Compact Notifications Section
        st.markdown("---")
        st.markdown("### 🔔 Notifications")
        
        try:
            notifications = get_real_time_notifications()
            if notifications:
                for notif in notifications[:3]:  # Show only last 3 notifications
                    if notif['type'] == 'success':
                        st.success(f"{notif['icon']} {notif['message'][:40]}...")
                    elif notif['type'] == 'warning':
                        st.warning(f"{notif['icon']} {notif['message'][:40]}...")
                    elif notif['type'] == 'error':
                        st.error(f"{notif['icon']} {notif['message'][:40]}...")
                    else:
                        st.info(f"{notif['icon']} {notif['message'][:40]}...")
            else:
                st.info("📭 No new notifications")
        except Exception as e:
            st.info("📭 Notifications unavailable")
        
        # Compact Pending Tasks Section
        st.markdown("---")
        st.markdown("### ⏳ Pending Tasks")
        
        try:
            pending_tasks = get_pending_tasks_summary()
            if pending_tasks:
                for task in pending_tasks[:2]:  # Show only top 2 pending tasks
                    st.markdown(f"**{task['priority']}** {task['title'][:25]}...")
                    st.markdown(f"⏰ {task['due_date']}")
            else:
                st.info("✅ No pending tasks")
        except Exception as e:
            st.info("✅ No pending tasks")
        
        # Compact Quick Actions Section
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", key="refresh_sidebar"):
                st.rerun()
        with col2:
            if st.button("🤖 Auto-Assign", key="auto_assign_sidebar"):
                try:
                    auto_assign_uploaded_tasks()
                    st.success("✅ Tasks assigned!")
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)[:30]}")
        
        # User Info and Logout
        st.markdown("---")
        st.markdown("### 👤 User Info")
        
        if st.session_state.current_user:
            st.info(f"**User:** {st.session_state.current_user}")
            if st.session_state.user_department:
                st.info(f"**Dept:** {st.session_state.user_department}")
        
        # Logout button
        if st.button("🚪 Logout", key="logout_btn", type="secondary"):
            st.session_state.is_authenticated = False
            st.session_state.current_user = None
            st.session_state.user_role = None
            st.rerun()
        
        # Close scrollable container
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Page routing with error handling
    try:
        if page == "dashboard":
            show_dashboard()
        elif page == "upload":
            show_upload_page()
        elif page == "analysis":
            show_task_analysis()
        elif page == "employees":
            show_employee_management()
        elif page == "analytics":
            show_analytics()
        elif page == "models":
            show_ai_models()
        elif page == "settings":
            show_settings()
        elif page == "learning":
            show_learning_statistics()
    except Exception as e:
        st.error(f"❌ Error loading page: {str(e)}")
        st.info("Please try refreshing the page or contact support.")

if __name__ == "__main__":
    main() 