"""
Utility functions for the AI Task Management System
"""

import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_employee_profiles(file_path: str = "data/employee_profiles.json") -> List[Dict]:
    """Load employee profiles from JSON file"""
    try:
        with open(file_path, 'r') as f:
            profiles = json.load(f)
        logger.info(f"Loaded {len(profiles)} employee profiles")
        return profiles
    except FileNotFoundError:
        logger.error(f"Employee profiles file not found: {file_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        return []


def connect_db(db_path: str = "db/tasks.db") -> sqlite3.Connection:
    """Create connection to SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise


def create_tasks_table(conn: sqlite3.Connection) -> None:
    """Create tasks table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        category TEXT,
        priority_score REAL,
        estimated_hours REAL,
        deadline DATE,
        assigned_to TEXT,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        tags TEXT,
        complexity_score REAL,
        urgency_score REAL
    );
    """
    try:
        conn.execute(create_table_sql)
        conn.commit()
        logger.info("Tasks table created successfully")
    except sqlite3.Error as e:
        logger.error(f"Error creating tasks table: {e}")
        raise


def calculate_days_until_deadline(deadline: str) -> int:
    """Calculate days between now and deadline"""
    try:
        deadline_date = datetime.strptime(deadline, '%Y-%m-%d')
        today = datetime.now()
        delta = deadline_date - today
        return delta.days
    except ValueError:
        logger.warning(f"Invalid deadline format: {deadline}")
        return 999  # Default to far future if date is invalid


def normalize_score(value: float, min_val: float = 0, max_val: float = 10) -> float:
    """Normalize a score to 0-1 range"""
    if max_val == min_val:
        return 0.5
    return max(0, min(1, (value - min_val) / (max_val - min_val)))


def get_employee_by_id(employee_id: str, profiles: List[Dict]) -> Optional[Dict]:
    """Find employee by ID"""
    for profile in profiles:
        if profile.get('employee_id') == employee_id:
            return profile
    return None


def calculate_workload_percentage(current_workload: int, max_capacity: int) -> float:
    """Calculate workload as percentage"""
    if max_capacity == 0:
        return 1.0
    return min(1.0, current_workload / max_capacity)


def extract_keywords_from_text(text: str) -> List[str]:
    """Extract keywords from task description using simple NLP"""
    import re
    from collections import Counter
    
    # Simple keyword extraction (can be enhanced with NLP libraries)
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Remove common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Return most common keywords
    return [word for word, count in Counter(keywords).most_common(5)]


def save_model_metrics(model_name: str, metrics: Dict[str, float], file_path: str = "models/model_metrics.json") -> None:
    """Save model performance metrics"""
    try:
        # Load existing metrics if file exists
        try:
            with open(file_path, 'r') as f:
                all_metrics = json.load(f)
        except FileNotFoundError:
            all_metrics = {}
        
        # Add timestamp to metrics
        metrics['timestamp'] = datetime.now().isoformat()
        all_metrics[model_name] = metrics
        
        # Save updated metrics
        with open(file_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        logger.info(f"Saved metrics for {model_name}")
    except Exception as e:
        logger.error(f"Error saving model metrics: {e}")


def format_task_for_display(task: Dict) -> str:
    """Format task information for display"""
    return f"""
Task: {task.get('title', 'N/A')}
Priority: {task.get('priority_score', 0):.2f}
Estimated Hours: {task.get('estimated_hours', 0)}
Deadline: {task.get('deadline', 'N/A')}
Assigned To: {task.get('assigned_to', 'Unassigned')}
Status: {task.get('status', 'pending')}
    """.strip() 