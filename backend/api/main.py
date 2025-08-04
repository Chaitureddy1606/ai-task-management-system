from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
import sqlite3
import datetime
import random
import os
import json
from fastapi import Request
import bcrypt
import secrets
import time
RESET_TOKENS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../db/reset_tokens.json'))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ai_task_management.db"))

PROFILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../db/admin_profile.json'))
NOTIF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../db/notification_settings.json'))
API_KEYS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../db/api_keys.json'))

class Task(BaseModel):
    id: Optional[int]
    title: str
    description: Optional[str] = ""
    category: Optional[str] = ""
    priority_score: Optional[float] = 0.0
    estimated_hours: Optional[float] = 0.0
    deadline: Optional[str] = None
    assigned_to: Optional[str] = ""
    status: Optional[str] = "pending"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: Optional[str] = ""
    complexity_score: Optional[float] = 0.0
    urgency_score: Optional[float] = 0.0
    subtasks: Optional[Any] = None

class Notification(BaseModel):
    id: int
    type: str  # 'reminder', 'urgency', 'nudge'
    message: str
    task_id: Optional[int] = None
    created_at: str
    read: bool = False

class Employee(BaseModel):
    id: Optional[int]
    name: str
    email: str
    phone: Optional[str] = ""
    skills: Optional[str] = ""
    status: Optional[str] = "Active"

quiet_mode = False  # In-memory for now

# --- TASKS ---
@app.get("/api/tasks", response_model=List[Task])
def get_tasks():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    result = []
    for row in rows:
        task = dict(row)
        if task.get("subtasks"):
            try:
                task["subtasks"] = json.loads(task["subtasks"])
            except Exception:
                task["subtasks"] = []
        else:
            task["subtasks"] = []
        result.append(task)
    return result

@app.post("/api/tasks", response_model=Task)
def add_task(task: Task):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = datetime.datetime.now().isoformat()
    subtasks_json = json.dumps(task.subtasks) if task.subtasks else None
    cur.execute(
        """
        INSERT INTO tasks (title, description, category, priority_score, estimated_hours, deadline, assigned_to, status, created_at, updated_at, tags, complexity_score, urgency_score, subtasks)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task.title,
            task.description,
            task.category,
            task.priority_score,
            task.estimated_hours,
            task.deadline,
            task.assigned_to,
            task.status,
            now,
            now,
            task.tags,
            task.complexity_score,
            task.urgency_score,
            subtasks_json,
        ),
    )
    task_id = cur.lastrowid
    conn.commit()
    conn.close()
    task.id = task_id
    task.created_at = now
    task.updated_at = now
    return task

# --- ACTIVITY ---
@app.get("/api/activity")
def get_activity():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id, title, status, updated_at FROM tasks ORDER BY updated_at DESC LIMIT 10")
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# --- EVENTS/MEETINGS ---
@app.get("/api/events")
def get_events():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    today = datetime.date.today().isoformat()
    cur.execute("SELECT id, title, deadline, category FROM tasks WHERE deadline IS NOT NULL AND deadline >= ? ORDER BY deadline ASC LIMIT 10", (today,))
    rows = cur.fetchall()
    conn.close()
    # Add type and icon for demo
    events = []
    for row in rows:
        event_type = "meeting" if "meeting" in (row["category"] or "").lower() else "deadline"
        events.append({
            **dict(row),
            "type": event_type,
            "icon": "video" if event_type == "meeting" else "briefcase"
        })
    return events

# --- AI SUGGESTIONS ---
@app.get("/api/suggestions")
def get_suggestions():
    suggestions = [
        "Focus on your most important task first.",
        "Batch similar tasks to save time.",
        "Take a short break every hour to stay fresh.",
        "Review your goals for the week.",
        "Try the Pomodoro technique for deep work.",
        "Check for overdue tasks and reschedule if needed.",
        "Celebrate small wins to stay motivated!"
    ]
    return {"suggestion": random.choice(suggestions)}

# --- ANALYTICS SUMMARY ---
@app.get("/api/analytics/summary")
def get_analytics_summary(
    date_from: str = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: str = Query(None, description="End date (YYYY-MM-DD)"),
    employee: str = Query(None, description="Employee name")
):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    now = datetime.datetime.now()
    one_week_ago = (now - datetime.timedelta(days=7)).isoformat()
    one_month_ago = (now - datetime.timedelta(days=30)).isoformat()

    # Build WHERE clauses for filters
    filters = []
    params = []
    if date_from:
        filters.append("created_at >= ?")
        params.append(date_from)
    if date_to:
        filters.append("created_at <= ?")
        params.append(date_to)
    if employee:
        filters.append("assigned_to = ?")
        params.append(employee)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    # Weekly stats (filtered)
    cur.execute(f"SELECT COUNT(*) as count FROM tasks {where} AND created_at >= ?" if where else "SELECT COUNT(*) as count FROM tasks WHERE created_at >= ?", (*params, one_week_ago))
    tasks_created_week = cur.fetchone()["count"]
    cur.execute(f"SELECT COUNT(*) as count FROM tasks {where} AND status = 'completed' AND updated_at >= ?" if where else "SELECT COUNT(*) as count FROM tasks WHERE status = 'completed' AND updated_at >= ?", (*params, one_week_ago))
    tasks_completed_week = cur.fetchone()["count"]
    cur.execute(f"SELECT AVG(julianday(updated_at) - julianday(created_at)) as avg_days FROM tasks {where} AND status = 'completed' AND updated_at >= ?" if where else "SELECT AVG(julianday(updated_at) - julianday(created_at)) as avg_days FROM tasks WHERE status = 'completed' AND updated_at >= ?", (*params, one_week_ago))
    avg_completion_week = cur.fetchone()["avg_days"] or 0

    # Monthly stats (filtered)
    cur.execute(f"SELECT COUNT(*) as count FROM tasks {where} AND created_at >= ?" if where else "SELECT COUNT(*) as count FROM tasks WHERE created_at >= ?", (*params, one_month_ago))
    tasks_created_month = cur.fetchone()["count"]
    cur.execute(f"SELECT COUNT(*) as count FROM tasks {where} AND status = 'completed' AND updated_at >= ?" if where else "SELECT COUNT(*) as count FROM tasks WHERE status = 'completed' AND updated_at >= ?", (*params, one_month_ago))
    tasks_completed_month = cur.fetchone()["count"]
    cur.execute(f"SELECT AVG(julianday(updated_at) - julianday(created_at)) as avg_days FROM tasks {where} AND status = 'completed' AND updated_at >= ?" if where else "SELECT AVG(julianday(updated_at) - julianday(created_at)) as avg_days FROM tasks WHERE status = 'completed' AND updated_at >= ?", (*params, one_month_ago))
    avg_completion_month = cur.fetchone()["avg_days"] or 0

    # Time spent per task (filtered)
    cur.execute(f"SELECT id, title, assigned_to, created_at, updated_at, (julianday(updated_at) - julianday(created_at)) as days_spent FROM tasks {where} AND status = 'completed'" if where else "SELECT id, title, assigned_to, created_at, updated_at, (julianday(updated_at) - julianday(created_at)) as days_spent FROM tasks WHERE status = 'completed'", (*params,))
    time_per_task = [dict(row) for row in cur.fetchall()]

    # AI Recommendations (reuse suggestions logic)
    suggestions = [
        "Focus on your most important task first.",
        "Batch similar tasks to save time.",
        "Take a short break every hour to stay fresh.",
        "Review your goals for the week.",
        "Try the Pomodoro technique for deep work.",
        "Check for overdue tasks and reschedule if needed.",
        "Celebrate small wins to stay motivated!"
    ]
    ai_recommendation = random.choice(suggestions)

    conn.close()
    return {
        "weekly": {
            "tasks_created": tasks_created_week,
            "tasks_completed": tasks_completed_week,
            "avg_completion_days": round(avg_completion_week, 2) if avg_completion_week else None
        },
        "monthly": {
            "tasks_created": tasks_created_month,
            "tasks_completed": tasks_completed_month,
            "avg_completion_days": round(avg_completion_month, 2) if avg_completion_month else None
        },
        "time_per_task": time_per_task,
        "ai_recommendation": ai_recommendation
    }

@app.get("/api/notifications", response_model=List[Notification])
def get_notifications():
    notifications = []
    now = datetime.datetime.now()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks")
    rows = cur.fetchall()
    conn.close()
    nid = 1
    for row in rows:
        task = dict(row)
        # Reminder: deadline within 24h and not completed
        if task.get("deadline") and task.get("status", "").lower() != "completed":
            try:
                deadline = datetime.datetime.fromisoformat(task["deadline"])
                hours_left = (deadline - now).total_seconds() / 3600
                if 0 < hours_left <= 24:
                    notifications.append(Notification(
                        id=nid,
                        type="reminder",
                        message=f"Task '{task['title']}' is due in {int(hours_left)}h.",
                        task_id=task["id"],
                        created_at=now.isoformat(),
                        read=False
                    ))
                    nid += 1
                # Nudge: deadline within 3h
                if 0 < hours_left <= 3:
                    notifications.append(Notification(
                        id=nid,
                        type="nudge",
                        message=f"Task '{task['title']}' is almost overdue!",
                        task_id=task["id"],
                        created_at=now.isoformat(),
                        read=False
                    ))
                    nid += 1
                # Overdue: negative hours left
                if hours_left < 0 and task.get("status", "").lower() != "completed":
                    notifications.append(Notification(
                        id=nid,
                        type="urgency",
                        message=f"Task '{task['title']}' is overdue!",
                        task_id=task["id"],
                        created_at=now.isoformat(),
                        read=False
                    ))
                    nid += 1
            except Exception:
                pass
        # AI-based urgency: high priority or high urgency_score
        if (task.get("priority_score", 0) >= 8 or task.get("urgency_score", 0) >= 8) and task.get("status", "").lower() != "completed":
            notifications.append(Notification(
                id=nid,
                type="urgency",
                message=f"Task '{task['title']}' is marked as urgent!",
                task_id=task["id"],
                created_at=now.isoformat(),
                read=False
            ))
            nid += 1
    return notifications

@app.get("/api/quiet_mode")
def get_quiet_mode():
    return {"quiet_mode": quiet_mode}

@app.post("/api/quiet_mode")
def set_quiet_mode(value: dict):
    global quiet_mode
    quiet_mode = bool(value.get("quiet_mode", False))
    return {"quiet_mode": quiet_mode}

# --- EMAIL NOTIFICATION (stub) ---
def send_email(to: str, subject: str, body: str):
    # TODO: Integrate with SMTP or SendGrid
    print(f"[EMAIL] To: {to} | Subject: {subject} | Body: {body}")
    return True

@app.post("/api/send_email")
def api_send_email(data: dict):
    to = data.get("to")
    subject = data.get("subject", "AI Task Manager Notification")
    body = data.get("body", "")
    if not to or not body:
        raise HTTPException(status_code=400, detail="Missing recipient or body")
    send_email(to, subject, body)
    return {"sent": True}

# --- PUSH NOTIFICATION REGISTRATION (stub) ---
push_subscriptions = []

@app.post("/api/register_push")
def register_push_subscription(data: dict):
    # In production, associate with user
    push_subscriptions.append(data)
    print(f"[PUSH] Registered: {data}")
    return {"registered": True}

@app.post("/api/auto_assign")
def auto_assign(task: dict):
    # Load employees (for demo, from JSON file)
    employees_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/employee_profiles.json"))
    try:
        with open(employees_path, "r") as f:
            employees_data = json.load(f)
        employees = employees_data["employees"]
    except Exception:
        return {"employee": None, "reason": "No employee data found"}
    # Simple skill match: count overlapping skills
    task_skills = set((task.get("skills") or []) + (task.get("description") or "").lower().split())
    best_employee = None
    best_score = -1
    for emp in employees:
        emp_skills = set([s.lower() for s in emp.get("skills", [])])
        score = len(task_skills & emp_skills)
        if score > best_score:
            best_employee = emp
            best_score = score
    if best_employee:
        return {"employee": best_employee["name"], "score": best_score}
    return {"employee": None, "reason": "No suitable match found"}

@app.delete("/api/employees/{employee_name}")
def delete_employee(employee_name: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Unassign all tasks for this employee
    cur.execute("UPDATE tasks SET assigned_to = '' WHERE assigned_to = ?", (employee_name,))
    # Optionally, delete from employees table if you have one
    # cur.execute("DELETE FROM employees WHERE name = ?", (employee_name,))
    conn.commit()
    conn.close()
    return {"success": True, "unassigned_tasks": employee_name}

# --- Admin Profile ---
@app.get('/api/profile')
def get_profile():
    if not os.path.exists(PROFILE_PATH):
        return {
            "name": "Admin User",
            "email": "admin@company.com",
            "avatar": "",
            "phone": "",
            "department": "",
            "position": "",
            "location": "",
            "timezone": "UTC",
            "language": "English",
            "bio": "",
            "skills": []
        }
    with open(PROFILE_PATH, 'r') as f:
        return json.load(f)

@app.post('/api/profile')
def update_profile(profile: dict = Body(...)):
    with open(PROFILE_PATH, 'w') as f:
        json.dump(profile, f, indent=2)
    return {"success": True}

# --- Notification Settings ---
@app.get('/api/notification-settings')
def get_notification_settings():
    if not os.path.exists(NOTIF_PATH):
        return {
            "email": True,
            "push": True,
            "sms": False,
            "taskUpdates": True,
            "deadlineReminders": True,
            "skillMatches": True,
            "systemUpdates": False,
            "weeklyReports": True,
            "monthlyReports": False
        }
    with open(NOTIF_PATH, 'r') as f:
        return json.load(f)

@app.post('/api/notification-settings')
def update_notification_settings(settings: dict = Body(...)):
    with open(NOTIF_PATH, 'w') as f:
        json.dump(settings, f, indent=2)
    return {"success": True}

# --- API Keys ---
@app.get('/api/api-keys')
def get_api_keys():
    if not os.path.exists(API_KEYS_PATH):
        return []
    with open(API_KEYS_PATH, 'r') as f:
        return json.load(f)

@app.post('/api/api-keys')
def create_api_key(key: dict = Body(...)):
    keys = []
    if os.path.exists(API_KEYS_PATH):
        with open(API_KEYS_PATH, 'r') as f:
            keys = json.load(f)
    keys.insert(0, key)
    with open(API_KEYS_PATH, 'w') as f:
        json.dump(keys, f, indent=2)
    return {"success": True}

@app.delete('/api/api-keys/{key_id}')
def delete_api_key(key_id: str):
    if not os.path.exists(API_KEYS_PATH):
        return {"success": False}
    with open(API_KEYS_PATH, 'r') as f:
        keys = json.load(f)
    keys = [k for k in keys if k.get('id') != key_id]
    with open(API_KEYS_PATH, 'w') as f:
        json.dump(keys, f, indent=2)
    return {"success": True}

@app.post('/api/auth/verify')
def verify_credentials(data: dict = Body(...)):
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        raise HTTPException(status_code=400, detail='Missing email or password')
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../db/users.db'))
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('SELECT id, email, name, password_hash FROM users WHERE email = ?', (email,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail='Invalid credentials')
    user_id, user_email, user_name, password_hash = row
    if not bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
        raise HTTPException(status_code=401, detail='Invalid credentials')
    return {"id": user_id, "email": user_email, "name": user_name}

@app.post('/api/auth/signup')
def signup(data: dict = Body(...)):
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    if not name or not email or not password:
        raise HTTPException(status_code=400, detail='Missing name, email, or password')
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../db/users.db'))
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, name TEXT)')
    cur.execute('SELECT id FROM users WHERE email = ?', (email,))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=409, detail='Email already registered')
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    cur.execute('INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)', (email, password_hash, name))
    conn.commit()
    conn.close()
    return {"success": True, "message": "User registered successfully"}

@app.post('/api/auth/request-reset')
def request_password_reset(data: dict = Body(...)):
    email = data.get('email')
    if not email:
        raise HTTPException(status_code=400, detail='Missing email')
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../db/users.db'))
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('SELECT id FROM users WHERE email = ?', (email,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail='Email not found')
    # Generate token
    token = secrets.token_urlsafe(32)
    expires = int(time.time()) + 3600  # 1 hour
    # Store token
    tokens = {}
    if os.path.exists(RESET_TOKENS_PATH):
        with open(RESET_TOKENS_PATH, 'r') as f:
            tokens = json.load(f)
    tokens[token] = {"email": email, "expires": expires}
    with open(RESET_TOKENS_PATH, 'w') as f:
        json.dump(tokens, f, indent=2)
    # For demo, print the reset link (in production, send email)
    print(f"Password reset link: http://localhost:3000/reset-password?token={token}")
    return {"success": True, "message": "If the email exists, a reset link has been sent."}

@app.post('/api/auth/reset-password')
def reset_password(data: dict = Body(...)):
    token = data.get('token')
    new_password = data.get('new_password')
    if not token or not new_password:
        raise HTTPException(status_code=400, detail='Missing token or new password')
    # Load tokens
    if not os.path.exists(RESET_TOKENS_PATH):
        raise HTTPException(status_code=400, detail='Invalid or expired token')
    with open(RESET_TOKENS_PATH, 'r') as f:
        tokens = json.load(f)
    token_data = tokens.get(token)
    if not token_data or token_data['expires'] < int(time.time()):
        raise HTTPException(status_code=400, detail='Invalid or expired token')
    email = token_data['email']
    # Update password
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../db/users.db'))
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    cur.execute('UPDATE users SET password_hash = ? WHERE email = ?', (password_hash, email))
    conn.commit()
    conn.close()
    # Delete token
    del tokens[token]
    with open(RESET_TOKENS_PATH, 'w') as f:
        json.dump(tokens, f, indent=2)
    return {"success": True, "message": "Password has been reset."}

@app.get("/health")
def health_check():
    return {"status": "ok"} 