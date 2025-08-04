from fastapi import APIRouter, HTTPException, Depends, Request, Body
from pydantic import BaseModel
from typing import List
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models.employee import Employee, Base
# SessionLocal is defined below in this file
import os
try:
    import joblib
    ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/employee_match_model.pkl')
    ml_model = joblib.load(ML_MODEL_PATH)
except Exception:
    ml_model = None

router = APIRouter()

# --- Database setup ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Password Hashing and JWT ---
SECRET_KEY = "supersecretkey"  # In production, use env var
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

security = HTTPBearer()

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- User Auth ---
class SignupRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post('/signup')
def signup(user: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(UserDB).filter(UserDB.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    hashed = hash_password(user.password)
    db_user = UserDB(email=user.email, hashed_password=hashed)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"message": "Signup successful"}

@router.post('/login')
def login(user: LoginRequest, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.email})
    return {"token": token}

# --- Tasks (existing code below) ---
class Task(BaseModel):
    id: str
    title: str
    status: str
    assignee: str
    dueDate: str

tasks_db = [
    Task(id="1", title="Design login page", status="Pending", assignee="Alice", dueDate="2024-07-01"),
    Task(id="2", title="Build API endpoint", status="In Progress", assignee="Bob", dueDate="2024-07-03"),
    Task(id="3", title="Write docs", status="Completed", assignee="Charlie", dueDate="2024-07-05"),
    Task(id="4", title="Test user flows", status="Pending", assignee="Alice", dueDate="2024-07-02"),
]

@router.get('/tasks', response_model=List[Task])
def get_tasks():
    return tasks_db

@router.post('/tasks', response_model=Task)
def create_task(task: Task):
    tasks_db.append(task)
    return task

@router.put('/tasks/{task_id}', response_model=Task)
def update_task(task_id: str, task: Task):
    for i, t in enumerate(tasks_db):
        if t.id == task_id:
            tasks_db[i] = task
            return task
    raise HTTPException(status_code=404, detail="Task not found")

@router.delete('/tasks/{task_id}')
def delete_task(task_id: str):
    for i, t in enumerate(tasks_db):
        if t.id == task_id:
            tasks_db.pop(i)
            return {"ok": True}
    raise HTTPException(status_code=404, detail="Task not found")

# In-memory employees data for demo (should be replaced with DB in production)
employees_data = [
    {"employee_id": "E001", "name": "Samantha Russell", "skills": ["database", "backend", "testing", "cloud", "UX"], "current_tasks": 4},
    {"employee_id": "E002", "name": "Sandra Casey", "skills": ["security", "UX", "data analysis", "cloud"], "current_tasks": 3},
    {"employee_id": "E003", "name": "Raymond Mercado", "skills": ["testing", "database", "DevOps", "data analysis"], "current_tasks": 1},
    {"employee_id": "E004", "name": "Cynthia Ho", "skills": ["NLP", "backend", "DevOps", "API", "cloud"], "current_tasks": 1},
    {"employee_id": "E005", "name": "Diana Jackson", "skills": ["data analysis", "ML", "database", "DevOps", "UX"], "current_tasks": 1},
    {"employee_id": "E006", "name": "Amy English", "skills": ["NLP", "backend", "cloud", "UI"], "current_tasks": 3},
    {"employee_id": "E007", "name": "James Pennington", "skills": ["security", "testing", "API", "DevOps", "data analysis"], "current_tasks": 2},
    {"employee_id": "E008", "name": "Alexis Woods", "skills": ["security", "database", "testing"], "current_tasks": 2},
    {"employee_id": "E009", "name": "Jacqueline Franco", "skills": ["UX", "security", "DevOps"], "current_tasks": 4},
    {"employee_id": "E010", "name": "Leslie Ray", "skills": ["UX", "security", "testing"], "current_tasks": 3},
    {"employee_id": "E011", "name": "Kathleen Steele", "skills": ["bugfix", "UX", "testing", "frontend", "API"], "current_tasks": 3},
    {"employee_id": "E012", "name": "Michael Jones", "skills": ["security", "data analysis", "API"], "current_tasks": 5},
    {"employee_id": "E013", "name": "Catherine Ellis", "skills": ["DevOps", "database", "data analysis", "testing", "bugfix", "ML"], "current_tasks": 2},
    {"employee_id": "E014", "name": "Dawn Bell", "skills": ["security", "testing", "ML", "frontend"], "current_tasks": 0},
    {"employee_id": "E015", "name": "Sandra Clark", "skills": ["API", "security", "database", "UI", "testing"], "current_tasks": 3},
    {"employee_id": "E016", "name": "Sean Garza MD", "skills": ["database", "API", "testing", "data analysis", "security", "frontend"], "current_tasks": 5},
    {"employee_id": "E017", "name": "Harry Collins", "skills": ["API", "NLP", "ML", "frontend", "testing"], "current_tasks": 5},
    {"employee_id": "E018", "name": "Nicole Smith", "skills": ["frontend", "testing", "UX", "UI"], "current_tasks": 3},
    {"employee_id": "E019", "name": "Amy Wiggins", "skills": ["bugfix", "data analysis", "NLP", "backend"], "current_tasks": 0},
    {"employee_id": "E020", "name": "Jacob Hayden", "skills": ["security", "NLP", "cloud", "ML", "UI", "testing"], "current_tasks": 2},
    {"employee_id": "E021", "name": "Jill Martinez", "skills": ["NLP", "DevOps", "security"], "current_tasks": 3},
    {"employee_id": "E022", "name": "Mark Becker", "skills": ["frontend", "bugfix", "UI"], "current_tasks": 5},
    {"employee_id": "E023", "name": "Shawn Smith", "skills": ["NLP", "data analysis", "cloud"], "current_tasks": 4},
    {"employee_id": "E024", "name": "Michael Blackburn", "skills": ["DevOps", "database", "API", "security", "bugfix"], "current_tasks": 1},
    {"employee_id": "E025", "name": "Noah Robinson", "skills": ["database", "frontend", "ML", "security", "UI"], "current_tasks": 2}
]

def skill_match_count(employee_skills, required):
    return len([s for s in required if s in employee_skills])

@router.post('/ai-suggest-employees')
def ai_suggest_employees(payload = Body(...)):
    required_skills = payload.get('requiredSkills', [])
    if not required_skills:
        return []
    
    # Use in-memory employees data for now
    employees_data = [
        {"employee_id": "E001", "name": "Samantha Russell", "skills": ["database", "backend", "testing", "cloud", "UX"], "current_tasks": 4},
        {"employee_id": "E002", "name": "Sandra Casey", "skills": ["security", "UX", "data analysis", "cloud"], "current_tasks": 3},
        {"employee_id": "E003", "name": "Raymond Mercado", "skills": ["testing", "database", "DevOps", "data analysis"], "current_tasks": 1},
        {"employee_id": "E004", "name": "Cynthia Ho", "skills": ["NLP", "backend", "DevOps", "API", "cloud"], "current_tasks": 1},
        {"employee_id": "E005", "name": "Diana Jackson", "skills": ["data analysis", "ML", "database", "DevOps", "UX"], "current_tasks": 1},
        {"employee_id": "E006", "name": "Amy English", "skills": ["NLP", "backend", "cloud", "UI"], "current_tasks": 3},
        {"employee_id": "E007", "name": "James Pennington", "skills": ["security", "testing", "API", "DevOps", "data analysis"], "current_tasks": 2},
        {"employee_id": "E008", "name": "Alexis Woods", "skills": ["security", "database", "testing"], "current_tasks": 2},
        {"employee_id": "E009", "name": "Jacqueline Franco", "skills": ["UX", "security", "DevOps"], "current_tasks": 4},
        {"employee_id": "E010", "name": "Leslie Ray", "skills": ["UX", "security", "testing"], "current_tasks": 3},
        {"employee_id": "E011", "name": "Kathleen Steele", "skills": ["bugfix", "UX", "testing", "frontend", "API"], "current_tasks": 3},
        {"employee_id": "E012", "name": "Michael Jones", "skills": ["security", "data analysis", "API"], "current_tasks": 5},
        {"employee_id": "E013", "name": "Catherine Ellis", "skills": ["DevOps", "database", "data analysis", "testing", "bugfix", "ML"], "current_tasks": 2},
        {"employee_id": "E014", "name": "Dawn Bell", "skills": ["security", "testing", "ML", "frontend"], "current_tasks": 0},
        {"employee_id": "E015", "name": "Sandra Clark", "skills": ["API", "security", "database", "UI", "testing"], "current_tasks": 3},
        {"employee_id": "E016", "name": "Sean Garza MD", "skills": ["database", "API", "testing", "data analysis", "security", "frontend"], "current_tasks": 5},
        {"employee_id": "E017", "name": "Harry Collins", "skills": ["API", "NLP", "ML", "frontend", "testing"], "current_tasks": 5},
        {"employee_id": "E018", "name": "Nicole Smith", "skills": ["frontend", "testing", "UX", "UI"], "current_tasks": 3},
        {"employee_id": "E019", "name": "Amy Wiggins", "skills": ["bugfix", "data analysis", "NLP", "backend"], "current_tasks": 0},
        {"employee_id": "E020", "name": "Jacob Hayden", "skills": ["security", "NLP", "cloud", "ML", "UI", "testing"], "current_tasks": 2},
        {"employee_id": "E021", "name": "Jill Martinez", "skills": ["NLP", "DevOps", "security"], "current_tasks": 3},
        {"employee_id": "E022", "name": "Mark Becker", "skills": ["frontend", "bugfix", "UI"], "current_tasks": 5},
        {"employee_id": "E023", "name": "Shawn Smith", "skills": ["NLP", "data analysis", "cloud"], "current_tasks": 4},
        {"employee_id": "E024", "name": "Michael Blackburn", "skills": ["DevOps", "database", "API", "security", "bugfix"], "current_tasks": 1},
        {"employee_id": "E025", "name": "Noah Robinson", "skills": ["database", "frontend", "ML", "security", "UI"], "current_tasks": 2}
    ]
    
    # If ML model is available, use it for ranking
    if ml_model:
        # Example: create feature vectors for each employee-task pair
        X = []
        for e in employees_data:
            skill_match = sum(1 for s in required_skills if s in e['skills'])
            X.append([skill_match, e['current_tasks']])
        # Predict scores (higher is better)
        scores = ml_model.predict(X)
        ranked = sorted(zip(employees_data, scores), key=lambda x: x[1], reverse=True)
        return [e['employee_id'] for e, _ in ranked[:2]]
    
    # Fallback: perfect match + lowest workload, else top 2 by skill match
    perfect_matches = [e for e in employees_data if skill_match_count(e['skills'], required_skills) == len(required_skills)]
    if perfect_matches:
        min_tasks = min(e['current_tasks'] for e in perfect_matches)
        return [e['employee_id'] for e in perfect_matches if e['current_tasks'] == min_tasks]
    
    sorted_emps = sorted(employees_data, key=lambda e: skill_match_count(e['skills'], required_skills), reverse=True)
    return [e['employee_id'] for e in sorted_emps[:2]] 