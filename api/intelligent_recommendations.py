#!/usr/bin/env python3
"""
Intelligent Employee Recommendation API
Provides AI-powered employee assignment recommendations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import sqlite3
import json
from datetime import datetime
import uvicorn

app = FastAPI(
    title="AI Task Management - Intelligent Recommendations API",
    description="API for intelligent employee-task assignment recommendations",
    version="1.0.0"
)

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
    expertise_areas: List[str]
    preferred_task_types: List[str]
    current_workload: int
    max_capacity: int
    experience_years: int
    confidence_score: float
    reasoning: str
    location: str
    availability: str

class AssignmentRequest(BaseModel):
    task_id: int
    employee_id: str
    assignment_reason: str
    confidence_score: float

class AssignmentResponse(BaseModel):
    assignment_id: int
    task_id: int
    employee_id: str
    employee_name: str
    assignment_reason: str
    confidence_score: float
    assigned_at: str
    status: str

def get_database_connection():
    """Get database connection"""
    try:
        conn = sqlite3.connect('ai_task_management.db', check_same_thread=False)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def get_ai_employee_recommendation(task_title: str, task_description: str, task_category: str, 
                                 task_priority: str, urgency_score: int, complexity_score: int,
                                 business_impact: int, estimated_hours: float):
    """Get AI recommendation for employee assignment"""
    try:
        conn = get_database_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        # Get employees
        employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
        conn.close()
        
        if employees_df.empty:
            raise HTTPException(status_code=404, detail="No employees available")
        
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
        raise HTTPException(status_code=500, detail=str(e))

def generate_employee_recommendation_reasoning(task_category: str, task_priority: str, 
                                            complexity_score: int, employee: Dict[str, Any]):
    """Generate reasoning for employee recommendation"""
    reasoning = []
    
    # Skills reasoning
    skills = employee['skills']
    reasoning.append(f"🎯 Skills Match: Employee has {len(skills)} relevant skills")
    
    # Experience reasoning
    experience = employee['experience_years']
    if experience >= 5:
        reasoning.append(f"📈 Senior Experience: {experience} years of experience")
    elif experience >= 3:
        reasoning.append(f"📊 Mid-level Experience: {experience} years of experience")
    else:
        reasoning.append(f"📚 Junior Experience: {experience} years of experience")
    
    # Workload reasoning
    workload = employee['current_workload']
    max_capacity = employee['max_capacity']
    utilization = (workload / max_capacity) * 100
    
    if utilization < 70:
        reasoning.append(f"⚖️ Available Capacity: {utilization:.1f}% workload utilization")
    elif utilization < 90:
        reasoning.append(f"⚠️ Moderate Workload: {utilization:.1f}% workload utilization")
    else:
        reasoning.append(f"🔥 High Workload: {utilization:.1f}% workload utilization")
    
    # Category reasoning
    preferred_types = employee['preferred_task_types']
    if task_category in preferred_types:
        reasoning.append(f"✅ Preferred Task Type: {task_category} is in preferred types")
    
    # Priority reasoning
    if task_priority in ['high', 'critical'] and experience >= 5:
        reasoning.append("🚨 High Priority Task: Assigned to experienced employee")
    
    return " | ".join(reasoning)

@app.post("/api/recommend-employee", response_model=EmployeeRecommendation)
async def recommend_employee(task: TaskRequest):
    """Get AI recommendation for employee assignment"""
    try:
        # Get recommendation
        recommended_employee = get_ai_employee_recommendation(
            task.title, task.description, task.category, task.priority,
            task.urgency_score, task.complexity_score, task.business_impact, task.estimated_hours
        )
        
        if recommended_employee:
            # Parse JSON fields
            skills = json.loads(recommended_employee['skills']) if recommended_employee['skills'] else []
            expertise_areas = json.loads(recommended_employee['expertise_areas']) if recommended_employee['expertise_areas'] else []
            preferred_task_types = json.loads(recommended_employee['preferred_task_types']) if recommended_employee['preferred_task_types'] else []
            
            reasoning = generate_employee_recommendation_reasoning(
                task.category, task.priority, task.complexity_score, recommended_employee
            )
            
            return EmployeeRecommendation(
                employee_id=recommended_employee['id'],
                employee_name=recommended_employee['name'],
                role=recommended_employee['role'],
                skills=skills,
                expertise_areas=expertise_areas,
                preferred_task_types=preferred_task_types,
                current_workload=recommended_employee['current_workload'],
                max_capacity=recommended_employee['max_capacity'],
                experience_years=recommended_employee['experience_years'],
                confidence_score=0.85,
                reasoning=reasoning,
                location=recommended_employee['location'],
                availability=recommended_employee['availability']
            )
        else:
            raise HTTPException(status_code=404, detail="No suitable employee found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/employees")
async def get_employees():
    """Get all employees"""
    try:
        conn = get_database_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
        conn.close()
        
        # Parse JSON fields
        employees = []
        for _, emp in employees_df.iterrows():
            employee_data = emp.to_dict()
            employee_data['skills'] = json.loads(emp['skills']) if emp['skills'] else []
            employee_data['expertise_areas'] = json.loads(emp['expertise_areas']) if emp['expertise_areas'] else []
            employee_data['preferred_task_types'] = json.loads(emp['preferred_task_types']) if emp['preferred_task_types'] else []
            employees.append(employee_data)
        
        return employees
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/assignments")
async def get_assignments():
    """Get all task assignments"""
    try:
        conn = get_database_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        assignments_df = pd.read_sql_query("""
            SELECT ta.*, t.title, t.category, t.priority, t.status
            FROM task_assignments ta
            JOIN tasks t ON ta.task_id = t.id
            ORDER BY ta.assigned_at DESC
        """, conn)
        conn.close()
        
        return assignments_df.to_dict('records')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/assign-task", response_model=AssignmentResponse)
async def assign_task(assignment: AssignmentRequest):
    """Assign a task to an employee"""
    try:
        conn = get_database_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = conn.cursor()
        
        # Get employee name
        cursor.execute("SELECT name FROM employees WHERE id = ?", (assignment.employee_id,))
        employee_result = cursor.fetchone()
        
        if not employee_result:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        employee_name = employee_result[0]
        
        # Create assignment record
        cursor.execute("""
            INSERT INTO task_assignments (
                task_id, employee_id, employee_name, assignment_reason,
                confidence_score, assigned_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            assignment.task_id,
            assignment.employee_id,
            employee_name,
            assignment.assignment_reason,
            assignment.confidence_score,
            datetime.now().isoformat(),
            'assigned'
        ))
        
        assignment_id = cursor.lastrowid
        
        # Update task assignment
        cursor.execute("""
            UPDATE tasks 
            SET assigned_to = ?, updated_at = ?
            WHERE id = ?
        """, (employee_name, datetime.now().isoformat(), assignment.task_id))
        
        conn.commit()
        conn.close()
        
        return AssignmentResponse(
            assignment_id=assignment_id,
            task_id=assignment.task_id,
            employee_id=assignment.employee_id,
            employee_name=employee_name,
            assignment_reason=assignment.assignment_reason,
            confidence_score=assignment.confidence_score,
            assigned_at=datetime.now().isoformat(),
            status='assigned'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/assignments")
async def get_assignment_analytics():
    """Get assignment analytics"""
    try:
        conn = get_database_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        # Get analytics data
        analytics_df = pd.read_sql_query("""
            SELECT 
                COUNT(*) as total_assignments,
                COUNT(DISTINCT employee_name) as unique_employees,
                AVG(confidence_score) as avg_confidence,
                COUNT(CASE WHEN t.status = 'completed' THEN 1 END) as completed_tasks,
                COUNT(CASE WHEN t.status = 'pending' THEN 1 END) as pending_tasks
            FROM task_assignments ta
            JOIN tasks t ON ta.task_id = t.id
        """, conn)
        
        # Get employee performance
        performance_df = pd.read_sql_query("""
            SELECT 
                ta.employee_name,
                COUNT(ta.id) as total_assignments,
                AVG(ta.confidence_score) as avg_confidence,
                COUNT(CASE WHEN t.status = 'completed' THEN 1 END) as completed_tasks
            FROM task_assignments ta
            JOIN tasks t ON ta.task_id = t.id
            GROUP BY ta.employee_name
            ORDER BY total_assignments DESC
        """, conn)
        
        conn.close()
        
        return {
            "overview": analytics_df.to_dict('records')[0] if not analytics_df.empty else {},
            "performance": performance_df.to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "AI Task Management - Intelligent Recommendations API",
        "version": "1.0.0",
        "endpoints": [
            "POST /api/recommend-employee",
            "GET /api/employees", 
            "GET /api/assignments",
            "POST /api/assign-task",
            "GET /api/analytics/assignments"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 