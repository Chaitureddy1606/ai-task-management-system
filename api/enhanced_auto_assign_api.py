#!/usr/bin/env python3
"""
Enhanced Auto-Assignment API
- Integrates enhanced auto-assignment with perfect-performance AI models
- Provides detailed scoring breakdown and recommendations
- Real-time assignment with confidence scoring
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import requests
import logging
from datetime import datetime
from enhanced_auto_assigner import EnhancedAutoAssigner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Auto-Assignment API",
    description="Advanced auto-assignment with AI integration and detailed scoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TaskAssignmentRequest(BaseModel):
    title: str
    description: str
    category: Optional[str] = None
    priority: Optional[str] = None
    urgency_score: Optional[int] = None
    complexity_score: Optional[int] = None
    business_impact: Optional[int] = None
    estimated_hours: Optional[float] = None
    days_until_deadline: Optional[int] = None
    assigned_to: Optional[int] = None
    created_by: Optional[int] = None

class AssignmentResponse(BaseModel):
    assigned_employee: str
    confidence: float
    method: str
    details: Dict
    ai_prediction: Optional[Dict] = None
    assignment_timestamp: str
    recommendations: List[Dict] = []

# Initialize enhanced assigner
assigner = EnhancedAutoAssigner()

@app.get("/api/enhanced-assignment/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Enhanced Auto-Assignment API",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/enhanced-assignment/assign", response_model=AssignmentResponse)
async def assign_task(request: TaskAssignmentRequest):
    """Enhanced auto-assignment with detailed scoring"""
    try:
        # Convert request to dictionary
        task_data = request.dict()
        
        # Perform enhanced assignment
        result = assigner.auto_assign_with_fallback(task_data)
        
        # Get recommendations
        recommendations = assigner.get_assignment_recommendations(task_data, top_k=3)
        result['recommendations'] = recommendations
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced assignment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enhanced-assignment/assign-simple")
async def assign_task_simple(request: TaskAssignmentRequest):
    """Simple assignment (original function style)"""
    try:
        task_data = request.dict()
        
        # Get employee data
        employee_data = assigner.get_employee_data_from_db()
        
        if not employee_data:
            raise HTTPException(status_code=404, detail="No employee data available")
        
        # Use original function logic with enhancements
        best_match = None
        best_score = -1
        
        for emp in employee_data:
            # Calculate score using enhanced logic
            skill_score = assigner.calculate_skill_match_score(task_data, emp)
            capacity_score = assigner.calculate_capacity_score(emp)
            role_score = assigner.calculate_role_match_score(task_data, emp)
            
            # Original scoring logic with enhancements
            score = (
                (skill_score * 10) +  # Skill match (0-10)
                (capacity_score * 2) +  # Capacity bonus (0-2)
                (role_score * 5)  # Role match (0-5)
            )
            
            if score > best_score:
                best_score = score
                best_match = emp.get('username', '')
        
        return {
            "assigned_employee": best_match,
            "confidence": min(1.0, best_score / 17.0),  # Normalize to 0-1
            "method": "enhanced_simple",
            "details": {
                "employee_name": best_match,
                "score": best_score,
                "assignment_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in simple assignment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/enhanced-assignment/recommendations")
async def get_recommendations(
    title: str,
    description: str,
    category: Optional[str] = None,
    urgency_score: Optional[int] = None,
    top_k: int = 3
):
    """Get assignment recommendations for a task"""
    try:
        task_data = {
            "title": title,
            "description": description,
            "category": category,
            "urgency_score": urgency_score or 5
        }
        
        recommendations = assigner.get_assignment_recommendations(task_data, top_k=top_k)
        
        return {
            "task": task_data,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/enhanced-assignment/employees")
async def get_employees():
    """Get all employees with their skills and performance data"""
    try:
        employee_data = assigner.get_employee_data_from_db()
        
        # Format employee data for display
        formatted_employees = []
        for emp in employee_data:
            formatted_emp = {
                "username": emp.get('username', ''),
                "full_name": emp.get('full_name', ''),
                "role": emp.get('role', ''),
                "department": emp.get('department', ''),
                "skills": emp.get('skills', []),
                "capacity_hours": emp.get('capacity_hours', 40),
                "success_rate": emp.get('success_rate', 0.7),
                "total_tasks_assigned": emp.get('total_tasks_assigned', 0),
                "completed_tasks": emp.get('completed_tasks', 0),
                "average_completion_time_hours": emp.get('average_completion_time_hours', 0)
            }
            formatted_employees.append(formatted_emp)
        
        return {
            "employees": formatted_employees,
            "total_employees": len(formatted_employees),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting employees: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enhanced-assignment/compare-methods")
async def compare_assignment_methods(request: TaskAssignmentRequest):
    """Compare different assignment methods"""
    try:
        task_data = request.dict()
        
        results = {}
        
        # Method 1: Enhanced hybrid (AI + rules)
        try:
            enhanced_result = assigner.auto_assign_with_fallback(task_data)
            results['enhanced_hybrid'] = enhanced_result
        except Exception as e:
            results['enhanced_hybrid'] = {"error": str(e)}
        
        # Method 2: AI only
        try:
            ai_result = assigner.get_ai_prediction(task_data)
            results['ai_only'] = ai_result
        except Exception as e:
            results['ai_only'] = {"error": str(e)}
        
        # Method 3: Simple rules-based
        try:
            employee_data = assigner.get_employee_data_from_db()
            if employee_data:
                best_match = None
                best_score = -1
                
                for emp in employee_data:
                    skill_score = assigner.calculate_skill_match_score(task_data, emp)
                    capacity_score = assigner.calculate_capacity_score(emp)
                    role_score = assigner.calculate_role_match_score(task_data, emp)
                    
                    score = skill_score + capacity_score + role_score
                    
                    if score > best_score:
                        best_score = score
                        best_match = emp.get('username', '')
                
                results['simple_rules'] = {
                    "assigned_employee": best_match,
                    "confidence": min(1.0, best_score / 3.0),
                    "method": "simple_rules",
                    "score": best_score
                }
            else:
                results['simple_rules'] = {"error": "No employee data available"}
        except Exception as e:
            results['simple_rules'] = {"error": str(e)}
        
        return {
            "task": task_data,
            "comparison_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error comparing methods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/enhanced-assignment/scoring-breakdown")
async def get_scoring_breakdown(
    title: str,
    description: str,
    category: Optional[str] = None,
    urgency_score: Optional[int] = None,
    complexity_score: Optional[int] = None
):
    """Get detailed scoring breakdown for all employees"""
    try:
        task_data = {
            "title": title,
            "description": description,
            "category": category,
            "urgency_score": urgency_score or 5,
            "complexity_score": complexity_score or 5
        }
        
        employee_data = assigner.get_employee_data_from_db()
        
        if not employee_data:
            raise HTTPException(status_code=404, detail="No employee data available")
        
        scoring_breakdown = []
        
        for emp in employee_data:
            skill_score = assigner.calculate_skill_match_score(task_data, emp)
            capacity_score = assigner.calculate_capacity_score(emp)
            role_score = assigner.calculate_role_match_score(task_data, emp)
            performance_score = assigner.calculate_performance_score(emp)
            urgency_score = assigner.calculate_urgency_compatibility(task_data, emp)
            
            total_score = (
                skill_score * 0.35 +
                capacity_score * 0.20 +
                role_score * 0.25 +
                performance_score * 0.15 +
                urgency_score * 0.05
            )
            
            breakdown = {
                "employee_name": emp.get('full_name', emp.get('username', 'Unknown')),
                "username": emp.get('username', ''),
                "role": emp.get('role', ''),
                "total_score": total_score,
                "scoring_breakdown": {
                    "skill_score": skill_score,
                    "capacity_score": capacity_score,
                    "role_score": role_score,
                    "performance_score": performance_score,
                    "urgency_score": urgency_score
                },
                "employee_details": {
                    "skills": emp.get('skills', []),
                    "capacity_hours": emp.get('capacity_hours', 40),
                    "success_rate": emp.get('success_rate', 0.7),
                    "total_tasks": emp.get('total_tasks_assigned', 0),
                    "completed_tasks": emp.get('completed_tasks', 0)
                }
            }
            
            scoring_breakdown.append(breakdown)
        
        # Sort by total score
        scoring_breakdown.sort(key=lambda x: x['total_score'], reverse=True)
        
        return {
            "task": task_data,
            "scoring_breakdown": scoring_breakdown,
            "total_employees": len(scoring_breakdown),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting scoring breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced Auto-Assignment API")
    print("=" * 50)
    print("✅ Enhanced auto-assignment with AI integration")
    print("✅ Detailed scoring breakdown")
    print("✅ Multiple assignment methods")
    print("✅ Real-time recommendations")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=5002) 