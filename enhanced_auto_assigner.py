#!/usr/bin/env python3
"""
Enhanced Auto-Assignment System
- Integrates with perfect-performance AI models
- Uses database employee data and historical performance
- Advanced scoring algorithm with multiple factors
- Real-time assignment with confidence scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import requests
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAutoAssigner:
    """Enhanced auto-assignment system with AI integration"""
    
    def __init__(self, api_base_url: str = "http://localhost:5001"):
        self.api_base_url = api_base_url
        self.category_role_mapping = {
            'bug': ['developer', 'senior_developer', 'qa_engineer'],
            'feature': ['developer', 'senior_developer', 'architect'],
            'testing': ['qa_engineer', 'developer', 'senior_developer'],
            'documentation': ['technical_writer', 'developer', 'senior_developer'],
            'optimization': ['senior_developer', 'architect', 'developer'],
            'security': ['security_engineer', 'senior_developer', 'architect'],
            'deployment': ['devops_engineer', 'senior_developer', 'developer'],
            'research': ['architect', 'senior_developer', 'developer']
        }
        
        # Skill importance weights
        self.skill_weights = {
            'python': 1.0,
            'javascript': 0.9,
            'java': 0.9,
            'react': 0.8,
            'api': 0.8,
            'testing': 0.7,
            'automation': 0.7,
            'qa': 0.7,
            'management': 0.6,
            'planning': 0.6,
            'coordination': 0.6,
            'leadership': 0.6,
            'architecture': 0.9,
            'devops': 0.8,
            'security': 0.9,
            'database': 0.8,
            'frontend': 0.7,
            'backend': 0.8
        }
    
    def get_employee_data_from_db(self) -> List[Dict]:
        """Get employee data from database"""
        try:
            response = requests.get(f"{self.api_base_url}/api/employees")
            if response.status_code == 200:
                data = response.json()
                return data.get('employees', [])
            else:
                logger.error(f"Failed to get employee data: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting employee data: {e}")
            return []
    
    def get_ai_prediction(self, task: Dict) -> Dict:
        """Get AI prediction for task assignment"""
        try:
            response = requests.post(f"{self.api_base_url}/api/assign-employee", json=task)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get AI prediction: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error getting AI prediction: {e}")
            return {}
    
    def calculate_skill_match_score(self, task: Dict, employee: Dict) -> float:
        """Calculate skill match score between task and employee"""
        task_skills = set()
        employee_skills = set(employee.get('skills', []))
        
        # Extract skills from task title and description
        task_text = f"{task.get('title', '')} {task.get('description', '')}".lower()
        
        # Map task text to required skills
        skill_keywords = {
            'python': ['python', 'django', 'flask', 'pandas', 'numpy'],
            'javascript': ['javascript', 'js', 'react', 'vue', 'angular', 'node'],
            'java': ['java', 'spring', 'maven', 'gradle'],
            'api': ['api', 'rest', 'graphql', 'endpoint'],
            'testing': ['test', 'testing', 'qa', 'automation', 'selenium'],
            'database': ['database', 'sql', 'postgres', 'mysql', 'mongodb'],
            'security': ['security', 'authentication', 'authorization', 'encryption'],
            'devops': ['devops', 'docker', 'kubernetes', 'ci/cd', 'deployment'],
            'frontend': ['frontend', 'ui', 'ux', 'html', 'css'],
            'backend': ['backend', 'server', 'api', 'database']
        }
        
        for skill, keywords in skill_keywords.items():
            if any(keyword in task_text for keyword in keywords):
                task_skills.add(skill)
        
        # Calculate weighted skill match
        if not task_skills:
            return 0.5  # Default score if no skills detected
        
        matched_skills = task_skills.intersection(employee_skills)
        total_weight = sum(self.skill_weights.get(skill, 0.5) for skill in task_skills)
        matched_weight = sum(self.skill_weights.get(skill, 0.5) for skill in matched_skills)
        
        return matched_weight / total_weight if total_weight > 0 else 0.0
    
    def calculate_capacity_score(self, employee: Dict) -> float:
        """Calculate capacity score (lower capacity = higher score)"""
        capacity = employee.get('capacity_hours', 40)
        max_capacity = 40
        
        # Normalize capacity score (lower capacity = higher score)
        capacity_score = (max_capacity - capacity) / max_capacity
        return max(0.1, capacity_score)  # Minimum 0.1 score
    
    def calculate_role_match_score(self, task: Dict, employee: Dict) -> float:
        """Calculate role match score"""
        task_category = task.get('category', 'feature')
        employee_role = employee.get('role', 'developer')
        
        # Get preferred roles for task category
        preferred_roles = self.category_role_mapping.get(task_category, ['developer'])
        
        if employee_role in preferred_roles:
            # Higher score for exact role match
            role_index = preferred_roles.index(employee_role)
            return 1.0 - (role_index * 0.2)  # 1.0 for first choice, 0.8 for second, etc.
        else:
            return 0.3  # Lower score for non-preferred roles
    
    def calculate_performance_score(self, employee: Dict) -> float:
        """Calculate performance score based on historical data"""
        success_rate = employee.get('success_rate', 0.7)
        completed_tasks = employee.get('completed_tasks', 0)
        total_tasks = employee.get('total_tasks_assigned', 1)
        
        # Base score from success rate
        performance_score = success_rate
        
        # Bonus for experience (more completed tasks)
        if total_tasks > 0:
            experience_bonus = min(0.2, completed_tasks / total_tasks * 0.2)
            performance_score += experience_bonus
        
        return min(1.0, performance_score)
    
    def calculate_urgency_compatibility(self, task: Dict, employee: Dict) -> float:
        """Calculate compatibility based on task urgency and employee availability"""
        task_urgency = task.get('urgency_score', 5)
        employee_capacity = employee.get('capacity_hours', 40)
        
        # High urgency tasks should go to employees with more capacity
        if task_urgency >= 8:  # High urgency
            if employee_capacity >= 30:
                return 1.0
            elif employee_capacity >= 20:
                return 0.7
            else:
                return 0.4
        elif task_urgency >= 5:  # Medium urgency
            return 0.8
        else:  # Low urgency
            return 0.6
    
    def enhanced_auto_assign(self, task: Dict, employee_data: List[Dict]) -> Tuple[str, float, Dict]:
        """Enhanced auto-assignment with multiple scoring factors"""
        best_match = None
        best_score = -1
        best_details = {}
        
        # Get AI prediction as reference
        ai_prediction = self.get_ai_prediction(task)
        ai_suggested_employee = ai_prediction.get('assigned_employee', '')
        ai_confidence = ai_prediction.get('confidence', 0.5)
        
        for emp in employee_data:
            # Calculate multiple scoring factors
            skill_score = self.calculate_skill_match_score(task, emp)
            capacity_score = self.calculate_capacity_score(emp)
            role_score = self.calculate_role_match_score(task, emp)
            performance_score = self.calculate_performance_score(emp)
            urgency_score = self.calculate_urgency_compatibility(task, emp)
            
            # AI confidence bonus
            ai_bonus = 0.2 if emp.get('username', '') == ai_suggested_employee else 0.0
            
            # Weighted combination of all factors
            total_score = (
                skill_score * 0.35 +           # 35% weight for skills
                capacity_score * 0.20 +        # 20% weight for capacity
                role_score * 0.25 +            # 25% weight for role match
                performance_score * 0.15 +     # 15% weight for performance
                urgency_score * 0.05 +         # 5% weight for urgency
                ai_bonus                        # AI confidence bonus
            )
            
            # Store detailed scoring for transparency
            emp_details = {
                'employee_name': emp.get('full_name', emp.get('username', 'Unknown')),
                'username': emp.get('username', ''),
                'role': emp.get('role', ''),
                'skills': emp.get('skills', []),
                'capacity_hours': emp.get('capacity_hours', 40),
                'success_rate': emp.get('success_rate', 0.7),
                'scoring_breakdown': {
                    'skill_score': skill_score,
                    'capacity_score': capacity_score,
                    'role_score': role_score,
                    'performance_score': performance_score,
                    'urgency_score': urgency_score,
                    'ai_bonus': ai_bonus,
                    'total_score': total_score
                },
                'ai_prediction': ai_suggested_employee == emp.get('username', ''),
                'ai_confidence': ai_confidence
            }
            
            if total_score > best_score:
                best_score = total_score
                best_match = emp.get('username', '')
                best_details = emp_details
        
        return best_match, best_score, best_details
    
    def auto_assign_with_fallback(self, task: Dict) -> Dict:
        """Auto-assign with fallback to AI prediction"""
        try:
            # Get employee data from database
            employee_data = self.get_employee_data_from_db()
            
            if not employee_data:
                logger.warning("No employee data available, using AI prediction only")
                ai_prediction = self.get_ai_prediction(task)
                return {
                    'assigned_employee': ai_prediction.get('assigned_employee', 'unknown'),
                    'confidence': ai_prediction.get('confidence', 0.5),
                    'method': 'ai_only',
                    'details': ai_prediction
                }
            
            # Perform enhanced assignment
            best_match, best_score, best_details = self.enhanced_auto_assign(task, employee_data)
            
            # Get AI prediction for comparison
            ai_prediction = self.get_ai_prediction(task)
            
            return {
                'assigned_employee': best_match,
                'confidence': best_score,
                'method': 'enhanced_hybrid',
                'details': best_details,
                'ai_prediction': ai_prediction,
                'assignment_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in auto assignment: {e}")
            return {
                'assigned_employee': 'unknown',
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def get_assignment_recommendations(self, task: Dict, top_k: int = 3) -> List[Dict]:
        """Get top-k assignment recommendations"""
        try:
            employee_data = self.get_employee_data_from_db()
            
            if not employee_data:
                return []
            
            recommendations = []
            
            for emp in employee_data:
                skill_score = self.calculate_skill_match_score(task, emp)
                capacity_score = self.calculate_capacity_score(emp)
                role_score = self.calculate_role_match_score(task, emp)
                performance_score = self.calculate_performance_score(emp)
                urgency_score = self.calculate_urgency_compatibility(task, emp)
                
                total_score = (
                    skill_score * 0.35 +
                    capacity_score * 0.20 +
                    role_score * 0.25 +
                    performance_score * 0.15 +
                    urgency_score * 0.05
                )
                
                recommendations.append({
                    'employee_name': emp.get('full_name', emp.get('username', 'Unknown')),
                    'username': emp.get('username', ''),
                    'role': emp.get('role', ''),
                    'score': total_score,
                    'skill_match': skill_score,
                    'capacity_available': emp.get('capacity_hours', 40),
                    'success_rate': emp.get('success_rate', 0.7)
                })
            
            # Sort by score and return top-k
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

def main():
    """Test the enhanced auto-assignment system"""
    print("🚀 Testing Enhanced Auto-Assignment System")
    print("=" * 50)
    
    # Initialize assigner
    assigner = EnhancedAutoAssigner()
    
    # Test task
    test_task = {
        "title": "Fix critical security vulnerability in login system",
        "description": "Urgent fix needed for SQL injection vulnerability in user authentication. This affects all users and poses high security risk.",
        "category": "security",
        "urgency_score": 9,
        "complexity_score": 7,
        "business_impact": 9,
        "estimated_hours": 12,
        "days_until_deadline": 1
    }
    
    print(f"📋 Test Task: {test_task['title']}")
    print(f"🏷️  Category: {test_task['category']}")
    print(f"⚡ Urgency: {test_task['urgency_score']}/10")
    print(f"🧩 Complexity: {test_task['complexity_score']}/10")
    print()
    
    # Test assignment
    result = assigner.auto_assign_with_fallback(test_task)
    
    print("🎯 Assignment Result:")
    print(f"   👤 Assigned Employee: {result['assigned_employee']}")
    print(f"   📊 Confidence Score: {result['confidence']:.3f}")
    print(f"   🔧 Method: {result['method']}")
    
    if 'details' in result:
        details = result['details']
        print(f"\n📋 Assignment Details:")
        print(f"   👤 Employee: {details['employee_name']}")
        print(f"   🏷️  Role: {details['role']}")
        print(f"   🛠️  Skills: {', '.join(details['skills'][:5])}...")
        print(f"   ⏰ Capacity: {details['capacity_hours']} hours")
        print(f"   📈 Success Rate: {details['success_rate']:.1%}")
        
        scoring = details['scoring_breakdown']
        print(f"\n📊 Scoring Breakdown:")
        print(f"   🎯 Skill Match: {scoring['skill_score']:.3f}")
        print(f"   ⏰ Capacity: {scoring['capacity_score']:.3f}")
        print(f"   🏷️  Role Match: {scoring['role_score']:.3f}")
        print(f"   📈 Performance: {scoring['performance_score']:.3f}")
        print(f"   ⚡ Urgency: {scoring['urgency_score']:.3f}")
        print(f"   🤖 AI Bonus: {scoring['ai_bonus']:.3f}")
        print(f"   📊 Total Score: {scoring['total_score']:.3f}")
    
    # Get recommendations
    print(f"\n🎯 Top 3 Recommendations:")
    recommendations = assigner.get_assignment_recommendations(test_task, top_k=3)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['employee_name']} ({rec['role']}) - Score: {rec['score']:.3f}")
        print(f"      Skills: {rec['skill_match']:.3f}, Capacity: {rec['capacity_available']}h, Success: {rec['success_rate']:.1%}")
    
    print(f"\n✅ Enhanced auto-assignment system ready!")

if __name__ == "__main__":
    main() 