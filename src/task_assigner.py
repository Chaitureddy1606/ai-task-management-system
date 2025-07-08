"""
Intelligent Task Assignment Module for AI Task Management System
Assigns tasks to employees based on skills, workload, and availability
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class IntelligentTaskAssigner:
    """Intelligent task assignment system"""
    
    def __init__(self, employee_profiles: List[Dict]):
        """
        Initialize task assigner with employee profiles
        
        Args:
            employee_profiles: List of employee profile dictionaries
        """
        self.employee_profiles = employee_profiles
        self.skill_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        self.is_fitted = False
        
        # Prepare employee skill vectors
        self._prepare_employee_skills()
    
    def _prepare_employee_skills(self):
        """Prepare employee skill vectors for similarity matching"""
        if not self.employee_profiles:
            logger.warning("No employee profiles provided")
            return
        
        # Extract all skills and expertise areas
        employee_skills = []
        for profile in self.employee_profiles:
            skills = profile.get('skills', [])
            expertise = profile.get('expertise_areas', [])
            preferred_types = profile.get('preferred_task_types', [])
            
            # Combine all skill-related information
            all_skills = skills + expertise + preferred_types
            skill_text = ' '.join(all_skills)
            employee_skills.append(skill_text)
        
        # Fit TF-IDF vectorizer if we have skills
        if any(employee_skills):
            self.employee_skill_vectors = self.skill_vectorizer.fit_transform(employee_skills)
            self.is_fitted = True
            logger.info(f"Prepared skill vectors for {len(self.employee_profiles)} employees")
        else:
            logger.warning("No skills found in employee profiles")
    
    def calculate_skill_match_score(self, task_description: str, task_category: str = "") -> np.ndarray:
        """
        Calculate skill match scores for all employees
        
        Args:
            task_description: Description of the task
            task_category: Category/type of the task
        
        Returns:
            Array of skill match scores for each employee
        """
        if not self.is_fitted:
            logger.warning("Skill vectorizer not fitted, returning default scores")
            return np.array([0.5] * len(self.employee_profiles))
        
        # Combine task description and category
        task_text = f"{task_description} {task_category}".strip()
        
        # Vectorize task requirements
        task_vector = self.skill_vectorizer.transform([task_text])
        
        # Calculate cosine similarity with employee skills
        similarities = cosine_similarity(task_vector, self.employee_skill_vectors)[0]
        
        return similarities
    
    def calculate_workload_score(self) -> np.ndarray:
        """Calculate workload availability scores for all employees"""
        workload_scores = []
        
        for profile in self.employee_profiles:
            current_workload = profile.get('current_workload', 5)
            max_capacity = profile.get('max_capacity', 10)
            
            if max_capacity == 0:
                workload_score = 0  # No capacity
            else:
                utilization = current_workload / max_capacity
                # Score is higher when utilization is lower (more available)
                workload_score = max(0, 1 - utilization)
            
            workload_scores.append(workload_score)
        
        return np.array(workload_scores)
    
    def calculate_preference_score(self, task_category: str) -> np.ndarray:
        """Calculate preference scores based on employee preferred task types"""
        preference_scores = []
        
        for profile in self.employee_profiles:
            preferred_types = profile.get('preferred_task_types', [])
            
            if not preferred_types:
                preference_score = 0.5  # Neutral if no preferences
            else:
                # Check if task category matches any preferred types
                task_category_lower = task_category.lower()
                matches = sum(1 for pref in preferred_types 
                             if pref.lower() in task_category_lower or 
                             task_category_lower in pref.lower())
                
                preference_score = min(1.0, matches / len(preferred_types) + 0.3)
            
            preference_scores.append(preference_score)
        
        return np.array(preference_scores)
    
    def calculate_experience_score(self, task_complexity: float) -> np.ndarray:
        """Calculate experience scores based on task complexity and employee experience"""
        experience_scores = []
        
        for profile in self.employee_profiles:
            experience_years = profile.get('experience_years', 1)
            
            # Normalize experience (assuming 10+ years is expert level)
            normalized_experience = min(1.0, experience_years / 10)
            
            # Match experience to task complexity
            # Complex tasks need more experienced people
            if task_complexity > 7:  # High complexity
                experience_score = normalized_experience
            elif task_complexity > 4:  # Medium complexity
                experience_score = 0.7 + 0.3 * normalized_experience
            else:  # Low complexity
                experience_score = 0.9  # Anyone can handle it
            
            experience_scores.append(experience_score)
        
        return np.array(experience_scores)
    
    def calculate_availability_score(self, estimated_hours: float, deadline_days: int) -> np.ndarray:
        """Calculate availability scores based on time requirements"""
        availability_scores = []
        
        for profile in self.employee_profiles:
            availability = profile.get('availability', {})
            
            if not availability:
                # Default availability if not specified
                avg_daily_hours = 6
            else:
                # Calculate average daily availability
                daily_hours = list(availability.values())
                avg_daily_hours = sum(daily_hours) / len(daily_hours) if daily_hours else 6
            
            # Calculate if employee can complete task by deadline
            available_hours = avg_daily_hours * max(1, deadline_days)
            
            if available_hours >= estimated_hours:
                # Can complete on time
                availability_score = 1.0
            elif available_hours >= estimated_hours * 0.7:
                # Might complete with some overtime
                availability_score = 0.7
            else:
                # Unlikely to complete on time
                availability_score = 0.3
            
            availability_scores.append(availability_score)
        
        return np.array(availability_scores)
    
    def assign_task(self, 
                   task: Dict[str, Any],
                   weights: Dict[str, float] = None) -> List[Tuple[str, float]]:
        """
        Assign a task to the best suited employee
        
        Args:
            task: Task dictionary with details
            weights: Weights for different scoring factors
        
        Returns:
            List of (employee_id, score) tuples sorted by score
        """
        if not self.employee_profiles:
            logger.error("No employee profiles available for assignment")
            return []
        
        # Default weights
        if weights is None:
            weights = {
                'skill_match': 0.35,
                'workload': 0.25,
                'preference': 0.15,
                'experience': 0.15,
                'availability': 0.10
            }
        
        # Extract task information
        description = task.get('description', '')
        category = task.get('category', '')
        complexity = task.get('complexity_score', 5.0)
        estimated_hours = task.get('estimated_hours', 8.0)
        
        # Calculate days until deadline
        deadline = task.get('deadline', '')
        try:
            if deadline:
                deadline_date = pd.to_datetime(deadline)
                today = pd.Timestamp.now()
                deadline_days = max(1, (deadline_date - today).days)
            else:
                deadline_days = 30  # Default 30 days
        except:
            deadline_days = 30
        
        # Calculate individual scores
        skill_scores = self.calculate_skill_match_score(description, category)
        workload_scores = self.calculate_workload_score()
        preference_scores = self.calculate_preference_score(category)
        experience_scores = self.calculate_experience_score(complexity)
        availability_scores = self.calculate_availability_score(estimated_hours, deadline_days)
        
        # Calculate weighted total scores
        total_scores = (
            skill_scores * weights['skill_match'] +
            workload_scores * weights['workload'] +
            preference_scores * weights['preference'] +
            experience_scores * weights['experience'] +
            availability_scores * weights['availability']
        )
        
        # Create results with employee IDs and scores
        results = []
        for i, profile in enumerate(self.employee_profiles):
            employee_id = profile.get('employee_id', f'EMP{i:03d}')
            score = total_scores[i]
            results.append((employee_id, score))
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Log assignment details
        if results:
            best_employee = results[0]
            logger.info(f"Best assignment for task: {best_employee[0]} (score: {best_employee[1]:.3f})")
        
        return results
    
    def assign_multiple_tasks(self, 
                             tasks: List[Dict[str, Any]], 
                             balance_workload: bool = True) -> Dict[str, List[Dict]]:
        """
        Assign multiple tasks optimizing for workload balance
        
        Args:
            tasks: List of task dictionaries
            balance_workload: Whether to balance workload across employees
        
        Returns:
            Dictionary mapping employee_id to list of assigned tasks
        """
        assignments = {profile['employee_id']: [] for profile in self.employee_profiles}
        current_workloads = {profile['employee_id']: profile.get('current_workload', 0) 
                           for profile in self.employee_profiles}
        
        # Sort tasks by priority if available
        tasks_sorted = sorted(tasks, 
                            key=lambda x: x.get('priority_score', 5), 
                            reverse=True)
        
        for task in tasks_sorted:
            # Get assignment recommendations
            recommendations = self.assign_task(task)
            
            if not recommendations:
                continue
            
            # Find best available employee
            assigned = False
            for employee_id, score in recommendations:
                # Check capacity if balancing workload
                if balance_workload:
                    current_load = current_workloads[employee_id]
                    max_capacity = next(
                        (p['max_capacity'] for p in self.employee_profiles 
                         if p['employee_id'] == employee_id), 10
                    )
                    
                    task_hours = task.get('estimated_hours', 8)
                    
                    if current_load + task_hours <= max_capacity:
                        # Assign task
                        assignments[employee_id].append(task)
                        current_workloads[employee_id] += task_hours
                        assigned = True
                        break
                else:
                    # Assign to best match regardless of workload
                    assignments[recommendations[0][0]].append(task)
                    assigned = True
                    break
            
            if not assigned:
                logger.warning(f"Could not assign task: {task.get('title', 'Unknown')}")
        
        # Remove employees with no assignments
        assignments = {emp_id: tasks for emp_id, tasks in assignments.items() if tasks}
        
        return assignments
    
    def get_assignment_explanation(self, 
                                  task: Dict[str, Any], 
                                  employee_id: str) -> Dict[str, Any]:
        """Get detailed explanation for why a task was assigned to an employee"""
        # Find employee profile
        employee_profile = None
        for profile in self.employee_profiles:
            if profile['employee_id'] == employee_id:
                employee_profile = profile
                break
        
        if not employee_profile:
            return {"error": "Employee not found"}
        
        # Calculate all scores for this employee
        description = task.get('description', '')
        category = task.get('category', '')
        complexity = task.get('complexity_score', 5.0)
        estimated_hours = task.get('estimated_hours', 8.0)
        
        # Get deadline days
        deadline = task.get('deadline', '')
        try:
            if deadline:
                deadline_date = pd.to_datetime(deadline)
                today = pd.Timestamp.now()
                deadline_days = max(1, (deadline_date - today).days)
            else:
                deadline_days = 30
        except:
            deadline_days = 30
        
        # Calculate individual scores for this employee
        employee_index = next(i for i, p in enumerate(self.employee_profiles) 
                            if p['employee_id'] == employee_id)
        
        skill_scores = self.calculate_skill_match_score(description, category)
        workload_scores = self.calculate_workload_score()
        preference_scores = self.calculate_preference_score(category)
        experience_scores = self.calculate_experience_score(complexity)
        availability_scores = self.calculate_availability_score(estimated_hours, deadline_days)
        
        explanation = {
            'employee_name': employee_profile.get('name', 'Unknown'),
            'task_title': task.get('title', 'Unknown Task'),
            'scores': {
                'skill_match': float(skill_scores[employee_index]),
                'workload_availability': float(workload_scores[employee_index]),
                'task_preference': float(preference_scores[employee_index]),
                'experience_match': float(experience_scores[employee_index]),
                'time_availability': float(availability_scores[employee_index])
            },
            'employee_details': {
                'skills': employee_profile.get('skills', []),
                'experience_years': employee_profile.get('experience_years', 0),
                'current_workload': employee_profile.get('current_workload', 0),
                'max_capacity': employee_profile.get('max_capacity', 10),
                'preferred_task_types': employee_profile.get('preferred_task_types', [])
            },
            'task_requirements': {
                'category': category,
                'complexity': complexity,
                'estimated_hours': estimated_hours,
                'deadline_days': deadline_days
            }
        }
        
        return explanation
    
    def update_employee_workload(self, employee_id: str, additional_hours: float) -> bool:
        """Update employee workload after task assignment"""
        for profile in self.employee_profiles:
            if profile['employee_id'] == employee_id:
                current_workload = profile.get('current_workload', 0)
                profile['current_workload'] = current_workload + additional_hours
                logger.info(f"Updated workload for {employee_id}: +{additional_hours} hours")
                return True
        
        logger.warning(f"Employee {employee_id} not found for workload update")
        return False
    
    def get_team_workload_summary(self) -> Dict[str, Any]:
        """Get summary of team workload distribution"""
        summary = {
            'employees': [],
            'total_capacity': 0,
            'total_workload': 0,
            'average_utilization': 0
        }
        
        utilizations = []
        for profile in self.employee_profiles:
            workload = profile.get('current_workload', 0)
            capacity = profile.get('max_capacity', 10)
            utilization = (workload / capacity * 100) if capacity > 0 else 0
            
            employee_info = {
                'employee_id': profile['employee_id'],
                'name': profile.get('name', 'Unknown'),
                'workload': workload,
                'capacity': capacity,
                'utilization': utilization,
                'available_hours': max(0, capacity - workload)
            }
            
            summary['employees'].append(employee_info)
            summary['total_capacity'] += capacity
            summary['total_workload'] += workload
            utilizations.append(utilization)
        
        if utilizations:
            summary['average_utilization'] = sum(utilizations) / len(utilizations)
        
        return summary 