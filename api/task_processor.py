#!/usr/bin/env python3
"""
API endpoint for automated task processing and employee assignment
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append('..')

from auto_train_and_assign import AutoTaskManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskProcessorAPI:
    """API for processing tasks and providing recommendations"""
    
    def __init__(self):
        self.manager = AutoTaskManager()
        
        # Ensure models are trained
        if not self.manager.models_trained:
            logger.info("Training models on startup...")
            self.manager.train_models()
    
    def process_new_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a new task and return recommendations
        
        Args:
            task_data: Dictionary containing task information
                - title: Task title
                - description: Task description
                - urgency_score: 0-10 urgency level
                - complexity_score: 0-10 complexity level
                - days_until_deadline: Days until deadline
                - business_impact: 0-10 business impact
                - estimated_hours: Estimated hours to complete
        
        Returns:
            Dictionary with task analysis and employee recommendations
        """
        try:
            # Validate required fields
            required_fields = ['title', 'description']
            for field in required_fields:
                if field not in task_data:
                    return {
                        "error": f"Missing required field: {field}",
                        "status": "error"
                    }
            
            # Set default values for optional fields
            defaults = {
                'urgency_score': 5,
                'complexity_score': 5,
                'days_until_deadline': 30,
                'business_impact': 5.0,
                'estimated_hours': 8
            }
            
            for key, default_value in defaults.items():
                if key not in task_data:
                    task_data[key] = default_value
            
            # Process the task
            result = self.manager.process_new_task(task_data)
            
            if "error" in result:
                return {
                    "error": result["error"],
                    "status": "error"
                }
            
            # Add success status
            result["status"] = "success"
            result["message"] = "Task processed successfully"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def retrain_models(self, new_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Retrain models with new task data
        
        Args:
            new_tasks: List of new task dictionaries
        
        Returns:
            Dictionary with training results
        """
        try:
            success = self.manager.retrain_with_new_data(new_tasks)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Models retrained with {len(new_tasks)} new tasks",
                    "tasks_processed": len(new_tasks)
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to retrain models"
                }
                
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_employee_recommendations(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get employee recommendations for a task without full processing
        
        Args:
            task_data: Task information
        
        Returns:
            Dictionary with employee recommendations
        """
        try:
            # Process task to get recommendations
            result = self.process_new_task(task_data)
            
            if result["status"] == "error":
                return result
            
            # Return only the recommendations part
            return {
                "status": "success",
                "task_id": result.get("task_id"),
                "category": result.get("classification", {}).get("category"),
                "priority": result.get("priority", {}).get("level"),
                "priority_score": result.get("priority", {}).get("score"),
                "employee_recommendations": result.get("employee_recommendations", []),
                "best_match": result.get("best_match")
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def update_employee_workload(self, employee_id: str, additional_hours: float) -> Dict[str, Any]:
        """
        Update employee workload after task assignment
        
        Args:
            employee_id: Employee ID
            additional_hours: Hours to add to workload
        
        Returns:
            Dictionary with update result
        """
        try:
            success = self.manager.update_employee_workload(employee_id, additional_hours)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Updated workload for {employee_id}",
                    "employee_id": employee_id,
                    "additional_hours": additional_hours
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to update workload"
                }
                
        except Exception as e:
            logger.error(f"Error updating workload: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get current model status and performance metrics
        
        Returns:
            Dictionary with model status
        """
        try:
            status = {
                "status": "success",
                "models_trained": self.manager.models_trained,
                "classifier_loaded": self.manager.classifier is not None,
                "priority_model_loaded": self.manager.priority_model is not None,
                "assigner_loaded": self.manager.assigner is not None,
                "timestamp": datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

# Example usage and testing
def test_api():
    """Test the API functionality"""
    api = TaskProcessorAPI()
    
    # Test task processing
    test_task = {
        "title": "Fix critical payment processing bug",
        "description": "Users unable to complete credit card payments due to API timeout",
        "urgency_score": 9,
        "complexity_score": 6,
        "days_until_deadline": 2,
        "business_impact": 8.0,
        "estimated_hours": 12
    }
    
    print("Testing task processing...")
    result = api.process_new_task(test_task)
    print(json.dumps(result, indent=2))
    
    print("\nTesting model status...")
    status = api.get_model_status()
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    test_api() 