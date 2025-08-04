#!/usr/bin/env python3
"""
Flask API server for automated task processing and employee assignment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append('..')

from task_processor import TaskProcessorAPI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the task processor
task_processor = TaskProcessorAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Task Management API"
    })

@app.route('/api/process-task', methods=['POST'])
def process_task():
    """
    Process a new task and provide employee recommendations
    
    Expected JSON payload:
    {
        "title": "Task title",
        "description": "Task description",
        "urgency_score": 8,
        "complexity_score": 6,
        "days_until_deadline": 5,
        "business_impact": 7.5,
        "estimated_hours": 12
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "status": "error"
            }), 400
        
        # Process the task
        result = task_processor.process_new_task(data)
        
        if result.get("status") == "error":
            return jsonify(result), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing task: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """
    Get employee recommendations for a task
    
    Expected JSON payload:
    {
        "title": "Task title",
        "description": "Task description",
        "urgency_score": 8,
        "complexity_score": 6,
        "days_until_deadline": 5,
        "business_impact": 7.5,
        "estimated_hours": 12
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "status": "error"
            }), 400
        
        # Get recommendations
        result = task_processor.get_employee_recommendations(data)
        
        if result.get("status") == "error":
            return jsonify(result), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_models():
    """
    Retrain models with new task data
    
    Expected JSON payload:
    {
        "new_tasks": [
            {
                "title": "Task 1",
                "description": "Description 1",
                "category": "bug",
                "priority": "high",
                "urgency_score": 8,
                "complexity_score": 6,
                "days_until_deadline": 5,
                "assigned_to": "John Doe",
                "status": "completed"
            }
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'new_tasks' not in data:
            return jsonify({
                "error": "No new_tasks array provided",
                "status": "error"
            }), 400
        
        # Retrain models
        result = task_processor.retrain_models(data['new_tasks'])
        
        if result.get("status") == "error":
            return jsonify(result), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/update-workload', methods=['POST'])
def update_workload():
    """
    Update employee workload after task assignment
    
    Expected JSON payload:
    {
        "employee_id": "emp_001",
        "additional_hours": 12.5
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'employee_id' not in data or 'additional_hours' not in data:
            return jsonify({
                "error": "Missing employee_id or additional_hours",
                "status": "error"
            }), 400
        
        # Update workload
        result = task_processor.update_employee_workload(
            data['employee_id'], 
            data['additional_hours']
        )
        
        if result.get("status") == "error":
            return jsonify(result), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error updating workload: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get model status and performance metrics"""
    try:
        status = task_processor.get_model_status()
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/employees', methods=['GET'])
def get_employees():
    """Get list of all employees"""
    try:
        # Load employee profiles
        with open('data/employee_profiles.json', 'r') as f:
            employees = json.load(f)
        
        return jsonify({
            "status": "success",
            "employees": employees
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting employees: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get list of training tasks"""
    try:
        import pandas as pd
        
        if os.path.exists('data/training_tasks.csv'):
            df = pd.read_csv('data/training_tasks.csv')
            tasks = df.to_dict('records')
        else:
            tasks = []
        
        return jsonify({
            "status": "success",
            "tasks": tasks,
            "count": len(tasks)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "status": "error"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "status": "error"
    }), 500

if __name__ == '__main__':
    # Run the Flask app
    print("Starting AI Task Management API Server...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /api/process-task - Process new task")
    print("  POST /api/recommendations - Get employee recommendations")
    print("  POST /api/retrain - Retrain models with new data")
    print("  POST /api/update-workload - Update employee workload")
    print("  GET  /api/status - Get model status")
    print("  GET  /api/employees - Get employee list")
    print("  GET  /api/tasks - Get training tasks")
    
    app.run(host='0.0.0.0', port=5000, debug=True) 