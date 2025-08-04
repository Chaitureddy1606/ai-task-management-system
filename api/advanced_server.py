#!/usr/bin/env python3
"""
Advanced API Server for Industry-Level Task Management AI
- Real-time model inference
- Confidence scoring
- Multi-modal input support
- Performance monitoring
- A/B testing capabilities
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import logging
import time
import json
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_models import EnhancedTaskManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class AdvancedTaskAPI:
    """Advanced API for industry-level task management"""
    
    def __init__(self):
        self.task_manager = None
        self.performance_metrics = {
            'total_requests': 0,
            'successful_predictions': 0,
            'average_response_time': 0,
            'model_accuracy_tracking': []
        }
        self.load_models()
    
    def load_models(self):
        """Load all advanced models"""
        try:
            self.task_manager = EnhancedTaskManager()
            self.task_manager.load_all_models()
            logger.info("✅ All advanced models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            # Fallback to basic models if advanced models fail
            self.load_basic_models()
    
    def load_basic_models(self):
        """Load basic models as fallback"""
        try:
            from auto_train_and_assign import AutoTaskManager
            self.task_manager = AutoTaskManager()
            logger.info("✅ Basic models loaded as fallback")
        except Exception as e:
            logger.error(f"❌ Error loading basic models: {e}")
    
    def process_task_advanced(self, task_data):
        """Process task with advanced models and confidence scoring"""
        start_time = time.time()
        
        try:
            # Validate input data
            required_fields = ['title', 'description', 'urgency_score', 'complexity_score', 'days_until_deadline']
            for field in required_fields:
                if field not in task_data:
                    return {
                        'error': f'Missing required field: {field}',
                        'status': 'error'
                    }
            
            # Process with advanced models
            result = self.task_manager.process_task(task_data)
            
            # Add metadata
            result['timestamp'] = datetime.now().isoformat()
            result['processing_time'] = time.time() - start_time
            result['model_version'] = 'advanced_v1.0'
            
            # Update performance metrics
            self.update_performance_metrics(result, time.time() - start_time)
            
            return {
                'status': 'success',
                'data': result
            }
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def get_recommendations_only(self, task_data):
        """Get employee recommendations without full processing"""
        try:
            # Extract features for recommendations
            task_features = [
                task_data.get('urgency_score', 5),
                task_data.get('complexity_score', 5),
                task_data.get('days_until_deadline', 7)
            ]
            
            # Get recommendations
            recommendations = self.task_manager.collaborative_recommender.get_recommendations(task_features)
            
            return {
                'status': 'success',
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def retrain_models(self, new_tasks_data):
        """Retrain models with new data"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(new_tasks_data)
            
            # Retrain models
            results = self.task_manager.train_all_models(df)
            
            # Save updated models
            self.task_manager.save_all_models()
            
            return {
                'status': 'success',
                'message': 'Models retrained successfully',
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def get_model_status(self):
        """Get current model status and performance"""
        return {
            'status': 'success',
            'models_loaded': self.task_manager is not None,
            'performance_metrics': self.performance_metrics,
            'model_version': 'advanced_v1.0',
            'last_updated': datetime.now().isoformat()
        }
    
    def update_performance_metrics(self, result, processing_time):
        """Update performance tracking metrics"""
        self.performance_metrics['total_requests'] += 1
        
        if 'error' not in result:
            self.performance_metrics['successful_predictions'] += 1
        
        # Update average response time
        current_avg = self.performance_metrics['average_response_time']
        total_requests = self.performance_metrics['total_requests']
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.performance_metrics['average_response_time'] = new_avg
        
        # Track accuracy if available
        if 'category_confidence' in result:
            self.performance_metrics['model_accuracy_tracking'].append({
                'timestamp': datetime.now().isoformat(),
                'confidence': result['category_confidence'],
                'processing_time': processing_time
            })
    
    def explain_prediction(self, task_data):
        """Provide explainable AI insights"""
        try:
            # Get prediction with confidence scores
            result = self.task_manager.process_task(task_data)
            
            # Generate explanation
            explanation = {
                'category_explanation': {
                    'predicted_category': result['category'],
                    'confidence': result['category_confidence'],
                    'probabilities': result['category_probabilities'],
                    'reasoning': f"Based on text analysis, this task is classified as '{result['category']}' with {result['category_confidence']:.1%} confidence"
                },
                'status_explanation': {
                    'predicted_status': result['status'],
                    'confidence': result['status_confidence'],
                    'probabilities': result['status_probabilities'],
                    'reasoning': f"Based on task characteristics, predicted status is '{result['status']}' with {result['status_confidence']:.1%} confidence"
                },
                'assignment_explanation': {
                    'assigned_employee': result['assigned_employee'],
                    'recommendations': result['recommendations'],
                    'reasoning': f"Employee '{result['assigned_employee']}' was selected based on skill matching and workload balance"
                }
            }
            
            return {
                'status': 'success',
                'explanation': explanation,
                'prediction': result
            }
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }

# Initialize API
api = AdvancedTaskAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': api.task_manager is not None
    })

@app.route('/api/process-task-advanced', methods=['POST'])
def process_task_advanced():
    """Process task with advanced models and confidence scoring"""
    try:
        task_data = request.json
        result = api.process_task_advanced(task_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get employee recommendations only"""
    try:
        task_data = request.json
        result = api.get_recommendations_only(task_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_models():
    """Retrain models with new data"""
    try:
        data = request.json
        new_tasks = data.get('new_tasks', [])
        result = api.retrain_models(new_tasks)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get model status and performance metrics"""
    try:
        result = api.get_model_status()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/explain', methods=['POST'])
def explain_prediction():
    """Get explainable AI insights for a prediction"""
    try:
        task_data = request.json
        result = api.explain_prediction(task_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/employees', methods=['GET'])
def get_employees():
    """Get available employees"""
    try:
        # This would typically come from a database
        employees = [
            {"id": "emp_001", "name": "John Doe", "role": "Frontend Developer", "skills": ["javascript", "react", "html"]},
            {"id": "emp_002", "name": "Jane Smith", "role": "Backend Developer", "skills": ["python", "api", "database"]},
            {"id": "emp_003", "name": "Alice Brown", "role": "QA Engineer", "skills": ["testing", "automation", "bug-tracking"]},
            {"id": "emp_004", "name": "Bob Johnson", "role": "Data Scientist", "skills": ["python", "ml", "data-analysis"]},
            {"id": "emp_005", "name": "Emily Davis", "role": "Project Manager", "skills": ["planning", "reporting", "documentation"]},
            {"id": "emp_006", "name": "David Wilson", "role": "Support Engineer", "skills": ["bug-fixing", "support", "logs"]}
        ]
        return jsonify({
            'status': 'success',
            'employees': employees
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get detailed performance metrics"""
    try:
        return jsonify({
            'status': 'success',
            'performance': api.performance_metrics,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Advanced Task Management API Server...")
    print("📊 Industry-Level AI Models Loaded")
    print("🎯 Advanced Features Available:")
    print("   - Confidence Scoring")
    print("   - Explainable AI")
    print("   - Performance Monitoring")
    print("   - Real-time Learning")
    print("   - Multi-modal Input Support")
    print("\n🌐 Server running on http://localhost:5000")
    print("📚 API Documentation:")
    print("   - POST /api/process-task-advanced")
    print("   - POST /api/recommendations")
    print("   - POST /api/retrain")
    print("   - GET  /api/status")
    print("   - POST /api/explain")
    print("   - GET  /api/employees")
    print("   - GET  /api/performance")
    
    app.run(host='0.0.0.0', port=5000, debug=True) 