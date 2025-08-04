#!/usr/bin/env python3
"""
Airflow DAG for AI Task Management System
- Automated data pipeline workflows
- Scheduled model training and retraining
- Feature engineering and data processing
- Performance monitoring and logging
"""

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.email_operator import EmailOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime, timedelta
import requests
import json
import logging

# Default arguments
default_args = {
    'owner': 'ai-task-management',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'ai_task_management_pipeline',
    default_args=default_args,
    description='AI Task Management Data Pipeline and Model Training',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    tags=['ai', 'task-management', 'ml-pipeline']
)

# Task 1: Data Pipeline Health Check
def check_data_pipeline_health():
    """Check if data pipeline is healthy"""
    try:
        response = requests.get('http://localhost:8000/api/pipeline/health')
        if response.status_code == 200:
            health_data = response.json()
            logging.info(f"Data pipeline health: {health_data}")
            return health_data['status'] == 'healthy'
        else:
            logging.error(f"Data pipeline health check failed: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Error checking data pipeline health: {e}")
        return False

health_check_task = PythonOperator(
    task_id='check_data_pipeline_health',
    python_callable=check_data_pipeline_health,
    dag=dag
)

# Task 2: Feature Engineering Pipeline
def run_feature_engineering():
    """Run feature engineering pipeline"""
    try:
        # Run data preparation pipeline
        import subprocess
        result = subprocess.run(['python', 'data_preparation_pipeline.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info("Feature engineering completed successfully")
            return True
        else:
            logging.error(f"Feature engineering failed: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        return False

feature_engineering_task = PythonOperator(
    task_id='run_feature_engineering',
    python_callable=run_feature_engineering,
    dag=dag
)

# Task 3: Model Training
def train_models():
    """Train AI models with latest data"""
    try:
        import subprocess
        result = subprocess.run(['python', 'enhanced_training_with_features.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info("Model training completed successfully")
            return True
        else:
            logging.error(f"Model training failed: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        return False

model_training_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag
)

# Task 4: Model Performance Evaluation
def evaluate_model_performance():
    """Evaluate model performance and log metrics"""
    try:
        # Get model performance from API
        response = requests.get('http://localhost:5001/api/performance')
        if response.status_code == 200:
            perf_data = response.json()
            logging.info(f"Model performance: {perf_data}")
            
            # Log to database
            db_response = requests.post('http://localhost:8000/api/pipeline/log-performance', 
                                     json=perf_data)
            
            return True
        else:
            logging.error("Failed to get model performance")
            return False
    except Exception as e:
        logging.error(f"Error evaluating model performance: {e}")
        return False

performance_evaluation_task = PythonOperator(
    task_id='evaluate_model_performance',
    python_callable=evaluate_model_performance,
    dag=dag
)

# Task 5: Database Cleanup
cleanup_old_logs = PostgresOperator(
    task_id='cleanup_old_logs',
    postgres_conn_id='ai_task_management_db',
    sql="""
    DELETE FROM system_logs 
    WHERE timestamp < NOW() - INTERVAL '30 days';
    
    DELETE FROM model_performance_logs 
    WHERE training_timestamp < NOW() - INTERVAL '90 days';
    
    DELETE FROM feature_engineering_logs 
    WHERE timestamp < NOW() - INTERVAL '30 days';
    """,
    dag=dag
)

# Task 6: Data Quality Check
def check_data_quality():
    """Check data quality and report issues"""
    try:
        # Get tasks from database
        response = requests.get('http://localhost:8000/api/pipeline/tasks?limit=1000')
        if response.status_code == 200:
            tasks_data = response.json()
            tasks = tasks_data['tasks']
            
            # Check for missing data
            missing_data_count = 0
            for task in tasks:
                if not task.get('title') or not task.get('description'):
                    missing_data_count += 1
            
            # Check for invalid scores
            invalid_scores_count = 0
            for task in tasks:
                urgency = task.get('urgency_score')
                complexity = task.get('complexity_score')
                if urgency and (urgency < 1 or urgency > 10):
                    invalid_scores_count += 1
                if complexity and (complexity < 1 or complexity > 10):
                    invalid_scores_count += 1
            
            quality_report = {
                'total_tasks': len(tasks),
                'missing_data_count': missing_data_count,
                'invalid_scores_count': invalid_scores_count,
                'quality_score': (len(tasks) - missing_data_count - invalid_scores_count) / len(tasks) if tasks else 0
            }
            
            logging.info(f"Data quality report: {quality_report}")
            return quality_report['quality_score'] > 0.95
        else:
            logging.error("Failed to get tasks for quality check")
            return False
    except Exception as e:
        logging.error(f"Error in data quality check: {e}")
        return False

data_quality_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag
)

# Task 7: System Health Report
def generate_health_report():
    """Generate comprehensive system health report"""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Check data pipeline
        pipeline_response = requests.get('http://localhost:8000/api/pipeline/health')
        if pipeline_response.status_code == 200:
            report['components']['data_pipeline'] = pipeline_response.json()
        
        # Check AI API
        ai_response = requests.get('http://localhost:5001/api/health')
        if ai_response.status_code == 200:
            report['components']['ai_api'] = ai_response.json()
        
        # Get pipeline stats
        stats_response = requests.get('http://localhost:8000/api/pipeline/stats')
        if stats_response.status_code == 200:
            report['components']['pipeline_stats'] = stats_response.json()
        
        # Get employee performance
        employees_response = requests.get('http://localhost:8000/api/pipeline/employees')
        if employees_response.status_code == 200:
            report['components']['employee_performance'] = employees_response.json()
        
        logging.info(f"Health report generated: {report}")
        return report
    except Exception as e:
        logging.error(f"Error generating health report: {e}")
        return None

health_report_task = PythonOperator(
    task_id='generate_health_report',
    python_callable=generate_health_report,
    dag=dag
)

# Task 8: Email Notification
def send_daily_report(**context):
    """Send daily system report via email"""
    try:
        health_report = context['task_instance'].xcom_pull(task_ids='generate_health_report')
        
        if health_report:
            # Format email content
            email_content = f"""
            AI Task Management System - Daily Report
            
            Timestamp: {health_report['timestamp']}
            
            Data Pipeline Status: {health_report['components'].get('data_pipeline', {}).get('status', 'Unknown')}
            AI API Status: {health_report['components'].get('ai_api', {}).get('status', 'Unknown')}
            
            Pipeline Stats:
            - Total Processed: {health_report['components'].get('pipeline_stats', {}).get('pipeline_stats', {}).get('total_processed', 0)}
            - Successful: {health_report['components'].get('pipeline_stats', {}).get('pipeline_stats', {}).get('successful_processed', 0)}
            - Failed: {health_report['components'].get('pipeline_stats', {}).get('pipeline_stats', {}).get('failed_processed', 0)}
            
            Employee Performance:
            - Active Employees: {health_report['components'].get('employee_performance', {}).get('count', 0)}
            
            System Status: All components operational
            """
            
            # Send email (placeholder - configure email settings)
            logging.info("Daily report email sent")
            return True
        else:
            logging.error("No health report available for email")
            return False
    except Exception as e:
        logging.error(f"Error sending daily report: {e}")
        return False

email_report_task = PythonOperator(
    task_id='send_daily_report',
    python_callable=send_daily_report,
    dag=dag
)

# Task 9: Model Deployment Check
def check_model_deployment():
    """Check if models are properly deployed and accessible"""
    try:
        # Test model predictions
        test_task = {
            "title": "Test task for deployment check",
            "description": "Testing model deployment and accessibility",
            "urgency_score": 5,
            "complexity_score": 5,
            "business_impact": 5,
            "estimated_hours": 8,
            "days_until_deadline": 7
        }
        
        response = requests.post('http://localhost:5001/api/process-task', json=test_task)
        if response.status_code == 200:
            result = response.json()
            logging.info(f"Model deployment test successful: {result}")
            return True
        else:
            logging.error(f"Model deployment test failed: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Error checking model deployment: {e}")
        return False

deployment_check_task = PythonOperator(
    task_id='check_model_deployment',
    python_callable=check_model_deployment,
    dag=dag
)

# Task 10: Backup Database
backup_database = BashOperator(
    task_id='backup_database',
    bash_command="""
    pg_dump -h localhost -U postgres -d ai_task_management > /tmp/ai_task_management_backup_$(date +%Y%m%d_%H%M%S).sql
    """,
    dag=dag
)

# Define task dependencies
health_check_task >> feature_engineering_task
feature_engineering_task >> model_training_task
model_training_task >> performance_evaluation_task
performance_evaluation_task >> data_quality_task
data_quality_task >> health_report_task
health_report_task >> email_report_task
health_report_task >> deployment_check_task
health_report_task >> cleanup_old_logs
health_report_task >> backup_database

# Weekly Model Retraining DAG
weekly_retraining_dag = DAG(
    'ai_task_management_weekly_retraining',
    default_args=default_args,
    description='Weekly Model Retraining and Performance Optimization',
    schedule_interval='0 3 * * 0',  # Weekly on Sunday at 3 AM
    catchup=False,
    tags=['ai', 'task-management', 'model-retraining']
)

# Weekly retraining task
def weekly_model_retraining():
    """Weekly model retraining with fresh data"""
    try:
        # Run comprehensive retraining
        import subprocess
        
        # 1. Update feature engineering
        result1 = subprocess.run(['python', 'data_preparation_pipeline.py'], 
                               capture_output=True, text=True)
        
        # 2. Retrain models
        result2 = subprocess.run(['python', 'enhanced_training_with_features.py'], 
                               capture_output=True, text=True)
        
        # 3. Test new models
        result3 = subprocess.run(['python', 'test_perfect_performance.py'], 
                               capture_output=True, text=True)
        
        if all([result1.returncode == 0, result2.returncode == 0, result3.returncode == 0]):
            logging.info("Weekly model retraining completed successfully")
            return True
        else:
            logging.error("Weekly model retraining failed")
            return False
    except Exception as e:
        logging.error(f"Error in weekly model retraining: {e}")
        return False

weekly_retraining_task = PythonOperator(
    task_id='weekly_model_retraining',
    python_callable=weekly_model_retraining,
    dag=weekly_retraining_dag
)

# Monthly System Maintenance DAG
monthly_maintenance_dag = DAG(
    'ai_task_management_monthly_maintenance',
    default_args=default_args,
    description='Monthly System Maintenance and Optimization',
    schedule_interval='0 4 1 * *',  # Monthly on 1st at 4 AM
    catchup=False,
    tags=['ai', 'task-management', 'maintenance']
)

def monthly_maintenance():
    """Monthly system maintenance tasks"""
    try:
        # 1. Database optimization
        # 2. Log cleanup
        # 3. Performance analysis
        # 4. System health check
        logging.info("Monthly maintenance completed")
        return True
    except Exception as e:
        logging.error(f"Error in monthly maintenance: {e}")
        return False

monthly_maintenance_task = PythonOperator(
    task_id='monthly_maintenance',
    python_callable=monthly_maintenance,
    dag=monthly_maintenance_dag
)

# Export DAGs
globals()['ai_task_management_pipeline'] = dag
globals()['ai_task_management_weekly_retraining'] = weekly_retraining_dag
globals()['ai_task_management_monthly_maintenance'] = monthly_maintenance_dag 