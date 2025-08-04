#!/usr/bin/env python3
"""
Upload Processor API for AI Task Management System
- Handle file uploads (CSV, JSON, Excel, TXT)
- Process data through the pipeline
- Store in database
- Return processing results
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import io
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database_setup import DatabaseManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Upload Processor API",
    description="Handle file uploads and process data through the pipeline",
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

# Initialize database
db = DatabaseManager()

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    db.connect()
    logger.info("Upload processor initialized")

@app.post("/api/upload/process-csv")
async def process_csv_upload(file: UploadFile = File(...)):
    """Process CSV file upload"""
    try:
        # Validate file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV content
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Process each row
        processed_tasks = []
        for index, row in df.iterrows():
            # Handle missing values and NaN
            def safe_int(value, default=5):
                if pd.isna(value) or value == '':
                    return default
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    return default
            
            def safe_float(value, default=8.0):
                if pd.isna(value) or value == '':
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            def safe_str(value, default=''):
                if pd.isna(value) or value == '':
                    return default
                return str(value)
            
            task_data = {
                "title": safe_str(row.get('title', f'Task {index + 1}')),
                "description": safe_str(row.get('description', '')),
                "category": safe_str(row.get('category', 'feature')),
                "priority": safe_str(row.get('priority', 'medium')),
                "urgency_score": safe_int(row.get('urgency_score', 5)),
                "complexity_score": safe_int(row.get('complexity_score', 5)),
                "business_impact": safe_int(row.get('business_impact', 5)),
                "estimated_hours": safe_float(row.get('estimated_hours', 8.0)),
                "days_until_deadline": safe_int(row.get('days_until_deadline', 7)),
                "assigned_to": row.get('assigned_to') if not pd.isna(row.get('assigned_to')) else None,
                "created_by": 1
            }
            
            # Insert into database
            cursor = db.connection.cursor()
            cursor.execute("""
                INSERT INTO tasks (
                    title, description, category, priority, urgency_score, 
                    complexity_score, business_impact, estimated_hours, 
                    days_until_deadline, status, assigned_to, created_by,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_data["title"],
                task_data["description"],
                task_data["category"],
                task_data["priority"],
                task_data["urgency_score"],
                task_data["complexity_score"],
                task_data["business_impact"],
                task_data["estimated_hours"],
                task_data["days_until_deadline"],
                "pending",
                task_data["assigned_to"],
                task_data["created_by"],
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            task_id = cursor.lastrowid
            processed_tasks.append({
                "task_id": task_id,
                "title": task_data["title"],
                "status": "processed"
            })
        
        db.connection.commit()
        
        return {
            "success": True,
            "message": f"Successfully processed {len(processed_tasks)} tasks from CSV",
            "processed_tasks": processed_tasks,
            "total_rows": len(df),
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.post("/api/upload/process-json")
async def process_json_upload(file: UploadFile = File(...)):
    """Process JSON file upload"""
    try:
        # Validate file
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be a JSON")
        
        # Read JSON content
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        # Process data (could be employees, tasks, or other data)
        processed_items = []
        
        if isinstance(data, list):
            # Process list of items
            for item in data:
                if 'name' in item and 'role' in item:
                    # Employee data
                    cursor = db.connection.cursor()
                    cursor.execute("""
                        INSERT OR IGNORE INTO users (username, email, full_name, role, department, skills, capacity_hours)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        item.get('username', item.get('name', '').lower().replace(' ', '_')),
                        item.get('email', f"{item.get('name', '').lower().replace(' ', '_')}@company.com"),
                        item.get('name', ''),
                        item.get('role', 'employee'),
                        item.get('department', ''),
                        json.dumps(item.get('skills', [])),
                        item.get('capacity_hours', 40)
                    ))
                    processed_items.append({
                        "type": "employee",
                        "name": item.get('name', ''),
                        "status": "processed"
                    })
                else:
                    # Task data
                    task_data = {
                        "title": item.get('title', 'Task'),
                        "description": item.get('description', ''),
                        "category": item.get('category', 'feature'),
                        "priority": item.get('priority', 'medium'),
                        "urgency_score": int(item.get('urgency_score', 5)),
                        "complexity_score": int(item.get('complexity_score', 5)),
                        "business_impact": int(item.get('business_impact', 5)),
                        "estimated_hours": float(item.get('estimated_hours', 8.0)),
                        "days_until_deadline": int(item.get('days_until_deadline', 7)),
                        "assigned_to": item.get('assigned_to'),
                        "created_by": 1
                    }
                    
                    cursor = db.connection.cursor()
                    cursor.execute("""
                        INSERT INTO tasks (
                            title, description, category, priority, urgency_score, 
                            complexity_score, business_impact, estimated_hours, 
                            days_until_deadline, status, assigned_to, created_by,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        task_data["title"],
                        task_data["description"],
                        task_data["category"],
                        task_data["priority"],
                        task_data["urgency_score"],
                        task_data["complexity_score"],
                        task_data["business_impact"],
                        task_data["estimated_hours"],
                        task_data["days_until_deadline"],
                        "pending",
                        task_data["assigned_to"],
                        task_data["created_by"],
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    ))
                    
                    task_id = cursor.lastrowid
                    processed_items.append({
                        "type": "task",
                        "task_id": task_id,
                        "title": task_data["title"],
                        "status": "processed"
                    })
        
        db.connection.commit()
        
        return {
            "success": True,
            "message": f"Successfully processed {len(processed_items)} items from JSON",
            "processed_items": processed_items,
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Error processing JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing JSON: {str(e)}")

@app.post("/api/upload/process-excel")
async def process_excel_upload(file: UploadFile = File(...)):
    """Process Excel file upload"""
    try:
        # Validate file
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File must be an Excel file")
        
        # Read Excel content
        content = await file.read()
        df = pd.read_excel(io.BytesIO(content))
        
        # Process each row
        processed_tasks = []
        for index, row in df.iterrows():
            task_data = {
                "title": row.get('title', f'Task {index + 1}'),
                "description": row.get('description', ''),
                "category": row.get('category', 'feature'),
                "priority": row.get('priority', 'medium'),
                "urgency_score": int(row.get('urgency_score', 5)),
                "complexity_score": int(row.get('complexity_score', 5)),
                "business_impact": int(row.get('business_impact', 5)),
                "estimated_hours": float(row.get('estimated_hours', 8.0)),
                "days_until_deadline": int(row.get('days_until_deadline', 7)),
                "assigned_to": row.get('assigned_to'),
                "created_by": 1
            }
            
            # Insert into database
            cursor = db.connection.cursor()
            cursor.execute("""
                INSERT INTO tasks (
                    title, description, category, priority, urgency_score, 
                    complexity_score, business_impact, estimated_hours, 
                    days_until_deadline, status, assigned_to, created_by,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_data["title"],
                task_data["description"],
                task_data["category"],
                task_data["priority"],
                task_data["urgency_score"],
                task_data["complexity_score"],
                task_data["business_impact"],
                task_data["estimated_hours"],
                task_data["days_until_deadline"],
                "pending",
                task_data["assigned_to"],
                task_data["created_by"],
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            task_id = cursor.lastrowid
            processed_tasks.append({
                "task_id": task_id,
                "title": task_data["title"],
                "status": "processed"
            })
        
        db.connection.commit()
        
        return {
            "success": True,
            "message": f"Successfully processed {len(processed_tasks)} tasks from Excel",
            "processed_tasks": processed_tasks,
            "total_rows": len(df),
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Error processing Excel: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing Excel: {str(e)}")

@app.post("/api/upload/process-text")
async def process_text_upload(file: UploadFile = File(...)):
    """Process text file upload"""
    try:
        # Validate file
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="File must be a text file")
        
        # Read text content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Simple text processing - extract potential tasks
        lines = text_content.split('\n')
        processed_items = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line and len(line) > 10:  # Skip empty or very short lines
                # Create a task from the line
                task_data = {
                    "title": line[:50] + "..." if len(line) > 50 else line,
                    "description": line,
                    "category": "documentation",
                    "priority": "medium",
                    "urgency_score": 5,
                    "complexity_score": 3,
                    "business_impact": 5,
                    "estimated_hours": 2.0,
                    "days_until_deadline": 7,
                    "assigned_to": None,
                    "created_by": 1
                }
                
                # Insert into database
                cursor = db.connection.cursor()
                cursor.execute("""
                    INSERT INTO tasks (
                        title, description, category, priority, urgency_score, 
                        complexity_score, business_impact, estimated_hours, 
                        days_until_deadline, status, assigned_to, created_by,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_data["title"],
                    task_data["description"],
                    task_data["category"],
                    task_data["priority"],
                    task_data["urgency_score"],
                    task_data["complexity_score"],
                    task_data["business_impact"],
                    task_data["estimated_hours"],
                    task_data["days_until_deadline"],
                    "pending",
                    task_data["assigned_to"],
                    task_data["created_by"],
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                
                task_id = cursor.lastrowid
                processed_items.append({
                    "task_id": task_id,
                    "title": task_data["title"],
                    "status": "processed"
                })
        
        db.connection.commit()
        
        return {
            "success": True,
            "message": f"Successfully processed {len(processed_items)} items from text file",
            "processed_items": processed_items,
            "total_lines": len(lines),
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Error processing text file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing text file: {str(e)}")

@app.get("/api/upload/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database_connected": db.connection is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Upload Processor API")
    print("=" * 50)
    print("✅ File upload processing")
    print("✅ CSV, JSON, Excel, TXT support")
    print("✅ Database integration")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8001) 