import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from api import predict, tasks
from api import main as api_main

app = FastAPI()

# Allow CORS for local frontend-backend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api")
app.include_router(tasks.router, prefix="/api")

# Mount the main API app (for /health and other endpoints)
app.mount("/", api_main.app)

# Load models (assuming they are in the models/ directory)
try:
    classifier = joblib.load('../models/classifier.pkl')
    priority_model = joblib.load('../models/priority_model.pkl')
except Exception as e:
    # Only warn if models are missing; allow API to start for dev
    print('Model loading failed:', e)
    classifier = None
    priority_model = None

# Simple route for prediction
# from backend.api.predict import predict_category, predict_priority, assign_task  # Assuming this is defined in predict.py
# app.include_router(predict_router, prefix='/api', tags=['predict'])

# To run backend:
# 1. pip install -r requirements.txt
# 2. uvicorn app:app --reload 