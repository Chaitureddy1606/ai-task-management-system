"""
Data preprocessing module for the AI Task Management System
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class TaskDataPreprocessor:
    """Preprocessor for task management data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_urgency_indicators(self, text: str) -> int:
        """Extract urgency indicators from text"""
        if pd.isna(text):
            return 0
        
        text = text.lower()
        urgency_keywords = {
            'urgent': 3,
            'asap': 3,
            'critical': 3,
            'emergency': 3,
            'immediate': 3,
            'high priority': 2,
            'important': 2,
            'soon': 1,
            'when possible': -1,
            'low priority': -2
        }
        
        urgency_score = 0
        for keyword, score in urgency_keywords.items():
            if keyword in text:
                urgency_score += score
        
        return max(-2, min(3, urgency_score))  # Clamp between -2 and 3
    
    def calculate_complexity_score(self, description: str, estimated_hours: float) -> float:
        """Calculate task complexity based on description and estimated hours"""
        if pd.isna(description):
            description = ""
        
        # Base complexity from estimated hours
        if pd.isna(estimated_hours) or estimated_hours <= 0:
            hour_complexity = 1
        else:
            # Normalize hours to 0-10 scale
            hour_complexity = min(10, max(1, estimated_hours / 2))
        
        # Additional complexity from description
        text_complexity = 1
        if description:
            complexity_indicators = [
                'integrate', 'algorithm', 'machine learning', 'optimize',
                'refactor', 'architecture', 'system', 'database',
                'api', 'framework', 'migration', 'security'
            ]
            
            description_lower = description.lower()
            matches = sum(1 for indicator in complexity_indicators if indicator in description_lower)
            text_complexity = 1 + (matches * 0.5)
        
        return min(10, hour_complexity * text_complexity)
    
    def process_deadline_features(self, deadline: str) -> Tuple[int, float]:
        """Process deadline into days_until and urgency_multiplier"""
        try:
            if pd.isna(deadline):
                return 365, 0.1  # Default to 1 year, low urgency
            
            deadline_date = pd.to_datetime(deadline)
            today = pd.Timestamp.now()
            days_until = (deadline_date - today).days
            
            # Calculate urgency multiplier based on days until deadline
            if days_until < 0:
                urgency_multiplier = 2.0  # Past due
            elif days_until <= 1:
                urgency_multiplier = 1.8  # Due today/tomorrow
            elif days_until <= 7:
                urgency_multiplier = 1.5  # Due this week
            elif days_until <= 30:
                urgency_multiplier = 1.2  # Due this month
            else:
                urgency_multiplier = 1.0  # Future deadline
            
            return max(0, days_until), urgency_multiplier
            
        except (ValueError, TypeError):
            return 365, 0.1
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features"""
        df_encoded = df.copy()
        categorical_columns = ['category', 'status']
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    # Handle missing values
                    df_encoded[col] = df_encoded[col].fillna('unknown')
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    if col in self.label_encoders:
                        df_encoded[col] = df_encoded[col].fillna('unknown')
                        # Handle unseen categories
                        unique_values = set(df_encoded[col])
                        known_values = set(self.label_encoders[col].classes_)
                        unseen_values = unique_values - known_values
                        
                        if unseen_values:
                            logger.warning(f"Unseen values in {col}: {unseen_values}")
                            # Map unseen values to 'unknown'
                            df_encoded[col] = df_encoded[col].apply(
                                lambda x: x if x in known_values else 'unknown'
                            )
                        
                        df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw task data"""
        df_features = df.copy()
        
        # Clean text fields
        for text_col in ['title', 'description']:
            if text_col in df_features.columns:
                df_features[f'{text_col}_clean'] = df_features[text_col].apply(self.clean_text)
        
        # Extract urgency from description
        if 'description' in df_features.columns:
            df_features['description_urgency'] = df_features['description'].apply(
                self.extract_urgency_indicators
            )
        
        # Calculate complexity score
        df_features['complexity_score'] = df_features.apply(
            lambda row: self.calculate_complexity_score(
                row.get('description', ''),
                row.get('estimated_hours', 1)
            ), axis=1
        )
        
        # Process deadline features
        if 'deadline' in df_features.columns:
            deadline_features = df_features['deadline'].apply(self.process_deadline_features)
            df_features['days_until_deadline'] = deadline_features.apply(lambda x: x[0])
            df_features['deadline_urgency_multiplier'] = deadline_features.apply(lambda x: x[1])
        
        # Calculate overall urgency score
        df_features['urgency_score'] = (
            df_features.get('description_urgency', 0) * 
            df_features.get('deadline_urgency_multiplier', 1)
        )
        
        # Fill missing values
        numeric_columns = ['estimated_hours', 'complexity_score', 'urgency_score']
        for col in numeric_columns:
            if col in df_features.columns:
                df_features[col] = df_features[col].fillna(df_features[col].median())
        
        return df_features
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform data"""
        logger.info("Fitting preprocessor and transforming data")
        
        # Create features
        df_processed = self.create_features(df)
        
        # Encode categorical features
        df_processed = self.encode_categorical_features(df_processed, fit=True)
        
        # Scale numerical features
        numerical_features = [
            'estimated_hours', 'complexity_score', 'urgency_score',
            'days_until_deadline', 'deadline_urgency_multiplier'
        ]
        
        existing_numerical = [col for col in numerical_features if col in df_processed.columns]
        
        if existing_numerical:
            df_processed[existing_numerical] = self.scaler.fit_transform(
                df_processed[existing_numerical]
            )
        
        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")
        
        return df_processed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.info("Transforming new data")
        
        # Create features
        df_processed = self.create_features(df)
        
        # Encode categorical features
        df_processed = self.encode_categorical_features(df_processed, fit=False)
        
        # Scale numerical features
        numerical_features = [
            'estimated_hours', 'complexity_score', 'urgency_score',
            'days_until_deadline', 'deadline_urgency_multiplier'
        ]
        
        existing_numerical = [col for col in numerical_features if col in df_processed.columns]
        
        if existing_numerical:
            df_processed[existing_numerical] = self.scaler.transform(
                df_processed[existing_numerical]
            )
        
        return df_processed


def load_and_preprocess_task_data(file_path: str = None, db_path: str = "ai_task_management.db") -> pd.DataFrame:
    """Load and preprocess task data from database or file"""
    if file_path:
        # Load from CSV file
        df = pd.read_csv(file_path)
    else:
        # Load from database
        import sqlite3
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM tasks", conn)
        conn.close()
    
    preprocessor = TaskDataPreprocessor()
    df_processed = preprocessor.fit_transform(df)
    
    return df_processed, preprocessor 