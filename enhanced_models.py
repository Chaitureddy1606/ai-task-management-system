#!/usr/bin/env python3
"""
Enhanced AI Models for Task Management System
- Enhanced TF-IDF + Logistic Regression for Classification
- XGBoost for Status Prediction
- Collaborative Filtering for Recommendations
- Random Forest for Auto-assignment
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, r2_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTaskClassifier:
    """Enhanced TF-IDF + Logistic Regression classifier"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.label_encoder = LabelEncoder()
        
    def train(self, texts, labels):
        """Train enhanced classifier"""
        logger.info(f"Training enhanced classifier with {len(texts)} samples")
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Transform text to TF-IDF features
        X = self.tfidf.fit_transform(texts)
        
        # Train classifier
        self.classifier.fit(X, y)
        
        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X, y, cv=5)
        accuracy = cv_scores.mean()
        
        logger.info(f"Enhanced classifier training completed. Accuracy: {accuracy:.3f}")
        return {"accuracy": accuracy, "cv_scores": cv_scores}
    
    def predict(self, text):
        """Predict task category"""
        X = self.tfidf.transform([text])
        prediction = self.classifier.predict(X)[0]
        return self.label_encoder.inverse_transform([prediction])[0]
    
    def predict_proba(self, text):
        """Get prediction probabilities"""
        X = self.tfidf.transform([text])
        probabilities = self.classifier.predict_proba(X)[0]
        return dict(zip(self.label_encoder.classes_, probabilities))
    
    def save_model(self, filepath):
        """Save enhanced classifier"""
        joblib.dump({
            'tfidf': self.tfidf,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder
        }, filepath)
        logger.info(f"Enhanced classifier saved to {filepath}")
    
    def load_model(self, filepath):
        """Load enhanced classifier"""
        data = joblib.load(filepath)
        self.tfidf = data['tfidf']
        self.classifier = data['classifier']
        self.label_encoder = data['label_encoder']
        logger.info(f"Enhanced classifier loaded from {filepath}")

class XGBoostStatusPredictor:
    """XGBoost-based status prediction model"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            'urgency_score', 'complexity_score', 'days_until_deadline',
            'business_impact', 'estimated_hours', 'deadline_urgency_multiplier'
        ]
        
    def prepare_features(self, df):
        """Prepare features for status prediction"""
        # Handle missing columns
        available_columns = [col for col in self.feature_columns if col in df.columns]
        features = df[available_columns].copy()
        
        # Add missing columns with default values
        for col in self.feature_columns:
            if col not in df.columns:
                if col == 'business_impact':
                    features[col] = 5.0  # Default business impact
                elif col == 'estimated_hours':
                    features[col] = 8.0  # Default estimated hours
                elif col == 'deadline_urgency_multiplier':
                    # Calculate from days_until_deadline
                    features[col] = df['days_until_deadline'].apply(
                        lambda x: 10 if x < 0 else (9 if x == 0 else (8 if x <= 3 else (6 if x <= 7 else (4 if x <= 30 else 2))))
                    )
                else:
                    features[col] = 0.0
        
        # Add derived features
        if 'priority' in df.columns:
            features['priority_encoded'] = df['priority'].map({'low': 1, 'medium': 2, 'high': 3})
        else:
            features['priority_encoded'] = 2  # Default to medium
            
        if 'category' in df.columns:
            features['category_encoded'] = df['category'].astype('category').cat.codes
        else:
            features['category_encoded'] = 0  # Default category
        
        # Add interaction features
        features['urgency_complexity'] = features['urgency_score'] * features['complexity_score']
        features['deadline_urgency'] = features['days_until_deadline'] * features['deadline_urgency_multiplier']
        features['business_impact_hours'] = features['business_impact'] * features['estimated_hours']
        
        return features
    
    def train(self, df, test_size=0.2):
        """Train XGBoost status predictor"""
        logger.info("Training XGBoost status predictor")
        
        # Prepare features and labels
        X = self.prepare_features(df)
        y = self.label_encoder.fit_transform(df['status'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train XGBoost with optimized parameters
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"XGBoost training completed. Accuracy: {accuracy:.3f}")
        return {"accuracy": accuracy}
    
    def predict(self, task_data):
        """Predict task status"""
        features = self.prepare_features(pd.DataFrame([task_data]))
        prediction = self.model.predict(features)[0]
        return self.label_encoder.inverse_transform([prediction])[0]
    
    def predict_proba(self, task_data):
        """Get status prediction probabilities"""
        features = self.prepare_features(pd.DataFrame([task_data]))
        probabilities = self.model.predict_proba(features)[0]
        return dict(zip(self.label_encoder.classes_, probabilities))
    
    def save_model(self, filepath):
        """Save XGBoost model"""
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }, filepath)
        logger.info(f"XGBoost model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load XGBoost model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.feature_columns = data['feature_columns']
        logger.info(f"XGBoost model loaded from {filepath}")

class CollaborativeFilteringRecommender:
    """Collaborative filtering for employee-task recommendations"""
    
    def __init__(self):
        self.user_item_matrix = None
        self.employee_embeddings = None
        self.task_embeddings = None
        self.employee_ids = None
        
    def create_user_item_matrix(self, df):
        """Create user-item interaction matrix"""
        # Create employee-task interaction matrix
        interaction_matrix = df.groupby(['assigned_to', 'category']).size().unstack(fill_value=0)
        
        # Add task complexity and priority as features
        task_features = df.groupby('assigned_to').agg({
            'complexity_score': 'mean',
            'urgency_score': 'mean',
            'priority': lambda x: x.map({'low': 1, 'medium': 2, 'high': 3}).mean(),
            'business_impact': 'mean',
            'estimated_hours': 'mean'
        }).fillna(0)
        
        # Combine interaction matrix with task features
        self.user_item_matrix = pd.concat([interaction_matrix, task_features], axis=1)
        self.employee_ids = self.user_item_matrix.index.tolist()
        
        return self.user_item_matrix
    
    def train(self, df, n_factors=50):
        """Train collaborative filtering model"""
        logger.info("Training collaborative filtering recommender")
        
        # Create user-item matrix
        self.create_user_item_matrix(df)
        
        # Simple matrix factorization (SVD-like approach)
        matrix = self.user_item_matrix.values
        
        # Normalize matrix
        matrix_norm = (matrix - matrix.mean()) / matrix.std()
        
        # Simple SVD decomposition
        U, S, Vt = np.linalg.svd(matrix_norm, full_matrices=False)
        
        # Create embeddings
        self.employee_embeddings = U[:, :n_factors]
        self.task_embeddings = Vt[:n_factors, :].T
        
        logger.info("Collaborative filtering training completed")
        return {"n_factors": n_factors, "matrix_shape": matrix.shape}
    
    def get_recommendations(self, task_features, top_k=3):
        """Get employee recommendations for a task"""
        # Create task embedding with proper dimensions
        task_vector = np.array(task_features).reshape(1, -1)
        
        # Ensure task vector matches the embedding dimensions
        if task_vector.shape[1] != self.employee_embeddings.shape[1]:
            # Pad or truncate to match dimensions
            if task_vector.shape[1] < self.employee_embeddings.shape[1]:
                # Pad with zeros
                padding = np.zeros((1, self.employee_embeddings.shape[1] - task_vector.shape[1]))
                task_vector = np.hstack([task_vector, padding])
            else:
                # Truncate
                task_vector = task_vector[:, :self.employee_embeddings.shape[1]]
        
        # Calculate similarities with all employees
        similarities = np.dot(self.employee_embeddings, task_vector.T).flatten()
        
        # Get top-k recommendations
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            employee_id = self.employee_ids[idx]
            score = similarities[idx]
            recommendations.append({
                'employee_id': employee_id,
                'score': float(score)
            })
        
        return recommendations
    
    def save_model(self, filepath):
        """Save collaborative filtering model"""
        joblib.dump({
            'employee_embeddings': self.employee_embeddings,
            'task_embeddings': self.task_embeddings,
            'employee_ids': self.employee_ids,
            'user_item_matrix': self.user_item_matrix
        }, filepath)
        logger.info(f"Collaborative filtering model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load collaborative filtering model"""
        data = joblib.load(filepath)
        self.employee_embeddings = data['employee_embeddings']
        self.task_embeddings = data['task_embeddings']
        self.employee_ids = data['employee_ids']
        self.user_item_matrix = data['user_item_matrix']
        logger.info(f"Collaborative filtering model loaded from {filepath}")

class EnhancedTaskManager:
    """Enhanced task manager with sophisticated ML models"""
    
    def __init__(self):
        self.enhanced_classifier = EnhancedTaskClassifier()
        self.xgboost_predictor = XGBoostStatusPredictor()
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        self.random_forest_assigner = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            random_state=42
        )
        
    def train_all_models(self, df):
        """Train all enhanced models"""
        logger.info("Training all enhanced models...")
        
        results = {}
        
        # Train enhanced classifier
        logger.info("Training enhanced classifier...")
        classifier_results = self.enhanced_classifier.train(
            df['description'] + " " + df['title'],
            df['category']
        )
        results['enhanced_classifier'] = classifier_results
        
        # Train XGBoost status predictor
        logger.info("Training XGBoost status predictor...")
        xgb_results = self.xgboost_predictor.train(df)
        results['xgboost_predictor'] = xgb_results
        
        # Train collaborative filtering
        logger.info("Training collaborative filtering...")
        cf_results = self.collaborative_recommender.train(df)
        results['collaborative_filtering'] = cf_results
        
        # Train Random Forest for assignment
        logger.info("Training Random Forest assigner...")
        X_assign = df[['urgency_score', 'complexity_score', 'days_until_deadline', 
                      'business_impact', 'estimated_hours']]
        y_assign = df['assigned_to']
        self.random_forest_assigner.fit(X_assign, y_assign)
        results['random_forest_assigner'] = {"trained": True}
        
        logger.info("All enhanced models trained successfully!")
        return results
    
    def process_task(self, task_data):
        """Process task with enhanced models"""
        # Enhanced classification
        category = self.enhanced_classifier.predict(
            task_data['description'] + " " + task_data['title']
        )
        category_proba = self.enhanced_classifier.predict_proba(
            task_data['description'] + " " + task_data['title']
        )
        
        # XGBoost status prediction
        status = self.xgboost_predictor.predict(task_data)
        status_proba = self.xgboost_predictor.predict_proba(task_data)
        
        # Random Forest assignment
        assignment_features = [
            task_data['urgency_score'],
            task_data['complexity_score'],
            task_data['days_until_deadline'],
            task_data.get('business_impact', 5.0),
            task_data.get('estimated_hours', 8.0)
        ]
        assigned_employee = self.random_forest_assigner.predict([assignment_features])[0]
        
        # Collaborative filtering recommendations
        task_features = [
            task_data['urgency_score'],
            task_data['complexity_score'],
            task_data['days_until_deadline']
        ]
        recommendations = self.collaborative_recommender.get_recommendations(task_features)
        
        return {
            'category': category,
            'category_confidence': max(category_proba.values()),
            'status': status,
            'status_confidence': max(status_proba.values()),
            'assigned_employee': assigned_employee,
            'recommendations': recommendations,
            'category_probabilities': category_proba,
            'status_probabilities': status_proba
        }
    
    def save_all_models(self, base_path="models/enhanced/"):
        """Save all enhanced models"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        self.enhanced_classifier.save_model(f"{base_path}enhanced_classifier.pkl")
        self.xgboost_predictor.save_model(f"{base_path}xgboost_predictor.pkl")
        self.collaborative_recommender.save_model(f"{base_path}collaborative_filtering.pkl")
        joblib.dump(self.random_forest_assigner, f"{base_path}random_forest_assigner.pkl")
        
        logger.info("All enhanced models saved")
    
    def load_all_models(self, base_path="models/enhanced/"):
        """Load all enhanced models"""
        self.enhanced_classifier.load_model(f"{base_path}enhanced_classifier.pkl")
        self.xgboost_predictor.load_model(f"{base_path}xgboost_predictor.pkl")
        self.collaborative_recommender.load_model(f"{base_path}collaborative_filtering.pkl")
        self.random_forest_assigner = joblib.load(f"{base_path}random_forest_assigner.pkl")
        
        logger.info("All enhanced models loaded")

def train_enhanced_models():
    """Train all enhanced models with the combined dataset"""
    # Load data
    df = pd.read_csv('data/combined_training_tasks.csv')
    
    # Initialize enhanced task manager
    enhanced_manager = EnhancedTaskManager()
    
    # Train all models
    results = enhanced_manager.train_all_models(df)
    
    # Save models
    enhanced_manager.save_all_models()
    
    # Print results
    print("\n🎯 Enhanced Models Training Results:")
    print("=" * 50)
    print(f"Enhanced Classifier Accuracy: {results['enhanced_classifier']['accuracy']:.3f}")
    print(f"XGBoost Status Predictor Accuracy: {results['xgboost_predictor']['accuracy']:.3f}")
    print(f"Collaborative Filtering Factors: {results['collaborative_filtering']['n_factors']}")
    print(f"Random Forest Assigner: Trained")
    
    # Test the enhanced system
    test_task = {
        'title': 'Fix critical payment bug',
        'description': 'Users unable to complete credit card payments',
        'urgency_score': 9,
        'complexity_score': 6,
        'days_until_deadline': 2,
        'business_impact': 8.5,
        'estimated_hours': 12,
        'priority': 'high',
        'category': 'bug'
    }
    
    result = enhanced_manager.process_task(test_task)
    print(f"\n🧪 Test Result:")
    print(f"Category: {result['category']} (confidence: {result['category_confidence']:.3f})")
    print(f"Status: {result['status']} (confidence: {result['status_confidence']:.3f})")
    print(f"Assigned Employee: {result['assigned_employee']}")
    print(f"Top Recommendations: {result['recommendations'][:3]}")
    
    return enhanced_manager

if __name__ == "__main__":
    train_enhanced_models() 