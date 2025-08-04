#!/usr/bin/env python3
"""
Advanced AI Models for Task Management System
- BERT for Task Classification
- XGBoost for Status Prediction
- Collaborative Filtering for Recommendations
- Enhanced Auto-assignment with Random Forest
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
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTTaskClassifier:
    """BERT-based task classification model"""
    
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, texts, labels):
        """Prepare data for BERT training"""
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        return encodings, torch.tensor(encoded_labels)
    
    def train(self, texts, labels, epochs=3, batch_size=16):
        """Train BERT classifier"""
        logger.info(f"Training BERT classifier with {len(texts)} samples")
        
        # Prepare data
        encodings, labels_tensor = self.prepare_data(texts, labels)
        
        # Initialize model
        self.model = AutoModel.from_pretrained(self.model_name)
        num_labels = len(self.label_encoder.classes_)
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.to(self.device)
        self.classifier.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**encodings)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
            logits = self.classifier(pooled_output)
            loss = criterion(logits, labels_tensor.to(self.device))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        logger.info("BERT training completed")
        return {"accuracy": self._evaluate(texts, labels)}
    
    def _evaluate(self, texts, labels):
        """Evaluate BERT model"""
        self.model.eval()
        self.classifier.eval()
        
        encodings, labels_tensor = self.prepare_data(texts, labels)
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(pooled_output)
            predictions = torch.argmax(logits, dim=1)
        
        accuracy = accuracy_score(labels_tensor, predictions.cpu())
        return accuracy
    
    def predict(self, text):
        """Predict task category"""
        self.model.eval()
        self.classifier.eval()
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(pooled_output)
            prediction = torch.argmax(logits, dim=1)
        
        return self.label_encoder.inverse_transform(prediction.cpu())[0]
    
    def save_model(self, filepath):
        """Save BERT model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'label_encoder': self.label_encoder,
            'tokenizer': self.tokenizer
        }, filepath)
        logger.info(f"BERT model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load BERT model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.label_encoder = checkpoint['label_encoder']
        self.tokenizer = checkpoint['tokenizer']
        logger.info(f"BERT model loaded from {filepath}")

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
        features = df[self.feature_columns].copy()
        
        # Add derived features
        features['priority_encoded'] = df['priority'].map({'low': 1, 'medium': 2, 'high': 3})
        features['category_encoded'] = df['category'].astype('category').cat.codes
        
        # Add interaction features
        features['urgency_complexity'] = features['urgency_score'] * features['complexity_score']
        features['deadline_urgency'] = features['days_until_deadline'] * features['deadline_urgency_multiplier']
        
        return features
    
    def train(self, df, test_size=0.2):
        """Train XGBoost status predictor"""
        logger.info("Training XGBoost status predictor")
        
        # Prepare features and labels
        X = self.prepare_features(df)
        y = self.label_encoder.fit_transform(df['status'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
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
        self.task_features = None
        
    def create_user_item_matrix(self, df):
        """Create user-item interaction matrix"""
        # Create employee-task interaction matrix
        interaction_matrix = df.groupby(['assigned_to', 'category']).size().unstack(fill_value=0)
        
        # Add task complexity and priority as features
        task_features = df.groupby('assigned_to').agg({
            'complexity_score': 'mean',
            'urgency_score': 'mean',
            'priority': lambda x: x.map({'low': 1, 'medium': 2, 'high': 3}).mean()
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
        # Create task embedding
        task_vector = np.array(task_features).reshape(1, -1)
        
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

class AdvancedTaskManager:
    """Advanced task manager with sophisticated ML models"""
    
    def __init__(self):
        self.bert_classifier = BERTTaskClassifier()
        self.xgboost_predictor = XGBoostStatusPredictor()
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        self.random_forest_assigner = RandomForestClassifier(random_state=42)
        
    def train_all_models(self, df):
        """Train all advanced models"""
        logger.info("Training all advanced models...")
        
        results = {}
        
        # Train BERT classifier
        logger.info("Training BERT classifier...")
        bert_results = self.bert_classifier.train(
            df['description'] + " " + df['title'],
            df['category']
        )
        results['bert_classifier'] = bert_results
        
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
        X_assign = df[['urgency_score', 'complexity_score', 'days_until_deadline']]
        y_assign = df['assigned_to']
        self.random_forest_assigner.fit(X_assign, y_assign)
        results['random_forest_assigner'] = {"trained": True}
        
        logger.info("All advanced models trained successfully!")
        return results
    
    def process_task(self, task_data):
        """Process task with advanced models"""
        # BERT classification
        category = self.bert_classifier.predict(
            task_data['description'] + " " + task_data['title']
        )
        
        # XGBoost status prediction
        status = self.xgboost_predictor.predict(task_data)
        
        # Random Forest assignment
        assignment_features = [
            task_data['urgency_score'],
            task_data['complexity_score'],
            task_data['days_until_deadline']
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
            'status': status,
            'assigned_employee': assigned_employee,
            'recommendations': recommendations
        }
    
    def save_all_models(self, base_path="models/advanced/"):
        """Save all advanced models"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        self.bert_classifier.save_model(f"{base_path}bert_classifier.pt")
        self.xgboost_predictor.save_model(f"{base_path}xgboost_predictor.pkl")
        self.collaborative_recommender.save_model(f"{base_path}collaborative_filtering.pkl")
        joblib.dump(self.random_forest_assigner, f"{base_path}random_forest_assigner.pkl")
        
        logger.info("All advanced models saved")
    
    def load_all_models(self, base_path="models/advanced/"):
        """Load all advanced models"""
        self.bert_classifier.load_model(f"{base_path}bert_classifier.pt")
        self.xgboost_predictor.load_model(f"{base_path}xgboost_predictor.pkl")
        self.collaborative_recommender.load_model(f"{base_path}collaborative_filtering.pkl")
        self.random_forest_assigner = joblib.load(f"{base_path}random_forest_assigner.pkl")
        
        logger.info("All advanced models loaded")

def train_advanced_models():
    """Train all advanced models with the combined dataset"""
    # Load data
    df = pd.read_csv('data/combined_training_tasks.csv')
    
    # Initialize advanced task manager
    advanced_manager = AdvancedTaskManager()
    
    # Train all models
    results = advanced_manager.train_all_models(df)
    
    # Save models
    advanced_manager.save_all_models()
    
    # Print results
    print("\n🎯 Advanced Models Training Results:")
    print("=" * 50)
    print(f"BERT Classifier Accuracy: {results['bert_classifier']['accuracy']:.3f}")
    print(f"XGBoost Status Predictor Accuracy: {results['xgboost_predictor']['accuracy']:.3f}")
    print(f"Collaborative Filtering Factors: {results['collaborative_filtering']['n_factors']}")
    print(f"Random Forest Assigner: Trained")
    
    return advanced_manager

if __name__ == "__main__":
    train_advanced_models() 