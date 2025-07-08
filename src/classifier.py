"""
Task Classification Module for AI Task Management System
Classifies tasks into categories and types automatically
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import logging
from typing import Dict, List, Any, Tuple, Optional
import re
import os

logger = logging.getLogger(__name__)


class TaskClassifier:
    """Classifier for automatically categorizing tasks"""
    
    def __init__(self, model_type: str = 'naive_bayes'):
        """
        Initialize task classifier
        
        Args:
            model_type: Type of classifier ('naive_bayes', 'random_forest', 'logistic_regression')
        """
        self.model_type = model_type
        self.pipeline = None
        self.is_trained = False
        self.categories = None
        
        # Initialize model pipeline
        self._create_pipeline()
    
    def _create_pipeline(self):
        """Create sklearn pipeline with vectorizer and classifier"""
        
        # Text vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,
            max_df=0.95
        )
        
        # Choose classifier
        if self.model_type == 'naive_bayes':
            classifier = MultinomialNB(alpha=0.1)
        elif self.model_type == 'random_forest':
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'logistic_regression':
            classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for classification"""
        if pd.isna(text) or not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, df: pd.DataFrame) -> List[str]:
        """Extract text features for classification"""
        features = []
        
        for _, row in df.iterrows():
            # Combine title and description
            title = self.preprocess_text(str(row.get('title', '')))
            description = self.preprocess_text(str(row.get('description', '')))
            
            # Combine text features
            combined_text = f"{title} {description}".strip()
            features.append(combined_text)
        
        return features
    
    def create_sample_training_data(self, n_samples: int = 1000) -> Tuple[List[str], List[str]]:
        """Create sample training data with realistic task examples"""
        
        # Define categories and their keywords/patterns
        category_patterns = {
            'bug': [
                'fix bug', 'error handling', 'debug issue', 'resolve crash', 
                'memory leak', 'performance issue', 'null pointer', 'exception',
                'broken functionality', 'not working', 'fix problem'
            ],
            'feature': [
                'implement feature', 'add functionality', 'new requirement',
                'enhance user experience', 'create module', 'build component',
                'develop feature', 'add support for', 'implement API'
            ],
            'security': [
                'security vulnerability', 'authentication', 'authorization',
                'encrypt data', 'security audit', 'penetration testing',
                'secure endpoint', 'access control', 'security patch'
            ],
            'maintenance': [
                'code cleanup', 'refactor code', 'update dependencies',
                'maintenance task', 'optimize performance', 'remove deprecated',
                'update documentation', 'clean database', 'backup system'
            ],
            'design': [
                'UI design', 'user interface', 'improve UX', 'design mockup',
                'visual design', 'responsive design', 'redesign page',
                'design system', 'user experience', 'wireframe'
            ],
            'documentation': [
                'write documentation', 'update readme', 'create guide',
                'document API', 'user manual', 'technical documentation',
                'code comments', 'wiki page', 'help documentation'
            ]
        }
        
        # Generate synthetic training data
        texts = []
        labels = []
        
        samples_per_category = n_samples // len(category_patterns)
        
        for category, patterns in category_patterns.items():
            for i in range(samples_per_category):
                # Randomly combine patterns to create realistic task descriptions
                pattern1 = np.random.choice(patterns)
                pattern2 = np.random.choice(patterns)
                
                # Create variations
                if np.random.random() > 0.5:
                    text = f"{pattern1} in the user management system"
                else:
                    text = f"{pattern1} and {pattern2} for better performance"
                
                # Add some variation
                if 'system' not in text and np.random.random() > 0.7:
                    text += " in the application"
                
                texts.append(text)
                labels.append(category)
        
        return texts, labels
    
    def train(self, df: pd.DataFrame, target_column: str = 'category') -> Dict[str, Any]:
        """
        Train the task classifier
        
        Args:
            df: Training dataframe
            target_column: Column containing category labels
        
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_type} task classifier")
        
        # Extract features
        X = self.extract_features(df)
        
        # Check if we have target labels
        if target_column in df.columns and not df[target_column].isnull().all():
            y = df[target_column].values
        else:
            # Create sample training data if no labels available
            logger.info("No category labels found, creating sample training data")
            X, y = self.create_sample_training_data()
        
        # Store categories
        self.categories = list(set(y))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train pipeline
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5, scoring='accuracy')
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'categories': self.categories
        }
        
        logger.info(f"Model training completed. Accuracy: {accuracy:.3f}")
        
        return metrics
    
    def predict(self, texts: List[str]) -> List[str]:
        """Predict categories for new texts"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Make predictions
        predictions = self.pipeline.predict(processed_texts)
        
        return predictions.tolist()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Get probabilities
        probabilities = self.pipeline.predict_proba(processed_texts)
        
        return probabilities
    
    def classify_task(self, title: str, description: str = "") -> Dict[str, Any]:
        """
        Classify a single task and return detailed results
        
        Args:
            title: Task title
            description: Task description
        
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before classification")
        
        # Combine title and description
        combined_text = f"{title} {description}".strip()
        
        # Get prediction and probabilities
        prediction = self.predict([combined_text])[0]
        probabilities = self.predict_proba([combined_text])[0]
        
        # Get class labels
        class_labels = self.pipeline.classes_
        
        # Create probability dictionary
        prob_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities)}
        
        # Get confidence (max probability)
        confidence = float(max(probabilities))
        
        result = {
            'predicted_category': prediction,
            'confidence': confidence,
            'probabilities': prob_dict,
            'text_analyzed': combined_text[:100] + "..." if len(combined_text) > 100 else combined_text
        }
        
        return result
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """Get most important features for each category"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        vectorizer = self.pipeline.named_steps['vectorizer']
        classifier = self.pipeline.named_steps['classifier']
        
        feature_names = vectorizer.get_feature_names_out()
        
        importance_dict = {}
        
        if hasattr(classifier, 'feature_log_prob_'):
            # Naive Bayes
            for i, category in enumerate(classifier.classes_):
                # Get log probabilities for this class
                log_probs = classifier.feature_log_prob_[i]
                
                # Get top features
                top_indices = log_probs.argsort()[-top_n:][::-1]
                top_features = [(feature_names[idx], float(log_probs[idx])) 
                              for idx in top_indices]
                
                importance_dict[category] = top_features
        
        elif hasattr(classifier, 'coef_'):
            # Logistic Regression
            for i, category in enumerate(classifier.classes_):
                coefs = classifier.coef_[i] if len(classifier.classes_) > 2 else classifier.coef_[0]
                
                # Get top features
                top_indices = np.abs(coefs).argsort()[-top_n:][::-1]
                top_features = [(feature_names[idx], float(coefs[idx])) 
                              for idx in top_indices]
                
                importance_dict[category] = top_features
        
        elif hasattr(classifier, 'feature_importances_'):
            # Random Forest
            importances = classifier.feature_importances_
            
            # Get top features overall
            top_indices = importances.argsort()[-top_n:][::-1]
            top_features = [(feature_names[idx], float(importances[idx])) 
                          for idx in top_indices]
            
            importance_dict['overall'] = top_features
        
        return importance_dict
    
    def save_model(self, file_path: str = "models/classifier.pkl") -> None:
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'pipeline': self.pipeline,
            'model_type': self.model_type,
            'categories': self.categories,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str = "models/classifier.pkl") -> None:
        """Load trained model from file"""
        try:
            model_data = joblib.load(file_path)
            
            self.pipeline = model_data['pipeline']
            self.model_type = model_data['model_type']
            self.categories = model_data['categories']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {file_path}")
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def bulk_classify_tasks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify multiple tasks and return dataframe with predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before classification")
        
        # Extract text features
        texts = self.extract_features(df)
        
        # Get predictions
        predictions = self.predict(texts)
        probabilities = self.predict_proba(texts)
        
        # Create results dataframe
        results_df = df.copy()
        results_df['predicted_category'] = predictions
        
        # Add confidence scores
        confidences = [max(prob_row) for prob_row in probabilities]
        results_df['prediction_confidence'] = confidences
        
        # Add probability columns for each category
        for i, category in enumerate(self.pipeline.classes_):
            results_df[f'prob_{category}'] = probabilities[:, i]
        
        return results_df


def evaluate_classification_model(df: pd.DataFrame, target_column: str = 'category') -> Dict[str, Any]:
    """Evaluate different classification models and return comparison"""
    
    models = ['naive_bayes', 'random_forest', 'logistic_regression']
    results = {}
    
    for model_type in models:
        logger.info(f"Evaluating {model_type}")
        
        try:
            classifier = TaskClassifier(model_type=model_type)
            metrics = classifier.train(df, target_column)
            
            results[model_type] = {
                'accuracy': metrics['accuracy'],
                'cv_accuracy_mean': metrics['cv_accuracy_mean'],
                'cv_accuracy_std': metrics['cv_accuracy_std'],
                'classification_report': metrics['classification_report']
            }
        
        except Exception as e:
            logger.error(f"Error evaluating {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    return results


def auto_categorize_tasks(df: pd.DataFrame, model_path: str = None) -> pd.DataFrame:
    """Automatically categorize tasks in a dataframe"""
    
    classifier = TaskClassifier()
    
    if model_path and os.path.exists(model_path):
        # Load existing model
        classifier.load_model(model_path)
    else:
        # Train new model
        logger.info("Training new classification model")
        classifier.train(df)
        
        # Save model
        if model_path:
            classifier.save_model(model_path)
    
    # Classify tasks
    categorized_df = classifier.bulk_classify_tasks(df)
    
    return categorized_df 